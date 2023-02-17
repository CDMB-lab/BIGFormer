# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     transform
   Description :
   Author :       zouqi
   date：          2023/2/17
-------------------------------------------------
   Change Activity:
                   2023/2/17:
-------------------------------------------------
"""
__author__ = 'zouqi'

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import degree, to_dense_adj
from torch_geometric.utils import remove_self_loops, add_self_loops


def get_dense_adj(edge_index, edge_attr=None, node_num=None):
    if not node_num:
        node_num = edge_index.max() + 1
    if edge_attr is not None:
        adj = torch.sparse.FloatTensor(edge_index, edge_attr, (node_num, node_num)).to_dense()
    else:
        adj = torch.sparse.FloatTensor(edge_index, torch.ones(edge_index.shape[1]), (node_num, node_num)).to_dense()

    return adj


def knn_graph(correlations: torch.Tensor, k: int, keep_greater: bool = True):
    """
        construction knn-graph based correlation matrix. each node has k-nearest neighbors.
        note: resulting correlation matrix is directed graph.
    :param correlations:
    :param k:
    :param keep_greater:
    :return:
    """
    assert k > 0
    k_indices = correlations.argsort(descending=keep_greater)[:, :k].long()
    knn_mask = torch.zeros(correlations.shape[0], correlations.shape[1])
    knn_mask.scatter_(1, k_indices, 1)
    # directed knn graph
    knn_corr = correlations * knn_mask
    return knn_corr


def radius_graph(correlations: torch.Tensor, r: float, keep_greater: bool = True):
    """
        construction radius-graph based correlation matrix. radius <= r
        note: resulting correlation matrix is directed graph.
    :param correlations:
    :param r:
    :param keep_greater:
    :return:
    """
    correlations = deepcopy(correlations)
    if keep_greater:
        correlations[correlations < r] = 0
    else:
        correlations[correlations > r] = 0
    return correlations


class FeatureExpander(MessagePassing):
    r"""Expand features.
    Args:
        degree (bool): whether to use degree feature.
        onehot_maxdeg (int): whether to use one_hot degree feature with
            max degree capped. disableid with 0.
        centrality (bool): whether to use centrality feature.
    """

    def __init__(self, max_node_num, degree=True, onehot_maxdeg=0, centrality=False):
        super(FeatureExpander, self).__init__('add', 'source_to_target')
        self.max_node_num = max_node_num
        self.degree = degree
        self.onehot_maxdeg = onehot_maxdeg

        self.centrality = centrality
        self.edge_norm_diag = 1e-8  # edge norm is used, and set A diag to it

    def __str__(self):
        return f'FeatureExpander(max_node_num={self.max_node_num}, degree={self.degree}, ' \
               f'onehot_maxdeg={self.onehot_maxdeg}, centrality={self.centrality})'

    def transform(self, data):
        if self.degree:
            deg, deg_onehot = self.compute_degree(data.edge_index, data.num_nodes)
            data.x = torch.cat([data.x, deg, deg_onehot], -1)
            # data.x = torch.cat([deg_onehot, deg], -1)
        if self.centrality:
            cent = self.compute_centrality(data)
            data.x = torch.cat([data.x, cent], -1)

        ### Create feature/adj/mask batch
        ##########################################################################
        features, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        adj_batch = to_dense_adj(edge_index, max_num_nodes=self.max_node_num)[0]
        adj_attr_batch = to_dense_adj(edge_index, edge_attr=edge_attr, max_num_nodes=self.max_node_num)[0].squeeze(-1)
        features_batch = self.create_features_batch(features, self.max_node_num)
        adj_batch = self.create_adj_batch(adj_batch, self.max_node_num)
        adj_attr_batch = self.create_adj_batch(adj_attr_batch, self.max_node_num)

        data.x, data.adj, data.adj_attr = features_batch, adj_batch, adj_attr_batch
        del data.edge_index
        del data.edge_attr
        ##########################################################################
        return data

    def omega(self, A):
        A_array = A.numpy()
        G = nx.from_numpy_matrix(A_array)

        sub_graphs = []
        subgraph_nodes_list = []
        sub_graphs_adj = []
        sub_graph_edges = []
        new_adj = torch.zeros(A_array.shape[0], A_array.shape[0])

        for i in np.arange(len(A_array)):
            s_indexes = []
            for j in np.arange(len(A_array)):
                s_indexes.append(i)
                if (A_array[i][j] == 1):
                    s_indexes.append(j)
            sub_graphs.append(G.subgraph(s_indexes))

        for i in np.arange(len(sub_graphs)):
            subgraph_nodes_list.append(list(sub_graphs[i].nodes))

        for index in np.arange(len(sub_graphs)):
            sub_graphs_adj.append(nx.adjacency_matrix(sub_graphs[index]).toarray())

        for index in np.arange(len(sub_graphs)):
            sub_graph_edges.append(sub_graphs[index].number_of_edges())

        for node in np.arange(len(subgraph_nodes_list)):
            sub_adj = sub_graphs_adj[node]
            for neighbors in np.arange(len(subgraph_nodes_list[node])):
                index = subgraph_nodes_list[node][neighbors]
                count = torch.tensor(0).float()
                if index == node:
                    continue
                else:
                    c_neighbors = set(subgraph_nodes_list[node]).intersection(subgraph_nodes_list[index])
                    if index in c_neighbors:
                        nodes_list = subgraph_nodes_list[node]
                        sub_graph_index = nodes_list.index(index)
                        c_neighbors_list = list(c_neighbors)
                        for i, item1 in enumerate(nodes_list):
                            if item1 in c_neighbors:
                                for item2 in c_neighbors_list:
                                    j = nodes_list.index(item2)
                                    count += sub_adj[i][j]

                    new_adj[node][index] = count / 2
                    new_adj[node][index] = new_adj[node][index] / (len(c_neighbors) * (len(c_neighbors) - 1))
                    new_adj[node][index] = new_adj[node][index] * (len(c_neighbors) ** 2)

        weight = torch.FloatTensor(new_adj)
        weight = weight / weight.sum(1, keepdim=True)

        weight = weight + torch.FloatTensor(A_array)

        coeff = weight.sum(1, keepdim=True)
        coeff = torch.diag(coeff.T[0])

        weight = weight + coeff

        weight = weight.detach().numpy()
        weight = np.nan_to_num(weight, nan=0)
        weight = torch.tensor(weight)

        return weight

    def compute_heat(self, A, t=5, self_loop=True):
        if self_loop:
            A = A + torch.eye(A.shape[0])
        D_ = torch.sum(A, 1)
        D_inv = torch.diag(torch.pow(D_, -0.5))
        return torch.exp(t * (A @ D_inv - 1))

    def compute_ppr(self, A, alpha=0.2, self_loop=True):
        if self_loop:
            A = A + torch.eye(A.shape[0])
        D_ = torch.sum(A, 1)
        D_inv = torch.diag(torch.pow(D_, -0.5))
        at = D_inv @ A @ D_inv
        S = torch.inverse(torch.eye(A.shape[0]) - (1 - alpha) * at)
        return alpha * S

    def compute_degree(self, edge_index, num_nodes):
        row, col = edge_index
        deg = degree(row, num_nodes)
        deg = deg.view((-1, 1))

        if self.onehot_maxdeg is not None and self.onehot_maxdeg > 0:
            max_deg = torch.tensor(self.onehot_maxdeg, dtype=deg.dtype)
            deg_capped = torch.min(deg, max_deg).type(torch.int64)
            deg_onehot = F.one_hot(
                deg_capped.view(-1), num_classes=self.onehot_maxdeg + 1)
            deg_onehot = deg_onehot.type(deg.dtype)
        else:
            deg_onehot = self.empty_feature(num_nodes)

        if not self.degree:
            deg = self.empty_feature(num_nodes)

        return deg, deg_onehot

    def compute_centrality(self, data):
        if not self.centrality:
            return self.empty_feature(data.num_nodes)

        G = nx.Graph(data.edge_index.numpy().T.tolist())
        G.add_nodes_from(range(data.num_nodes))  # in case missing node ids
        closeness = nx.algorithms.closeness_centrality(G)
        betweenness = nx.algorithms.betweenness_centrality(G)
        pagerank = nx.pagerank_numpy(G)
        centrality_features = torch.tensor(
            [[closeness[i], betweenness[i], pagerank[i]] for i in range(
                data.num_nodes)])
        return centrality_features

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, diag_val=1e-8, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),),
                                     dtype=dtype,
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        # Add edge_weight for loop edges.
        loop_weight = torch.full((num_nodes,),
                                 diag_val,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def empty_feature(self, num_nodes):
        return torch.zeros([num_nodes, 0])

    ### Create batch feature/adj/mask ###
    def cat_feature(self, feature, max_node_num):
        # pads the feature with zeros to the max node number
        padded_feature = nn.ConstantPad1d((0, 0, 0, max_node_num - feature.size(0)), 0)(feature)
        return padded_feature

    def create_features_batch(self, features, max_node_num):
        return self.cat_feature(features, max_node_num)

    def create_adj(self, adj, max_node_num):
        padded_adj = torch.zeros(max_node_num, max_node_num)
        padded_adj[:adj.size(0), :adj.size(1)] = adj
        return padded_adj

    def create_adj_batch(self, adj, max_node_num):
        adj_batch = self.create_adj(adj, max_node_num)
        return adj_batch

    def create_mask(self, num_node, max_node_num):
        x = torch.ones(num_node, dtype=torch.bool)
        x = nn.ConstantPad1d((0, max_node_num - num_node), 0)(x)
        return x

    def create_mask_batch(self, node_num, max_node_num):
        return self.create_mask(node_num, max_node_num)