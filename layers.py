# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     layers
   Description :
   Author :       zouqi
   date：          2023/2/17
-------------------------------------------------
   Change Activity:
                   2023/2/17:
-------------------------------------------------
"""
__author__ = 'zouqi'

import math

import torch
from torch import nn
from torch.nn import functional as F, LayerNorm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros


class DenseGCNConv(nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GCNConv`.
    """

    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.lin = Linear(in_channels, out_channels, bias=False,
                             weight_initializer='glorot')

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x, adj, mask=None, add_loop=True):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        out = self.lin(x)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')


class PLSA(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.mlp = nn.Sequential(nn.Linear(input_dim, output_dim),
                                 nn.ReLU(),
                                 nn.Linear(output_dim, output_dim),
                                 nn.ReLU())

        self.linear = nn.Linear(output_dim, output_dim)

        self.eps = nn.Parameter(torch.FloatTensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv_eps = 0.1 / math.sqrt(self.eps.size(0))
        nn.init.constant_(self.eps, stdv_eps)

    def forward(self, X, A):
        """
        Params
        ------
        A [batch x nodes x nodes]: adjacency matrix
        X [batch x nodes x features]: node features matrix

        Returns
        -------
        X' [batch x nodes x features]: updated node features matrix
        """

        batch, N = A.shape[:2]
        mask = torch.eye(N).unsqueeze(0).to(A.device)
        batch_diagonal = torch.diagonal(A, 0, 1, 2)
        batch_diagonal = self.eps * batch_diagonal
        A = mask * torch.diag_embed(batch_diagonal) + (1. - mask) * A

        X = self.mlp(A @ X)
        X = self.linear(X)

        return X


class GRIC(nn.Module):
    """Blended attention block"""

    def __init__(self, dim_Q: int, dim_K: int, dim_V: int,
                 conv: str, num_nodes: int,
                 num_heads: int, norm_type: str,
                 attention_type: str = 'BL'):
        super(GRIC, self).__init__()

        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.dim_V = dim_V

        self.num_nodes = num_nodes
        self.num_heads = num_heads
        self.norm_type = norm_type

        self.conv = conv

        if self.conv == 'PLSA-T':
            self.layer_Q = PLSA(self.dim_Q, self.dim_V * self.num_heads)
            self.layer_K = PLSA(self.dim_Q, self.dim_V * self.num_heads)
            self.layer_V = PLSA(self.dim_V, self.dim_V * self.num_heads)
        elif self.conv == 'GNN-T':
            self.layer_Q = DenseGCNConv(self.dim_Q, self.dim_V * self.num_heads)
            self.layer_K = DenseGCNConv(self.dim_K, self.dim_V * self.num_heads)
            self.layer_V = DenseGCNConv(self.dim_V, self.dim_V * self.num_heads)
        elif self.conv == 'VT':
            self.layer_Q = nn.Linear(self.dim_Q, self.dim_V * self.num_heads)
            self.layer_K = nn.Linear(self.dim_K, self.dim_V * self.num_heads)
            self.layer_V = nn.Linear(self.dim_V, self.dim_V * self.num_heads)
        else:
            raise NotImplemented(self.conv)

        self.attention_type = attention_type

        if self.attention_type != 0:
            # bias attention
            self.B = nn.Parameter(torch.Tensor(self.num_heads, self.num_nodes, self.num_nodes), requires_grad=True)
        if self.attention_type == 2:
            self.alpha = nn.Parameter(torch.Tensor(1), requires_grad=True)

        self.WO = nn.Linear(self.dim_V * self.num_heads, self.dim_Q, bias=False)

        if self.norm_type != 'none':
            self.GN0 = LayerNorm(self.dim_Q)
            self.GN1 = LayerNorm(self.dim_Q)
            self.GN2 = LayerNorm(self.dim_Q)
            self.GN3 = LayerNorm(self.dim_Q)

        self.rff = nn.Sequential(
            nn.Linear(self.dim_Q, self.dim_Q),
            nn.ReLU(),
            nn.Linear(self.dim_Q, self.dim_Q)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.ndimension() >= 2:
                    nn.init.kaiming_normal_(param)
                else:
                    nn.init.kaiming_normal_(param.unsqueeze(0))
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        if self.attention_type != 'V':
            # nn.init.kaiming_normal_(self.B)
            zeros(self.B)
        if self.attention_type == 'BL':
            nn.init.constant_(self.alpha, 0.5)

    def forward(self, H, A, return_attn=False):
        residual = H

        if self.norm_type in ['pre', 'sw']:
            H = self.GN0(H)

        if self.conv == 'VT':
            Q = self.layer_Q(H)
            K = self.layer_K(H)
            V = self.layer_V(H)
        else:
            # Q.shape: [batch_size,  num_nodes, dim_V * self.num_heads]
            Q = self.layer_Q(H, A)
            # K.shape: [batch_size,  num_nodes, dim_V * self.num_heads]
            K = self.layer_K(H, A)
            # V.shape: [batch_size,  num_nodes, dim_V * self.num_heads]
            # V = self.layer_V(H, A)
            V = self.layer_V(H)

        # Q_.shape: [batch_size, num_nodes, dim_V * self.num_heads]
        Q_ = torch.stack(Q.split(self.dim_V, -1), dim=1)
        # K_.shape: [batch_size, num_nodes, dim_V * self.num_heads]
        K_ = torch.stack(K.split(self.dim_V, -1), dim=1)
        # V_.shape: [batch_size, num_nodes, dim_V * self.num_heads]
        V_ = torch.stack(V.split(self.dim_V, -1), dim=1)

        ##### Attention ####
        # self_attention.shape: [batch_size, num_heads, num_nodes, num_nodes]
        self_attention = torch.matmul(Q_, K_.permute(0, 1, 3, 2))
        self_attention /= math.sqrt(self.dim_K)

        # attention
        if self.attention_type == 'V':
            attention = self_attention
        else:
            # bias_attention.shape: [batch_size, num_heads, num_nodes, num_nodes]
            bias_attention = self.B.repeat(Q_.shape[0], 1, 1, 1)
            if self.attention_type == 'B':
                attention = bias_attention
            elif self.attention_type == 'BL':
                # attention = self.alpha * self_attention + (1 - self.alpha) * bias_attention
                attention = self_attention + bias_attention

        attention = F.softmax(attention, -1)
        ##### Attention ####

        # MH.shape: [batch_size, num_heads, num_nodes, dim_V]
        MH = torch.matmul(attention, V_)

        if self.norm_type == 'sw':
            MH = self.GN1(MH)

        # O.shape: [batch_size, num_nodes, num_heads * dim_V]
        O = self.WO(torch.cat(MH.chunk(self.num_nodes, dim=1), dim=-1).squeeze(dim=1)) + residual

        if self.norm_type != 'none':
            O = self.GN2(O)

        RO = self.rff(O).relu()

        if self.norm_type == 'sw':
            RO = self.GN3(RO)

        O = O + RO

        if self.norm_type == 'post':
            O = self.GN3(O)

        return O
