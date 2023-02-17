# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     dataset
   Description :
   Author :       zouqi
   date：          2023/2/17
-------------------------------------------------
   Change Activity:
                   2023/2/17:
-------------------------------------------------
"""
__author__ = 'zouqi'

import re
from os.path import join as opj
from typing import Union, List, Tuple
import warnings

warnings.filterwarnings('ignore')
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, kendalltau
from torch_geometric.data import InMemoryDataset, Data

from transform import radius_graph, knn_graph, FeatureExpander


class BIGDataset(InMemoryDataset):
    def __init__(self, root: str, name='Dual',
                 temp='aal116',
                 corr='pearson', cutoff_model='Radius', cutoff_arg: Union[int, float] = 0,
                 task=0, feat_str='', model='GCN',
                 pre_transform=None, pre_filter=None):
        assert (corr in ['pearson', 'spearman', 'kendall'])
        assert (cutoff_model in ['Radius', 'KNN'])
        assert (task in range(10))
        self.name = name
        self.temp = temp
        self.corr = corr
        self.cutoff_model = cutoff_model
        if self.cutoff_model == 'KNN':
            cutoff_arg = int(cutoff_arg)
        self.cutoff_arg = cutoff_arg

        self.task = task
        self.feat_str = feat_str
        self.model = model

        super(BIGDataset, self).__init__(root, pre_transform=pre_transform, pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return opj(self.root, 'raw_data')

    @property
    def processed_dir(self) -> str:
        return opj(self.root, 'processed', f'{self.name}-{self.temp}-{self.corr}'
                                           f'-{self.cutoff_model}-{self.cutoff_arg}'
                                           f'-{self.task}-{self.feat_str}-{self.model}')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return ['gene', 'roi']

    @property
    def num_node_attributes(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.size(1)

    @property
    def num_edge_attributes(self) -> int:
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1)

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return 'data.pt'

    @property
    def num_classes(self) -> int:
        if self.task in [0, 1, 2, 3, 4]:
            return 2

    @property
    def num_nodes(self) -> int:
        num_gene = 34
        num_roi = 116 if self.temp == 'aal116' else 246

        if self.name == 'Gene':
            return num_gene
        elif self.name == 'ROI':
            return num_roi
        else:
            return num_gene + num_roi

    def process(self):
        raw_gene, raw_roi = self.raw_paths
        subjects = pd.read_csv(opj(self.root, f'task{self.task}.csv'))
        datalist = []
        cnt = 0
        for subject in tqdm(subjects['Subject ID']):
            if self.name == 'Gene':
                gene = np.loadtxt(opj(raw_gene, 'loc', f'{subject}.txt'))
                feat = gene
            elif self.name == 'ROI':
                roi = np.loadtxt(opj(raw_roi, self.temp, f'{subject}.txt'))
                feat = roi[:, :90]

            elif self.name == 'Dual':
                gene = np.loadtxt(opj(raw_gene, 'loc', f'{subject}.txt'))
                roi = np.loadtxt(opj(raw_roi, self.temp, f'{subject}.txt'))
                feat = np.concatenate([gene, roi[:, :90]], axis=0)
            else:
                raise NotImplemented(self.name)

            feat = torch.from_numpy(feat).float()
            mean = feat.mean(0, keepdim=True)
            std = feat.std(0, keepdim=True)
            feat = (feat - mean) / (std + 1e-5)

            if self.corr == 'pearson':
                corr = np.corrcoef(feat)
            elif self.corr == 'spearman':
                corr, _ = spearmanr(feat, axis=1)
            elif self.corr == 'kendall':
                corr = np.eye(feat.shape[0])
                for i in range(corr.shape[0]):
                    for j in range(i, corr.shape[1]):
                        k, _ = kendalltau(feat[i, :], feat[j, :])
                        corr[i, j] = k
                        corr[j, i] = k
            else:
                raise NotImplemented(self.corr)
            corr = torch.tensor(corr).abs()
            diag = torch.diag(corr)
            corr = corr - torch.diag_embed(diag)

            if self.name == 'Dual':
                gg_corr = corr[:34, :34]
                rr_corr = corr[34:, 34:]
                gr_corr = corr[:34, 34:]

                gg_corr -= gg_corr.min()
                gg_corr /= gg_corr.max()
                rr_corr -= rr_corr.min()
                rr_corr /= rr_corr.max()
                gr_corr -= gr_corr.min()
                gr_corr /= gr_corr.max()

                a = torch.cat([gg_corr, gr_corr], 1)
                b = torch.cat([gr_corr.t().contiguous(), rr_corr], 1)
                corr = torch.cat([a, b], 0)
            else:
                corr -= corr.min()
                corr /= corr.max()

            if self.cutoff_model == 'Radius':
                corr = radius_graph(corr, self.cutoff_arg)
            elif self.cutoff_model == 'KNN':
                corr = knn_graph(corr, self.cutoff_arg)
            else:
                raise NotImplemented(self.cutoff_model)
            corr = torch.where(torch.isnan(corr), torch.full_like(corr, 0), corr)

            edge_index = corr.nonzero().t().contiguous().long()
            edge_weight = corr[edge_index[0], edge_index[1]].float()
            label = subjects[subjects['Subject ID'] == subject]['Label'].values[0]

            data = Data(x=feat,
                        edge_index=edge_index,
                        edge_attr=edge_weight,
                        y=torch.tensor(label),
                        id=cnt)
            cnt += 1
            datalist.append(data)

        if self.pre_transform is not None:
            datalist = [self.pre_transform(g) for g in tqdm(datalist)]
        if self.pre_filter is not None:
            datalist = [g for g in tqdm(datalist) if self.pre_filter(g)]

        data, slices = self.collate(datalist)

        torch.save((data, slices), self.processed_paths[0])


def get_dataset(args, feat_str='deg+cen'):
    degree = feat_str.find("deg") >= 0
    onehot_maxdeg = re.findall("odeg(\d+)", feat_str)
    onehot_maxdeg = int(onehot_maxdeg[0]) if onehot_maxdeg else None
    centrality = feat_str.find("cen") >= 0

    num_roi = 90 if args.temp == 'aal116' else 246
    num_gene = 34
    if args.name == 'Dual':
        max_num_nodes = num_roi + num_gene
    elif args.name == 'Gene':
        max_num_nodes = num_gene
    elif args.name == 'fMRI':
        max_num_nodes = num_roi
    else:
        raise NotImplementedError(args.name)
    fe = FeatureExpander(max_node_num=max_num_nodes, degree=degree,
                         onehot_maxdeg=onehot_maxdeg, centrality=centrality).transform

    dataset = BIGDataset(root=args.root, name=args.name, temp=args.temp, corr=args.corr,
                         cutoff_model=args.cutoff_model, cutoff_arg=args.cutoff_arg,
                         task=args.task, feat_str=args.feat_str,
                         pre_transform=fe)

    return dataset, max_num_nodes