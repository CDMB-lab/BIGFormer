# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     parameters
   Description :
   Author :       zouqi
   date：          2023/2/17
-------------------------------------------------
   Change Activity:
                   2023/2/17:
-------------------------------------------------
"""
__author__ = 'zouqi'

import argparse


def parameters_parser():
    parser = argparse.ArgumentParser(description='BIGFormer')

    ######## Common ########
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_decay_step_size', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=1e-1)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--work-dir', type=str, default='')
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--model-path', type=str, default='models')
    parser.add_argument('--logger', default='promote', choices=['epoch', 'promote'])
    parser.add_argument('--log-path', default='logs')
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--save', action='store_true')
    ######## Common ########

    ######## dataset ########
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--name', default='Dual', choices=['Gene', 'fMRI', 'Dual'])
    parser.add_argument('--temp', default='aal116', choices=['aal116', 'bna246'])
    # 0: HC vs. AD; 1: HC vs. MCI; 2: MCI vs. AD; 3: sMCI vs. pMCI
    parser.add_argument('--task', default=0, type=int, choices=[0, 1, 2, 3])
    parser.add_argument('--corr', default='pearson', choices=['pearson', 'spearman', 'kendall'])
    parser.add_argument('--cutoff-model', default='Radius', choices=['KNN', 'Radius'])
    parser.add_argument('--cutoff-arg', type=float, default=0.7)
    parser.add_argument('--feat-str', type=str, default='')
    ######## dataset ########

    ####### models #######
    parser.add_argument('--model', default='BF', choices=['MLP', 'GCN', 'PLSA', 'VT', 'BF'])
    parser.add_argument('--num-PLSA', default=0, type=int)
    parser.add_argument('--num-layers', default=1, type=int)

    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--conv', type=str, default='PLSA-T', choices=['GNN-T', 'PLSA-T', 'VT'])
    parser.add_argument('--norm-type', type=str, default='sw', choices=['sw', 'pre', 'post', 'none'])
    parser.add_argument('--attention-type', type=str,  default='BL', choices=['V', 'B', 'BL'])
    ####### models #######

    return parser.parse_args()
