# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     train_eval_helper.py
   Description :
   Author :       zouqi
   date：          2023/2/17
-------------------------------------------------
   Change Activity:
                   2023/2/17:
-------------------------------------------------
"""
__author__ = 'zouqi'

import time
from os.path import join as opj

import numpy as np
import torch
from sklearn.model_selection import RepeatedStratifiedKFold
from torch import nn, tensor
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score, precision_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.backward_compatibility import worker_init_fn
from torch_geometric.loader import DenseDataLoader

from dataset import get_dataset
from model import GMLP, GCN, PLSA_G, BIGFormer
from utils import num_graphs, pickle_dump


def kfold(dataset, folds, repeats=1):
    skf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    val_indices = [test_indices[(r * 10) + (f + 1) % folds] for r in range(repeats) for f in range(folds)]

    for i in range(folds*repeats):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, val_indices, test_indices


def cross_validation(args, logger=None):
    if args.gpu != -1:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    if device != 'cpu':
        torch.cuda.synchronize()

    model_path = opj(args.work_dir, args.model_path)

    dataset, num_nodes = get_dataset(args, args.feat_str)
    num_features = dataset.num_node_attributes
    num_targets = dataset.num_classes
    num_layers = args.num_layers
    num_PLSA = args.num_PLSA
    train_losses, train_accuracies, \
    test_losses, test_accuracies, \
    test_precisions, test_sensitivities, \
    test_specificities, test_aucs, \
    test_f1s, durations = [], [], [], [], [], [], [], [], [], []

    test_y_trues, test_pred_outs = [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*kfold(dataset, args.folds, args.repeats))):
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        train_loader = DenseDataLoader(train_dataset, batch_size=args.batch_size,
                                       shuffle=True,
                                       worker_init_fn=worker_init_fn,
                                       num_workers=0)
        test_loader = DenseDataLoader(test_dataset, batch_size=args.batch_size,
                                      worker_init_fn=worker_init_fn,
                                      num_workers=0)
        if args.model == 'MLP':
            model = GMLP(in_dim=num_features, in_units=num_nodes, num_layers=num_layers, num_targets=num_targets)
        elif args.model == 'GCN':
            model = GCN(in_dim=num_features, in_units=num_nodes, num_layers=num_layers, num_targets=num_targets)
        elif args.model == 'PLSA':
            model = PLSA_G(in_dim=num_features, in_units=num_nodes, num_layers=num_layers, num_targets=num_targets)
        elif args.model == 'BIGFormer':
            model = BIGFormer(in_dim=num_features, in_units=num_nodes,
                              num_heads=args.num_heads, num_targets=num_targets, conv=args.conv,
                              norm_type=args.norm_type, attention_type=args.attention_type, num_PLSA=num_PLSA)

        criterion = nn.CrossEntropyLoss(reduction='sum')
        # criterion = margin_loss
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), verbose=False)

        t_start = time.perf_counter()
        best_ = 0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = train(model, optimizer, scheduler, criterion, train_loader, device)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            test_loss, test_accuracy, \
            test_auc, test_precision, test_sensitivity, \
            test_specificity, test_f1, \
            test_pred_out, test_y_true = eval_metrics(model, test_loader, criterion, device=device)

            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            test_aucs.append(test_auc)
            test_precisions.append(test_precision)
            test_sensitivities.append(test_sensitivity)
            test_specificities.append(test_specificity)
            test_f1s.append(test_f1)
            test_y_trues.append(test_y_true)
            test_pred_outs.append(test_pred_out)

            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_LOSS': train_loss,
                'train_ACC': train_accuracy,
                'test_LOSS': test_loss,
                'test_ACC': test_accuracy,
                'test_AUC': test_auc,
                'test_PRE': test_precision,
                'test_SEN': test_sensitivity,
                'test_SPE': test_specificity,
                'test_F1': test_f1
            }

            if epoch % args.lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr_decay_factor * param_group['lr']

            if best_ < test_accuracy:
                logger(eval_info)
                best_ = test_accuracy
                if args.save:
                    torch.save(model.state_dict(), opj(model_path, f'task_{args.task}_best_model_{fold}.pkl'))
                    pickle_dump(test_pred_out, opj(model_path, f'task_{args.task}_pred_out_{fold}.pkl'))

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    train_losses, train_accuracies, \
    test_losses, test_accuracies, \
    test_aucs, test_precisions, \
    test_sensitivities, test_specificities, \
    test_f1s = tensor(train_losses), tensor(train_accuracies), \
               tensor(test_losses), tensor(test_accuracies), \
               tensor(test_aucs), tensor(test_precisions), \
               tensor(test_sensitivities), tensor(test_specificities), \
               tensor(test_f1s)

    train_accuracies = train_accuracies.view(args.folds * args.repeats, args.epochs)
    train_losses = train_losses.view(args.folds * args.repeats, args.epochs)
    test_losses = test_losses.view(args.folds * args.repeats, args.epochs)
    test_accuracies = test_accuracies.view(args.folds * args.repeats, args.epochs)
    test_aucs = test_aucs.view(args.folds * args.repeats, args.epochs)
    test_precisions = test_precisions.view(args.folds * args.repeats, args.epochs)
    test_sensitivities = test_sensitivities.view(args.folds * args.repeats, args.epochs)
    test_specificities = test_specificities.view(args.folds * args.repeats, args.epochs)
    test_f1s = test_f1s.view(args.folds * args.repeats, args.epochs)
    _, selected_epoch_rep = test_accuracies.max(dim=1)

    train_LOSS = train_losses[torch.arange(args.folds * args.repeats, dtype=torch.long), selected_epoch_rep]
    train_ACC = train_accuracies[torch.arange(args.folds * args.repeats, dtype=torch.long), selected_epoch_rep]
    test_LOSS = test_losses[torch.arange(args.folds * args.repeats, dtype=torch.long), selected_epoch_rep]
    test_ACC = test_accuracies[torch.arange(args.folds * args.repeats, dtype=torch.long), selected_epoch_rep]
    test_AUC = test_aucs[torch.arange(args.folds * args.repeats, dtype=torch.long), selected_epoch_rep]
    test_SEN = test_sensitivities[torch.arange(args.folds * args.repeats, dtype=torch.long), selected_epoch_rep]
    test_SPE = test_specificities[torch.arange(args.folds * args.repeats, dtype=torch.long), selected_epoch_rep]
    test_PRE = test_precisions[torch.arange(args.folds * args.repeats, dtype=torch.long), selected_epoch_rep]
    test_F1 = test_f1s[torch.arange(args.folds * args.repeats, dtype=torch.long), selected_epoch_rep]

    train_LOSS_mean, train_LOSS_std = train_LOSS.mean().item(), train_LOSS.std().item()
    train_ACC_mean, train_ACC_std = train_ACC.mean().item(), train_ACC.std().item()
    test_LOSS_mean, test_LOSS_std = test_LOSS.mean().item(), test_LOSS.std().item()
    test_ACC_mean, test_ACC_std = test_ACC.mean().item(), test_ACC.std().item()
    test_AUC_mean, test_AUC_std = test_AUC.mean().item(), test_AUC.std().item()
    test_SEN_mean, test_SEN_std = test_SEN.mean().item(), test_SEN.std().item()
    test_SPE_mean, test_SPE_std = test_SPE.mean().item(), test_SPE.std().item()
    test_PRE_mean, test_PRE_std = test_PRE.mean().item(), test_PRE.std().item()
    test_F1_mean, test_F1_std = test_F1.mean().item(), test_F1.std().item()

    metrics = {
        'Train_LOSS': (train_LOSS_mean, train_LOSS_std, train_LOSS.numpy()),
        'Train_ACC': (train_ACC_mean, train_ACC_std, train_ACC.numpy()),
        'Test_LOSS': (test_LOSS_mean, test_LOSS_std, test_LOSS.numpy()),
        'Test_ACC': (test_ACC_mean, test_ACC_std, test_ACC.numpy()),
        'Test_AUC': (test_AUC_mean, test_AUC_std, test_AUC.numpy()),
        'Test_PRE': (test_PRE_mean, test_PRE_std, test_SEN.numpy()),
        'Test_SEN': (test_SEN_mean, test_SEN_std, test_SPE.numpy()),
        'Test_SPE': (test_SPE_mean, test_SPE_std, test_PRE.numpy()),
        'Test_F1': (test_F1_mean, test_F1_std, test_F1.numpy())
    }

    return metrics


def train(model, optimizer, scheduler, criterion, loader, device):
    model.train()
    model = model.to(device)
    total_loss = []
    correct = 0.
    total_ = len(loader.dataset)
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)

        out = model(data)
        loss = criterion(out, data.y.view(-1))

        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        total_loss.append(loss.item())

        loss.backward()
        optimizer.step()
    scheduler.step()

    return np.mean(total_loss), correct / total_


@torch.no_grad()
def eval_metrics(model, loader, criterion, device):
    model.eval()

    total_loss = 0.
    total_ = len(loader.dataset)
    y_true = []
    y_pred = []
    y_out = []

    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y.view(-1))
        pred = out.max(1)[1]

        y_true.append(data.y.view(-1).cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        y_out.append(out.cpu().numpy())
        total_loss += loss.item() * num_graphs(data)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_out = np.concatenate(y_out)

    auc = roc_auc_score(y_true, y_out[:, 1])
    f1 = f1_score(y_true, y_pred)

    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    loss = total_loss / total_
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    return loss, accuracy, auc, precision, sensitivity, specificity, f1, y_out, y_true