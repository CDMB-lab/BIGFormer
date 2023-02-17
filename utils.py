# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     utils.py
   Description :
   Author :       zouqi
   date：          2023/2/17
-------------------------------------------------
   Change Activity:
                   2023/2/17:
-------------------------------------------------
"""
__author__ = 'zouqi'


import os
import pickle
import random
import shutil
import sys
import warnings
from os.path import join as opj

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

warnings.filterwarnings('ignore')

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score, recall_score, f1_score, confusion_matrix, precision_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from texttable import Texttable


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init(worker_init):
    seed = 0xBADF00D
    np.random.seed(int(seed) + worker_init)


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def logger(info):
    fold, epoch = info['fold'], info['epoch']
    if epoch == 1 or epoch % 1 == 0:
        train_LOSS, train_ACC, \
        test_LOSS, test_ACC, \
        test_AUC, test_PRE, \
        test_SEN, test_SPE, \
        test_F1 = info['train_LOSS'], info['train_ACC'], \
                  info['test_LOSS'], info['test_ACC'], \
                  info['test_AUC'], info['test_PRE'], \
                  info['test_SEN'], info['test_SPE'], \
                  info['test_F1']

        print('{:02d}/{:03d}: Train Loss: {:.4f}, Train Acc: {:.3f}, Test Loss: {:.4f}, Test ACC: {:.5f}, '
              'Test AUC: {:.5f}, Test PRE: {:.5f}, Test SEN: {:.5f}, Test SPE: {:.5f}, Test F1: {:.5f}'.format(
            fold, epoch, train_LOSS, train_ACC, test_LOSS, test_ACC, test_AUC, test_PRE, test_SEN, test_SPE, test_F1))

    sys.stdout.flush()


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable(max_width=100)
    t.add_rows([['Parameter', 'Value']])
    t.add_rows([[k.replace('_', ' ').capitalize(), args[k]] for k in keys])
    return t.draw()


def make_dir(dir_, keep=True):
    """
        create a directory, if the directory already exists,
         it will be processed according to the behavior specified by `keep`.
    """
    exists = os.path.exists(dir_)
    if keep and exists:
        return False
    if exists:
        shutil.rmtree(dir_)
    os.makedirs(dir_)


def pickle_dump(data, dump_path):
    with open(dump_path, 'wb') as f:
        pickle.dump(data, f)


def specificity(y_ture, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_ture, y_pred).ravel()
    return tn / (tn + fp)


def get_performance(cl, x, y, n_splits=10, refit='accuracy', verbose=0,
                    save_path='/lab_data/data_cache/zouqi/codes/TAE/models'):
    save_path = opj(save_path, cl)
    make_dir(save_path, keep=False)
    pickle_dump(x, opj(save_path, 'x.pkl'))
    pickle_dump(y, opj(save_path, 'y.pkl'))

    s = {
        'accuracy': 'accuracy',
        'roc_auc': 'roc_auc',
        'precision': 'precision',
        'sensitivity': 'recall',
        'specificity': make_scorer(specificity, greater_is_better=True),
        'f1': 'f1'
    }

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=12345)

    if cl == 'svm':
        params = {
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [0.001, 0.0001, 0.00001, 0.000001]
        }
        classifier = GridSearchCV(SVC(), params, scoring=s, cv=kf, n_jobs=-1,
                                  verbose=verbose, refit=refit)
    elif cl == 'lr':
        params = [
            {'solver': ['newton-cg'], 'penalty': ['none', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]},
            {'solver': ['lbfgs'], 'penalty': ['none', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]},
            {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]},
            {'solver': ['sag'], 'penalty': ['l2', 'none'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]},
            {'solver': ['saga'], 'penalty': ['none', 'elasticnet', 'l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}
        ]

        classifier = GridSearchCV(LogisticRegression(), params, scoring=s, cv=kf, n_jobs=-1,
                                  verbose=verbose, refit=refit)
    elif cl == 'decision_tree':
        params = {
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [2, 5, 10, 20, 50, 100],
            'criterion': ['gini', 'entropy']
        }
        classifier = GridSearchCV(DecisionTreeClassifier(), params, scoring=s, cv=kf, n_jobs=-1,
                                  verbose=verbose, refit=refit)
    elif cl == 'rf':
        params = {
            'n_estimators': [10, 50, 100, 200, 500, 1000],
            'max_depth': [2, 5, 10, 20, 50, 100],
            'max_features': ['auto', 'sqrt', 'log2', None]
        }
        classifier = GridSearchCV(RandomForestClassifier(), params, scoring=s, cv=kf, n_jobs=-1,
                                  verbose=verbose, refit=refit)
    elif cl == 'adaboost':
        DTC = DecisionTreeClassifier(max_features='auto', class_weight='auto', max_depth=None)

        params = {
            'base_estimator__criterion': ['gini', 'entropy'],
            'base_estimator__smax_features': ['auto', 'sqrt', 'log2'],
            'n_estimators': [10, 50, 100, 200, 500, 1000],
            'learning_rate': [0.01, 0.1]
        }
        classifier = GridSearchCV(AdaBoostClassifier(base_estimator=DTC), params, scoring=s, cv=kf, n_jobs=-1,
                                  verbose=verbose, refit=refit)
    elif cl == 'gradientboost':
        params = {
            'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
            'max_depth': [3, 5, 8],
            'max_features': ['log2', 'sqrt'],
            'criterion': ['friedman_mse', 'mae'],
            'n_estimators': [10, 50, 100, 200, 500, 1000]
        }
        classifier = GridSearchCV(GradientBoostingClassifier(), params, scoring=s, cv=kf, n_jobs=-1,
                                  verbose=verbose, refit=refit)
    else:
        raise ValueError('Unknown classifier: {}'.format(cl))

    classifier.fit(x, y)
    best_params = classifier.best_params_
    res = classifier.cv_results_
    selected = classifier.best_index_

    pickle_dump(res, opj(save_path, 'results.pkl'))
    with open(opj(save_path, 'best_params.txt'), 'w') as f:
        for k, v in best_params.items():
            f.write(f'{k}: {v}\n')

    metrics_ = {
        'accuracy': (res['mean_test_accuracy'][selected], res['std_test_accuracy'][selected]),
        'roc_auc': (res['mean_test_roc_auc'][selected], res['std_test_roc_auc'][selected]),
        'precision': (res['mean_test_precision'][selected], res['std_test_precision'][selected]),
        'sensitivity': (res['mean_test_sensitivity'][selected], res['std_test_sensitivity'][selected]),
        'specificity': (res['mean_test_specificity'][selected], res['std_test_specificity'][selected]),
        'f1': (res['mean_test_f1'][selected], res['std_test_f1'][selected]),
    }
    metrics_df = pd.DataFrame(metrics_).T
    metrics_df.columns = ['Mean', 'Std']
    metrics_df.index = ['ACC', 'AUC', 'PRE', 'SEN', 'SPE', 'F1']
    metrics_df.to_csv(opj(save_path, 'metrics.csv'))

    train_ = []
    test_ = []
    for fold, (train_idx, test_idx) in tqdm(enumerate(kf.split(x, y)), desc=f'{cl} Predicting:', total=n_splits):
        train_.append(train_idx)
        test_.append(test_idx)
        model = classifier.estimator

        if cl == 'svm':
            best_params['probability'] = True

        model.set_params(**best_params)

        model.fit(x[train_idx, :], y[train_idx])
        pred = model.predict_proba(x[test_idx])
        np.savetxt(opj(save_path, f'pred_{fold}.txt'), pred)

    pickle_dump(train_, opj(save_path, 'train_index.pkl'))
    pickle_dump(test_, opj(save_path, 'test_index.pkl'))
    return metrics_df


def masked_softmax(x, mask, **kwargs):
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")

    return torch.softmax(x_masked, **kwargs)