# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     main.py
   Description :
   Author :       zouqi
   date：          2023/2/17
-------------------------------------------------
   Change Activity:
                   2023/2/17:
-------------------------------------------------
"""
__author__ = 'zouqi'

from os.path import join as opj

from parameters import parameters_parser
from train_eval_helper import cross_validation
from utils import seed_everything, logger, make_dir

args = parameters_parser()

seed_everything(args.seed)

if __name__ == '__main__':
    model_path = opj(args.work_dir, args.model_path, f'model_{args.model}', f'conv_{args.conv}',
                     f'task_{args.task}', f'cutoff_arg_{args.cutoff_arg}', f'num_heads_{args.num_heads}')
    args.model_path = model_path
    log_path = opj(args.work_dir, args.log_path)
    make_dir(model_path, keep=True)
    make_dir(log_path, keep=True)

    res = cross_validation(args, logger)

    print(f"{res['Test_ACC'][0]:.4f}, {res['Test_AUC'][0]:.4f}, {res['Test_PRE'][0]:.4f}, {res['Test_SEN'][0]:.4f}, "
          f"{res['Test_SPE'][0]:.4f}, {res['Test_F1'][0]:.4f}")
    print(f"{res['Test_ACC'][1]:.4f}, {res['Test_AUC'][1]:.4f}, {res['Test_PRE'][1]:.4f}, {res['Test_SEN'][1]:.4f}, "
          f"{res['Test_SPE'][1]:.4f}, {res['Test_F1'][1]:.4f}")