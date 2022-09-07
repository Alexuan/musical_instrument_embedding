import sys
import os
import time
import json
import shutil
import argparse
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from musyn.utils import Config
from musyn.data import create_dataset
from musyn.models import create_model
from musyn.models import get_optimizer, get_criterion, get_scheduler
from musyn.utils import mkdir_or_exist
from musyn.evaluation import compute_eer_det, plot_TSNE
from musyn.engine import create_engine


def update_args(cfg, args):

    cfg.pretrain_model_pth = args.pretrain_model_pth
    cfg.prefix = args.pretrain_model_pth.split('/')[-2]
    cfg.log_config.dir = 'log/'+cfg.prefix

    if 'asm' not in cfg.prefix:
        cfg.model.head1.type = 'LinearClsHead'
        cfg.model.head2.type = 'LinearClsHead'
        cfg.criterion.type = 'NLL'

    # cfg.seed = int(cfg.prefix.split('_')[-1][1:])
    
    return cfg


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--pretrain_model_pth', type=str, default=None)
    parser.add_argument('--models-listfile', type=str, default=None)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config_file)

    if args.models_listfile is not None:
        f = open(args.models_listfile, 'r')
        models_listfile = f.readlines()
    else:
        models_listfile = [args.pretrain_model_pth]
    
    for item in models_listfile:
        args.pretrain_model_pth = item.strip()
        cfg = update_args(cfg, args)

        # log
        mkdir_or_exist(cfg.log_config.dir)
        # engine
        trainer = create_engine(cfg)
        trainer.inference()
