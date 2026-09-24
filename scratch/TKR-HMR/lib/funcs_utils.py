import os
import sys
import time
import math
import numpy as np
import cv2
import shutil
from collections import OrderedDict

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from core.config import cfg


def lr_check(optimizer, epoch):
    warmup_epochs = getattr(cfg.TRAIN, 'warmup_epochs', 0)
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        lr_warmup(optimizer, cfg.TRAIN.lr, epoch, warmup_epochs)

    for param_group in optimizer.param_groups:
        curr_lr = param_group['lr']
    print(f"Current epoch {epoch}, lr: {curr_lr}")


def lr_warmup(optimizer, lr, epoch, base):
    """Linear warmup: ramp from min_lr to target lr over 'base' epochs."""
    min_lr = getattr(cfg.TRAIN, 'min_lr', 1e-6)
    warmup_lr = min_lr + (lr - min_lr) * (epoch / base)
    for param_group in optimizer.param_groups:
        param_group['lr'] = warmup_lr
    return warmup_lr


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        self.acc += time.time() - self.t0  # cacluate time diff

    def reset(self):
        self.acc = 0

    def print(self):
        return round(self.acc, 2)


def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()


def stop():
    sys.exit()


def check_data_pararell(train_weight):
    new_state_dict = OrderedDict()
    for k, v in train_weight.items():
        name = k[7:]  if k.startswith('module') else k  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizer(model):
    optimizer = None
    if cfg.TRAIN.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.lr,
            momentum=cfg.TRAIN.momentum,
            weight_decay=cfg.TRAIN.weight_decay,
            nesterov=cfg.TRAIN.nesterov
        )
    elif cfg.TRAIN.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=cfg.TRAIN.lr
        )
    elif cfg.TRAIN.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.lr
        )

    return optimizer


def get_scheduler(optimizer):
    scheduler = None
    if cfg.TRAIN.scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.lr_step, gamma=cfg.TRAIN.lr_factor)
    elif cfg.TRAIN.scheduler == 'cosine':
        min_lr = getattr(cfg.TRAIN, 'min_lr', 1e-6)
        warmup_epochs = getattr(cfg.TRAIN, 'warmup_epochs', 0)
        # T_max = tổng epochs trừ warmup (cosine chỉ chạy sau warmup)
        T_max = max(cfg.TRAIN.end_epoch - warmup_epochs, 1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr)
    elif cfg.TRAIN.scheduler == 'platue':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.TRAIN.lr_factor, patience=10, min_lr=1e-5)

    return scheduler


def save_checkpoint(states, epoch, is_best=None):
    # file_name = f'checkpoint{epoch}.pth.tar'
    output_dir = cfg.checkpoint_dir
    # if states['epoch'] == cfg.TRAIN.end_epoch:
    #     file_name = 'final.pth.tar'
    # torch.save(states, os.path.join(output_dir, file_name))

    if is_best:
        torch.save(states, os.path.join(output_dir, 'best.pth.tar'))


def load_checkpoint(load_dir, epoch=0, pick_best=False):
    try:
        print(f"Fetch model weight from {load_dir}")
        checkpoint = torch.load(load_dir, map_location='cuda')
        return checkpoint
    except Exception as e:
        raise ValueError("No checkpoint exists!\n", e)