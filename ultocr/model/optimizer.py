import math
import numpy as np
import torch


def AdamDecay(config, parameters):
    optimizer = torch.optim.Adam(parameters, lr=config['optimizer']['base_lr'],
                                 betas=(config['optimizer']['beta1'], config['optimizer']['beta2']),
                                 weight_decay=config['optimizer']['weight_decay'])
    return optimizer


def SGDDecay(config, parameters):
    optimizer = torch.optim.SGD(parameters, lr=config['optimizer']['base_lr'],
                                momentum=config['optimizer']['momentum'],
                                weight_decay=config['optimizer']['weight_decay'])
    return optimizer


def RMSPropDecay(config, parameters):
    optimizer = torch.optim.RMSprop(parameters, lr=config['optimizer']['base_lr'],
                                    alpha=config['optimizer']['alpha'],
                                    weight_decay=config['optimizer']['weight_decay'],
                                    momentum=config['optimizer']['momentum'])
    return optimizer


def lr_poly(base_lr, epoch, max_epoch=1200, factor=0.9):
    return base_lr * ((1 - float(epoch) / max_epoch) ** (factor))


def SGDR(lr_max, lr_min, T_cur, T_m, ratio=0.3):
    if T_cur % T_m == 0 and T_cur != 0:
        lr_max = lr_max - lr_max * ratio
    lr = lr_min + 1 / 2 * (lr_max - lr_min) * (1 + math.cos((T_cur % T_m / T_m) * math.pi))
    return lr, lr_max


def adjust_learning_rate(optimizer, base_lr, iter, all_iters, factor, warmup_iters=0, warmup_factor=1.0 / 3):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if iter < warmup_iters:
        alpha = float(iter) / warmup_iters
        rate = warmup_factor * (1 - alpha) + alpha
    else:
        rate = np.power(1.0 - iter / float(all_iters + 1), factor)
    lr = rate * base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

