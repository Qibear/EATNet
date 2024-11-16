import random
import numpy as np
import torch
import math


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


# def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
#     decay = decay_rate ** (epoch // decay_epoch)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = decay * init_lr
#         lr = param_group['lr']
#     return lr


# TODO 这种学习率衰减方式其实是cos的方式进行衰减，
def adjust_lr(optimizer, init_lr, epoch, T_max=250, min_lr=0.0001):
    if epoch >= T_max:
        lr = min_lr
    else:
        lr = 0.5 * init_lr * (1 + math.cos(epoch * math.pi / T_max))
        lr = max(lr, min_lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
