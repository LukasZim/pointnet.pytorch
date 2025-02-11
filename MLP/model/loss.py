import torch.nn as nn
import torch


def l1_loss():
    return nn.L1Loss()

def l2_loss():
    return nn.MSELoss()

def log_loss():
    return nn.NLLLoss()


def custom_loss(output, target):
    epsilon = 0.0000
    ratio = (output + epsilon) / (target + epsilon)
    ratio_inv = ratio ** -1
    ratio = torch.max(ratio, ratio_inv)
    alpha = 10
    loss = torch.sum(torch.abs(output - target) / (1 + alpha * target))
    loss = torch.sum(torch.abs(output- target))
    return loss / len(target)