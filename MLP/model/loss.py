import torch.nn as nn
import torch


def l1_loss():
    return nn.L1Loss()

def l2_loss():
    return nn.MSELoss()

def log_loss():
    return nn.NLLLoss()


def custom_loss(output, target):
    assert output.shape == target.shape
    alpha = 10
    loss = torch.sum(torch.abs(output - target) / (1 + alpha * target))
    return loss / len(target)