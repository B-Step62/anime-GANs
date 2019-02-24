import torch

def calc_mean_std(x, epsilon=1e-5):
    assert (len(x.size()) == 4)
    N, C = x.size()[:2]
    x_var = x.view(N, C, -1).var(dim=2) + epsilon
    x_std = x_var.sqrt().view(N, C, 1, 1)
    x_mean = x.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return x_mean, x_std
