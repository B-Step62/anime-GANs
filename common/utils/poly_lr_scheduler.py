import torch

def poly_lr_scheduler(optimizer, init_lr, iteration, lr_decay_iter=1,
                      max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iteration % lr_decay_iter or iteration > max_iter:
        return optimizer

    lr = init_lr*(1 - iteration/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
