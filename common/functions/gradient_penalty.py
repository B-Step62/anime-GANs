import torch

def gradient_penalty(x_real, x_fake, dis, device, y=None):
    epsilon = torch.rand(x_real.shape[0], 1, 1, 1).to(device).expand_as(x_real)
    x_hat = torch.autograd.Variable(epsilon * x_real.data + (1 - epsilon) * x_fake.data, requires_grad=True)

    d_hat = dis(x_hat, y=y) if y is not None else dis(x_hat)

    if isinstance(d_hat, list) or isinstance(d_hat, tuple):
        d_hat = d_hat[0]

    grad = torch.autograd.grad(outputs=d_hat,
                               inputs=x_hat,
                               grad_outputs=torch.ones(d_hat.shape).to(device),
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    grad = grad.view(grad.shape[0], -1)
    grad_norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    d_loss_gp = torch.mean((grad_norm - 1) ** 2)

    return d_loss_gp
