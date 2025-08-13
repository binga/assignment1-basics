import math
import torch

class MyAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.wd = weight_decay
        self.eps = eps
        self.step_count = 0

        # Initialize momentum and velocity for each parameter
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

        self.ckpt = {}
        self.ckpt['param_groups'] = [{
            'lr': lr,
            'eps': eps,
            'betas': (self.beta1, self.beta2),
            'weight_decay': self.wd
        }]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.step_count += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.data

            # Apply weight decay directly to parameters (AdamW style)
            param.data.mul_(1 - self.lr * self.wd)

            # Update biased first moment estimate
            # beta1 * i + (1-beta1) * grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad.pow(2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.step_count)

            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.step_count)

            # Update parameters
            param.data.add_(m_hat * -self.lr/ (v_hat.sqrt() + self.eps))

        

    def state_dict(self):
        return self.ckpt
    

def lr_scheduler(t, alpha_max, alpha_min, tw, tc):
    # t - time step
    # alpha_max - max learning rate
    # alpha min - min learning rate
    # tw - warmup iterations
    # tc - number of cosine annealing iterations
    if t < tw:
        lr = (t / tw) * alpha_max
    elif tw <= t < tc:
        lr = alpha_min + 0.5 * (1 + math.cos(math.pi * (t - tw) / (tc - tw))) * (alpha_max - alpha_min)
    else:
        lr = alpha_min
    return lr


def gradient_clipping(g, M, eps=1e-6):
    # g - gradient parameters
    # M - max norm
    norm = torch.norm(g)
    if norm < M:
        return g
    else:
        g.mul_(M / (norm + eps))
        return g
    
def save_checkpoint(model, optimizer, iteration, out):
    obj = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(obj, out)