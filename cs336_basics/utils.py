import math
import torch

class MyAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = {
            'lr': lr,
            'eps': eps,
            'betas': betas,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults=defaults)

    # not needed as the inherited class contains it
    # def zero_grad(self):
    #     for group in self.param_groups:
    #         for p in group['params']:
    #             if p.grad is not None:
    #                 p.grad.zero_()

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']

            for i, param in enumerate(group['params']):
                if param.grad is None:
                    continue

                grad = param.grad.data

                # get state associated with param
                state = self.state[param]
                if len(state) == 0:
                    state['t'] = 1
                    state['m'] = torch.zeros_like(param.data)
                    state['v'] = torch.zeros_like(param.data)

                t = state['t']
                m = state['m']
                v = state['v']

                # Apply weight decay directly to parameters (AdamW style)
                param.data.mul_(1 - lr * wd)

                # Update biased first moment estimate
                # beta1 * i + (1-beta1) * grad
                # self.m = self.beta1 * self.m + (1 - self.beta1) * grad
                m.mul_(beta1).add_(grad, alpha = 1-beta1)

                # Update biased second raw moment estimate
                # self.v = self.beta2 * self.v + (1 - self.beta2) * grad.pow(2)
                v.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)
                
                step_size = lr * ((1 - beta2 ** t) ** 0.5) / (1 - beta1 ** t)

                # Update parameters
                param.data.addcdiv_(m, v ** 0.5 + eps, value=-step_size)
                param.data.mul_(1 -lr * wd)

                # param.data.add_(m_hat * -lr/ (v_hat.sqrt() + eps))

                state['t'] = t + 1

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

def load_checkpoint(src, model, optimizer):
    ckpt = torch.load(src)
    model = model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['iteration']