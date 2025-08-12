import math

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