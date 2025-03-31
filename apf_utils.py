import torch
import torch.nn.functional as F
import numpy as np

def apf_mixup_sampling(buffer_x, buffer_y, mix_ratio=0.3, alpha=0.2):
    """
    Applies a simple APF-like mixup strategy to buffer samples.
    :param buffer_x: Tensor of shape (N, C, H, W)
    :param buffer_y: Tensor of shape (N,) with class indices
    :param mix_ratio: Proportion of mixed samples in total batch
    :param alpha: Mixup Beta distribution parameter
    """
    device = buffer_x.device
    n_samples = buffer_x.shape[0]
    n_mix = int(n_samples * mix_ratio)

    if n_samples < 2 or n_mix == 0:
        return buffer_x, buffer_y  # Fallback to normal

    # Random pair indices
    idx1 = torch.randint(0, n_samples, (n_mix,))
    idx2 = torch.randint(0, n_samples, (n_mix,))

    x1, x2 = buffer_x[idx1], buffer_x[idx2]
    y1, y2 = buffer_y[idx1], buffer_y[idx2]

    lam = torch.distributions.Beta(alpha, alpha).sample((n_mix,)).to(device)
    lam_x = lam.view(-1, 1, 1, 1)

    mixed_x = lam_x * x1 + (1 - lam_x) * x2
    mixed_y = lam * F.one_hot(y1, num_classes=buffer_y.max().item() + 1) + \
              (1 - lam) * F.one_hot(y2, num_classes=buffer_y.max().item() + 1)

    # Append to original data
    final_x = torch.cat([buffer_x, mixed_x], dim=0)
    final_y = torch.cat([F.one_hot(buffer_y, num_classes=buffer_y.max() + 1).float(), mixed_y], dim=0)

    return final_x, final_y
