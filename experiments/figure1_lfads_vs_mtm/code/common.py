import os
import numpy as np
import torch

DEVICE = torch.device(os.environ.get('FIG1_DEVICE', 'mps' if torch.backends.mps.is_available() else 'cpu'))


def r2(y, p):
    y = np.asarray(y, dtype=np.float64); p = np.asarray(p, dtype=np.float64)
    return float(1. - np.square(y - p).sum() / np.square(y - y.mean()).sum())
