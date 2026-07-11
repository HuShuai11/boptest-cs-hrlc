"""
Schedulers for β_time, α_u, α_d, and seed setting utility.
"""

import random
import numpy as np
import torch


def set_seed(seed):
    """Set all random seeds for reproducibility. seed=0 means no seed."""
    if seed != 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def beta_sched(step, warmup=96, ramp=192, bmax=1.0):
    """
    β_time schedule: TCN_T output is masked during warmup, then linearly ramped.
    Used in L1, L2, L3.
    """
    if step < warmup:
        return 0.0
    return bmax * min(1.0, (step - warmup) / max(1, ramp))


def alpha_u_sched(step, warmup=288, ramp=288, amax=1.0):
    """
    α_u schedule: TCN_U feedforward weight, gradual warmup to avoid early oscillations.
    Used in L3 (full).
    """
    if step < warmup:
        return 0.0
    return amax * min(1.0, (step - warmup) / max(1, ramp))
