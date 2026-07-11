"""
Neural network modules: TCN blocks, PureTCN (T_set head), TCNU (PID+FF dual head), buffers.
"""

from collections import deque
import numpy as np
import torch
import torch.nn as nn


# ═══════════════ TCN Building Blocks ═══════════════

class Chomp1d(nn.Module):
    def __init__(self, cs):
        super().__init__()
        self.cs = cs

    def forward(self, x):
        return x[:, :, :-self.cs].contiguous() if self.cs else x


class TemporalBlock(nn.Module):
    """TCN temporal block with 2 dilated causal conv layers + residual skip."""

    def __init__(self, ic, oc, ks=3, dilation=1, dropout=0.05):
        super().__init__()
        pad = (ks - 1) * dilation
        self.c1 = nn.Conv1d(ic, oc, ks, padding=pad, dilation=dilation)
        self.ch1 = Chomp1d(pad)
        self.r1 = nn.ReLU()
        self.d1 = nn.Dropout(dropout)
        self.c2 = nn.Conv1d(oc, oc, ks, padding=pad, dilation=dilation)
        self.ch2 = Chomp1d(pad)
        self.r2 = nn.ReLU()
        self.d2 = nn.Dropout(dropout)
        self.ds = nn.Conv1d(ic, oc, 1) if ic != oc else None
        self.or_ = nn.ReLU()

    def forward(self, x):
        y = self.d1(self.r1(self.ch1(self.c1(x))))
        y = self.d2(self.r2(self.ch2(self.c2(y))))
        return self.or_(y + (x if self.ds is None else self.ds(x)))


# ═══════════════ PureTCN (T_set head) ═══════════════

class PureTCN(nn.Module):
    """TCN for single ΔT output. 23-dim input, Tanh×dt_max head."""

    def __init__(self, input_dim=23, tcn_channels=(32, 64), kernel_size=3, dropout=0.05, dt_max=0.8):
        super().__init__()
        self.dt_max = dt_max
        blocks = []
        in_ch = input_dim
        for i, oc in enumerate(tcn_channels):
            blocks.append(TemporalBlock(in_ch, oc, kernel_size, 2 ** i, dropout))
            in_ch = oc
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Conv1d(tcn_channels[-1], 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, x_seq):
        z = self.encoder(x_seq.transpose(1, 2))
        return self.dt_max * self.tanh(self.head(z)[:, :, -1])


# ═══════════════ TCNU (PID gain + feedforward dual head) ═══════════════

class TCNU(nn.Module):
    """TCN_U: 16-dim input, dual head — PID gain correction (rp,ri,rd) + feedforward (u_ff)."""

    def __init__(self, input_dim=16, tcn_channels=(16, 32), kernel_size=3, dropout=0.05,
                 rmax=0.25, ff_scale=0.03):
        super().__init__()
        self.rmax = rmax
        self.ff_scale = ff_scale
        blocks = []
        in_ch = input_dim
        for i, oc in enumerate(tcn_channels):
            blocks.append(TemporalBlock(in_ch, oc, kernel_size, 2 ** i, dropout))
            in_ch = oc
        self.encoder = nn.Sequential(*blocks)
        self.gain_head = nn.Linear(tcn_channels[-1], 3)
        self.ff_head = nn.Linear(tcn_channels[-1], 1)
        nn.init.zeros_(self.gain_head.weight)
        nn.init.zeros_(self.gain_head.bias)
        nn.init.zeros_(self.ff_head.weight)
        nn.init.zeros_(self.ff_head.bias)
        self.tanh = nn.Tanh()

    def forward(self, x_seq):
        z = self.encoder(x_seq.transpose(1, 2))
        z_last = z[:, :, -1]
        r = self.rmax * self.tanh(self.gain_head(z_last))
        u_ff = self.ff_scale * self.tanh(self.ff_head(z_last))
        return r, u_ff


# ═══════════════ Replay Buffers ═══════════════

class TBuffer:
    """Single-label buffer for TCN_T training. Stores (x_seq, ΔT_teacher)."""

    def __init__(self, maxlen=50000):
        self.buf = deque(maxlen=maxlen)

    def add(self, x_seq, yT):
        self.buf.append(
            (np.asarray(x_seq, dtype=np.float32), np.asarray([yT], dtype=np.float32)))

    def __len__(self):
        return len(self.buf)

    def sample(self, bs):
        B = __import__('random').sample(self.buf, min(bs, len(self)))
        return np.stack([b[0] for b in B]), np.stack([b[1] for b in B])


class DualLabelBuf:
    """Dual-label buffer for TCN_U training. Stores (x_seq, gain_label, u_ff_teacher)."""

    def __init__(self, maxlen=10000):
        self.buf = deque(maxlen=maxlen)

    def add(self, xs, gl, ff):
        self.buf.append((
            np.asarray(xs, dtype=np.float32),
            np.asarray(gl, dtype=np.float32),
            np.asarray([ff], dtype=np.float32)))

    def __len__(self):
        return len(self.buf)

    def sample(self, bs):
        B = __import__('random').sample(self.buf, min(bs, len(self)))
        return (np.stack([b[0] for b in B]),
                np.stack([b[1] for b in B]),
                np.stack([b[2] for b in B]))


# ═══════════════ Sequence Builder ═══════════════

def build_seq(history, seq_len, feat_dim):
    """Build fixed-length sequence from history deque. Pad with first element if short."""
    items = [np.asarray(x, dtype=np.float32) for x in list(history)]
    if not items:
        return np.zeros((seq_len, feat_dim), dtype=np.float32)
    if len(items) >= seq_len:
        seq = items[-seq_len:]
    else:
        seq = [items[0]] * (seq_len - len(items)) + items
    return np.stack(seq, axis=0).astype(np.float32)
