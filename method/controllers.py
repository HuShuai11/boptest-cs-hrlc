"""
PID controllers: PaperPID (fixed gains) and AdaptivePID (TCN_U-modulated gains).
"""

import numpy as np

STEP = 900


class PaperPID:
    """Fixed-gain PID: Kp=0.40, Ki=2e-4, Kd=10.0, D-filter α=0.2, anti-windup."""

    def __init__(self):
        self.kp, self.ki, self.kd, self.da = 0.40, 2e-4, 10.0, 0.2
        self.I = 0.0
        self.e_prev = None
        self.d_filt = 0.0

    def step(self, e, dt=STEP):
        e = float(e)
        dt = float(dt)
        if self.e_prev is None:
            d_raw = 0.0
        else:
            d_raw = (e - self.e_prev) / dt
        self.d_filt = (1.0 - self.da) * self.d_filt + self.da * d_raw
        self.e_prev = e
        u_unsat = self.kp * e + self.ki * self.I + self.kd * self.d_filt
        u_sat = float(np.clip(u_unsat, 0.0, 1.0))
        pu = (u_sat >= 0.999) and (e > 0)
        pd = (u_sat <= 0.001) and (e < 0)
        if not (pu or pd):
            self.I += e * dt
            self.I = float(np.clip(self.I, -2e5, 2e5))
        u_raw = self.kp * e + self.ki * self.I + self.kd * self.d_filt
        return float(np.clip(u_raw, 0.0, 1.0)), float(u_raw), float(self.I), float(self.d_filt)


class AdaptivePID:
    """
    Adaptive PID: Kp_eff = Kp0×(1+rp), Ki_eff = Ki0×(1+ri), Kd_eff = Kd0×(1+rd).
    rp, ri, rd ∈ [-rmax, rmax] from TCN_U gain_head.
    """

    def __init__(self):
        self.kp0, self.ki0, self.kd0 = 0.40, 2e-4, 10.0
        self.I = 0.0
        self.e_prev = None
        self.d_filt = 0.0
        self.da = 0.2

    def step(self, e, rp, ri, rd, dt=STEP):
        e = float(e)
        dt = float(dt)
        if self.e_prev is None:
            d_raw = 0.0
        else:
            d_raw = (e - self.e_prev) / dt
        self.d_filt = (1.0 - self.da) * self.d_filt + self.da * d_raw
        self.e_prev = e
        kp_eff = self.kp0 * (1.0 + rp)
        ki_eff = self.ki0 * (1.0 + ri)
        kd_eff = self.kd0 * (1.0 + rd)
        u_unsat = kp_eff * e + ki_eff * self.I + kd_eff * self.d_filt
        u_sat = float(np.clip(u_unsat, 0.0, 1.0))
        pu = (u_sat >= 0.999) and (e > 0)
        pd = (u_sat <= 0.001) and (e < 0)
        if not (pu or pd):
            self.I += e * dt
            self.I = float(np.clip(self.I, -2e5, 2e5))
        u_raw = kp_eff * e + ki_eff * self.I + kd_eff * self.d_filt
        return (float(np.clip(u_raw, 0.0, 1.0)), float(u_raw), float(self.I),
                float(self.d_filt), kp_eff, ki_eff, kd_eff)
