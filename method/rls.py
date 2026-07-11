"""
RLS-based 5-parameter RC model identifier for building thermal dynamics.
Used to generate action teacher labels for TCN_U training.
"""

import numpy as np


class RLSIdentifier:
    """Online RLS identifier for 5-param RC model: Tz(k+1)=θ₁·Tz+θ₂·Tout+θ₃·Qsol/1000+θ₄·u+θ₅."""

    def __init__(self, forgetting=0.998):
        self.theta = np.array([0.95, 0.03, 0.05, 0.30, 0.0], dtype=float)
        self.P = np.eye(5) * 100.0
        self.lam = forgetting

    def make_phi(self, Tz, Tout, Qsol, u_hp):
        return np.array([float(Tz), float(Tout), float(Qsol) / 1000.0,
                         float(u_hp), 1.0], dtype=float)

    def predict(self, phi):
        return float(np.dot(self.theta, phi))

    def update(self, phi_prev, Tz_next):
        phi = phi_prev.reshape(-1, 1)
        y = float(Tz_next)
        tc = self.theta.reshape(-1, 1)
        denom = self.lam + (phi.T @ self.P @ phi).item()
        K = (self.P @ phi) / denom
        pred = (tc.T @ phi).item()
        err = y - pred
        tc = tc + K * err
        self.P = (self.P - K @ phi.T @ self.P) / self.lam
        self.theta = tc.flatten()
        self.theta[0] = np.clip(self.theta[0], 0.70, 1.05)
        self.theta[1] = np.clip(self.theta[1], -0.20, 0.20)
        self.theta[2] = np.clip(self.theta[2], -2.00, 2.00)
        self.theta[3] = np.clip(self.theta[3], 0.00, 3.00)
        self.theta[4] = np.clip(self.theta[4], -10.0, 10.0)
        return pred, err, self.theta.copy()
