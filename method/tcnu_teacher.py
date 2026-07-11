"""
TCN_U teacher labels: feature construction, gain labels, action teacher (feedforward label).
"""

import numpy as np


def make_tcnu_features(Tz, Tsp, T_rule, Tout, Qsol, price, occ, hour,
                       u_pid, u_d, u_base, u_prev):
    """16-dimensional TCN_U input features."""
    hr = 2.0 * np.pi * float(hour) / 24.0
    return np.array([
        (float(Tz) - 21.0) / 5.0,
        (float(Tsp) - 21.0) / 5.0,
        (float(Tz) - float(Tsp)) / 3.0,
        (float(T_rule) - 21.0) / 5.0,
        (float(Tout) - 10.0) / 15.0,
        float(Qsol) / 800.0,
        (float(price) - 0.25) / 0.15,
        1.0 if occ else 0.0,
        0.0, 0.30,
        float(u_pid),
        float(u_d) / 0.2,
        float(u_base),
        float(u_prev),
        np.sin(hr),
        np.cos(hr),
    ], dtype=np.float32)


def compute_gain_labels(Tz, T_low, T_high, derivative):
    """
    Heuristic PID gain labels.
    - cold margin → increase Kp, Ki (more aggressive heating)
    - hot margin  → decrease Kp, Ki (less aggressive)
    - oscillation → increase Kd (damping)
    """
    cold = np.clip((T_low + 0.5 - Tz) / 1.0, 0, 1)
    hot = np.clip((Tz - T_high + 0.5) / 1.0, 0, 1)
    osc = np.clip(abs(derivative) / 0.3, 0, 1)
    return np.array([
        +0.3 * cold - 0.1 * hot - 0.1 * osc,  # rp
        +0.2 * cold - 0.1 * hot,              # ri
        +0.3 * osc + 0.1 * hot,               # rd
    ], dtype=np.float32)


def action_teacher_delta_u(scenario, Tz, Tsp, T_low, T_high, Tout, Qsol, price,
                           theta, u_base, u_prev):
    """
    Feedforward teacher label: enumerate 21 candidate Δu ∈ [-0.08, 0.08],
    use RLS model θ to predict resulting Tz, minimize weighted cost.
    """
    sc = str(scenario).lower()
    th = np.asarray(theta, dtype=float)
    ub = float(np.clip(u_base, 0, 1))
    up = float(np.clip(u_prev, 0, 1))
    if sc == "typical":
        te = min(T_high, 23.6)
        tl = max(T_low, 20.5)
        wh, wl, wt, we, ws = 180., 80., 1., 0.02, 0.30
    else:
        te = T_high
        tl = max(T_low, 20.8)
        wh, wl, wt, we, ws = 60., 160., 1., 0.02, 0.25
    best_cost = None
    best_u = ub
    for du in np.linspace(-0.08, 0.08, 21):
        uc = np.clip(ub + du, 0, 1)
        Tp = float(np.dot(th, np.array([Tz, Tout, Qsol / 1000., uc, 1.])))
        J = (wl * max(0, tl - Tp) ** 2 +
             wh * max(0, Tp - te) ** 2 +
             wt * (Tp - Tsp) ** 2 +
             we * max(0, float(price)) * uc ** 2 +
             ws * (uc - up) ** 2)
        if best_cost is None or J < best_cost:
            best_cost = J
            best_u = uc
    return float(np.clip(best_u - ub, -0.08, 0.08))
