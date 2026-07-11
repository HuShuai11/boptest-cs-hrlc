"""
Perception layer: comfort bounds, T_rule, future features, TCN features, teacher label.
"""

import numpy as np
from datetime import datetime, timedelta

STEP = 900


def paper_comfort_bounds(dt_now):
    wd = dt_now.weekday()
    h = dt_now.hour + dt_now.minute / 60.0
    if wd < 5 and 7.0 <= h < 20.0:
        return 15.0, 30.0
    return 21.0, 24.0


def is_occupied(dt_now):
    wd = dt_now.weekday()
    h = dt_now.hour + dt_now.minute / 60.0
    return not (wd < 5 and 7.0 <= h < 20.0)


def comfort_bounds_kelvin(dt_now):
    wd = dt_now.weekday()
    h = dt_now.hour + dt_now.minute / 60.0
    return 288.15 if (wd < 5 and 7.0 <= h < 20.0) else 294.15


def paper_trule(scenario, dz_now, Tz, Tout, Qsol, price=None,
                trule_global_offset=0.3, enable_price=True):
    """~40-rule teacher target temperature."""
    occ = is_occupied(dz_now)
    T_heat_sp_K = comfort_bounds_kelvin(dz_now)
    TspC = float(T_heat_sp_K) - 273.15
    sc = str(scenario).lower()
    if sc == "peak":
        T_rule_C = TspC + 0.5 if occ else 21.2
    else:
        T_rule_C = TspC + 0.2 if occ else 20.5
        if occ:
            if Tz < 21.0: T_rule_C += 0.25
            elif Tz < 21.2: T_rule_C += 0.10
    if Qsol > 300: T_rule_C -= 0.10
    if Qsol > 600: T_rule_C -= 0.20
    if Tout > 12: T_rule_C -= 0.10
    if Tout > 16: T_rule_C -= 0.20
    if Tz > 23.0: T_rule_C -= 0.20
    if Tz > 23.5: T_rule_C -= 0.30
    if Tz < 21.0: T_rule_C += 0.15
    if Tz < 20.6: T_rule_C += 0.20
    if sc == "peak" and Tout < 2.0: T_rule_C += 0.15
    if enable_price and price is not None:
        try:
            if float(price) > 0.30 and Tz > 21.5: T_rule_C -= 0.10
            if float(price) < 0.22 and sc == "peak" and Tz < 22.0: T_rule_C += 0.05
        except:
            pass
    T_rule_C += float(trule_global_offset)
    lo, hi = (20.5, 22.2) if sc == "peak" else (20.0, 21.8)
    return float(np.clip(T_rule_C, lo, hi))


def compute_future_features(dt_now, i, horizon_steps, tout_arr, qsol_arr, price_arr):
    """12-dimensional future feature vector (6h window)."""

    def win(arr, idx, n):
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 0: return np.zeros(n)
        e = min(len(arr), idx + n)
        w = arr[idx:e]
        if len(w) == 0: w = np.array([arr[idx]])
        if len(w) < n: w = np.pad(w, (0, n - len(w)), mode="edge")
        return w

    tw = win(tout_arr, i, horizon_steps)
    qw = win(qsol_arr, i, horizon_steps)
    pw = win(price_arr, i, horizon_steps)
    tl, th = [], []
    for j in range(horizon_steps):
        dt_j = dt_now + timedelta(seconds=float(j) * STEP)
        lo, hi = paper_comfort_bounds(dt_j)
        tl.append(lo)
        th.append(hi)
    return {
        "Tout_mean_6h": float(np.mean(tw)), "Tout_max_6h": float(np.max(tw)),
        "Qsol_mean_6h": float(np.mean(qw)), "Qsol_max_6h": float(np.max(qw)),
        "Qsol_sum_6h": float(np.sum(qw) * STEP / 3600.0),
        "price_mean_6h": float(np.mean(pw)), "price_max_6h": float(np.max(pw)),
        "price_min_6h": float(np.min(pw)),
        "Tlow_mean_6h": float(np.mean(np.asarray(tl))),
        "Tlow_max_6h": float(np.max(np.asarray(tl))),
        "Thigh_mean_6h": float(np.mean(np.asarray(th))),
        "Thigh_min_6h": float(np.min(np.asarray(th))),
    }


def make_tcn_features(Tz, T_rule, Tout, Qsol, price, occ, hour, future_features):
    """23-dimensional TCN_T input features."""
    hr = 2.0 * np.pi * float(hour) / 24.0
    if future_features is None:
        future_features = {
            "Tout_mean_6h": Tout, "Tout_max_6h": Tout,
            "Qsol_mean_6h": Qsol, "Qsol_max_6h": Qsol,
            "Qsol_sum_6h": Qsol * 6.0,
            "price_mean_6h": price, "price_max_6h": price, "price_min_6h": price,
            "Tlow_mean_6h": 21.0, "Tlow_max_6h": 21.0,
            "Thigh_mean_6h": 24.0, "Thigh_min_6h": 24.0,
        }
    return np.array([
        (float(Tz) - 21.0) / 5.0, (float(T_rule) - 21.0) / 5.0,
        (float(Tz) - float(T_rule)) / 3.0, (float(Tout) - 10.0) / 15.0,
        float(Qsol) / 800.0, (float(price) - 0.25) / 0.15,
        1.0 if occ else 0.0, 0.0, 0.30, np.sin(hr), np.cos(hr),
        (float(future_features["Tout_mean_6h"]) - 10.0) / 15.0,
        (float(future_features["Tout_max_6h"]) - 10.0) / 15.0,
        float(future_features["Qsol_mean_6h"]) / 800.0,
        float(future_features["Qsol_max_6h"]) / 800.0,
        float(future_features["Qsol_sum_6h"]) / 4800.0,
        (float(future_features["price_mean_6h"]) - 0.25) / 0.15,
        (float(future_features["price_max_6h"]) - 0.25) / 0.15,
        (float(future_features["price_min_6h"]) - 0.25) / 0.15,
        (float(future_features["Tlow_mean_6h"]) - 21.0) / 5.0,
        (float(future_features["Tlow_max_6h"]) - 21.0) / 5.0,
        (float(future_features["Thigh_mean_6h"]) - 24.0) / 6.0,
        (float(future_features["Thigh_min_6h"]) - 24.0) / 6.0,
    ], dtype=np.float32)


def high_level_teacher(scenario, Tz, T_rule, Tout, Qsol, price, occ, fut):
    """~40-rule teacher label generator for TCN_T training. Output: ΔT ∈ [-0.8, 0.8]."""
    sc = str(scenario).lower()
    d = 0.0
    pv = float(price)
    if sc == "typical": d -= 0.15
    if Tz > 22.5: d -= 0.15
    if Tz > 22.8: d -= 0.25
    if Tz > 23.2: d -= 0.35
    if Tz > 23.6: d -= 0.45
    if Qsol > 250: d -= 0.15
    if Qsol > 500: d -= 0.35
    if Qsol > 700: d -= 0.45
    if Tout > 10: d -= 0.10
    if Tout > 14: d -= 0.20
    if Tout > 17: d -= 0.25
    if Tz < 21.1: d += 0.15
    if Tz < 20.8: d += 0.25
    if Tz < 20.5: d += 0.35
    if sc == "peak" and Tout < 5.0: d += 0.15
    if sc == "peak" and Tout < 0.0: d += 0.20
    sm = Tz - 21.0
    if pv > 0.30 and sm > 0.5: d -= 0.15
    if pv > 0.35 and sm > 0.7: d -= 0.25
    if pv > 0.40 and sm > 1.0: d -= 0.35
    if sc == "peak":
        if pv < 0.22 and Tz < 21.6: d += 0.08
        if pv < 0.20 and Tz < 21.4: d += 0.12
    if sc == "typical" and Tz < 21.0: d += 0.10
    if sc == "typical" and Tz < 20.8: d += 0.20
    if fut is not None:
        tm6 = fut["Tout_max_6h"]; qm6 = fut["Qsol_max_6h"]; qs6 = fut["Qsol_sum_6h"]
        pm6m = fut["price_mean_6h"]; pm6x = fut["price_max_6h"]
        tlm6 = fut["Tlow_max_6h"]; thm6 = fut["Thigh_min_6h"]
        if sc == "typical":
            if thm6 <= 24.5 and qm6 > 400 and Tz > 21.2: d -= 0.20
            if thm6 <= 24.5 and qm6 > 600 and Tz > 21.0: d -= 0.30
            if thm6 <= 24.5 and qs6 > 1800 and Tz > 21.0: d -= 0.25
            if tm6 > 15.0 and Tz > 21.2: d -= 0.15
            if tm6 > 17.0 and Tz > 21.0: d -= 0.25
        if sc == "peak":
            if qm6 > 600 and Tz > 22.0: d -= 0.10
            if tm6 > 8.0 and Tz > 22.0: d -= 0.10
        if pm6m > 0.30 and Tz > 21.5: d -= 0.12
        if pm6x > 0.35 and Tz > 21.7: d -= 0.18
        if pm6x > 0.40 and Tz > 22.0: d -= 0.25
        if sc == "peak":
            sp = pm6x - pv
            if sp > 0.08 and pv < 0.24 and Tz < 21.6 and tlm6 >= 21.0 and qm6 < 500: d += 0.08
            if sp > 0.12 and pv < 0.22 and Tz < 21.4 and tlm6 >= 21.0 and qm6 < 400: d += 0.12
            if tlm6 >= 21.0 and Tz < 21.2: d += 0.20
            if tlm6 >= 21.0 and Tz < 21.0: d += 0.30
    if sc == "peak":
        if Tz < 21.30: d = max(d, 0.0)
        elif Tz < 21.60: d = max(d, -0.10)
        elif Tz < 21.80: d = max(d, -0.20)
        if fut is not None:
            if fut.get("Tlow_max_6h", 0) >= 21.0 and Tz < 21.50: d = max(d, 0.0)
            elif fut.get("Tlow_max_6h", 0) >= 21.0 and Tz < 21.80: d = max(d, -0.10)
    return float(np.clip(d, -0.8, 0.8))
