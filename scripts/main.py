"""Main entry point — run best method (L3: TCN_T + TCN_U full) on Typical and Peak scenarios.

Parameters are read from configs/default.yaml (single source of truth).

Usage:
    python main.py                          # run Typical + Peak
    python main.py --scenario Typical       # Typical only
    python main.py --scenario Peak          # Peak only
    python main.py --config configs/default.yaml
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

# project root for imports
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from common.boptest_client import BoptestClient, tout_to_c, tout_to_c_array, FORECAST_PRICE_POINT
from method.rule_teacher import (
    paper_comfort_bounds,
    is_occupied,
    paper_trule,
    compute_future_features,
    make_tcn_features,
    high_level_teacher,
)
from method.networks import PureTCN, TCNU, TBuffer, DualLabelBuf, build_seq
from method.controllers import AdaptivePID
from method.rls import RLSIdentifier
from method.tcnu_teacher import (
    make_tcnu_features,
    compute_gain_labels,
    action_teacher_delta_u,
)
from method.scheduler import set_seed, beta_sched, alpha_u_sched
from method.action import paper_action

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
TCN_INPUT_DIM = 23
TCN_U_INPUT_DIM = 16
DEFAULT_CONFIG = os.path.join(_HERE, "configs", "default.yaml")

CSV_COLS = [
    "time", "day", "hour", "Tz", "Tlow", "Thigh", "Tout", "Qsol", "price",
    "T_rule", "T_set", "e", "I", "D", "u_pid", "u_final", "u_prev",
    "delta_T_teacher", "delta_T_nn", "beta", "loss_T",
    "rp", "ri", "rd", "u_ff_nn", "alpha_u", "loss_U",
    "upper_violation", "lower_violation", "upper_tdis_step", "lower_tdis_step",
]

# ---------------------------------------------------------------------------
# config loader
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # coerce known numeric fields (YAML safe_load may parse 1e-5 as string)
    for key in cfg:
        if isinstance(cfg[key], str):
            try:
                cfg[key] = float(cfg[key])
            except (ValueError, TypeError):
                pass
    for section in ("tcn_t", "tcn_u", "beta", "alpha_u", "pid", "rls"):
        if section in cfg and isinstance(cfg[section], dict):
            for k, v in cfg[section].items():
                if isinstance(v, str):
                    try:
                        cfg[section][k] = float(v)
                    except (ValueError, TypeError):
                        pass
    cfg["tcn_t"]["dt_max_neg"] = float(cfg["tcn_t"].get("dt_max_neg", -0.8))
    cfg["tcn_u"]["ff_scale"] = float(cfg["tcn_u"].get("ff_scale", 0.03))
    return cfg


# ---------------------------------------------------------------------------
# main run function
# ---------------------------------------------------------------------------

def run(scenario, cfg):
    """Run L3 full method (TCN_T + TCN_U dual-head) for one scenario."""

    seed = cfg["seed"]
    step_period = cfg["step_period"]
    spd = cfg["steps_per_day"]
    total_steps = cfg["total_steps"]
    warmup_hours = cfg["warmup_hours"]
    days = total_steps // spd

    # --- scenario settings ---
    start_day = {"Typical": 108, "Peak": 16}[scenario]
    if scenario == "Peak":
        # override Peak-specific settings
        beta_warmup_val = 96
        beta_ramp_val = 192
    else:
        beta_warmup_val = cfg["beta"]["warmup"]
        beta_ramp_val = cfg["beta"]["ramp"]

    start_s = start_day * 86400
    warmup_s = warmup_hours * 3600

    set_seed(seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rid = f"exp7_L3_{scenario}_{ts}"

    # --- results directories ---
    res_dir = os.path.join(_HERE, "results")
    traj_dir = os.path.join(res_dir, "trajectories")
    raw_dir = os.path.join(res_dir, "raw")
    tables_dir = os.path.join(res_dir, "tables")
    for d in [traj_dir, raw_dir, tables_dir]:
        os.makedirs(d, exist_ok=True)

    rdir = os.path.join(traj_dir, rid)
    os.makedirs(rdir, exist_ok=True)

    # --- BOPTEST ---
    print(f"\n{'=' * 60}\n  L3 Full Method | {scenario} | seed={seed}\n{'=' * 60}", flush=True)

    client = BoptestClient(step_period=step_period)
    client.start()
    client.initialize(start_time=start_s, warmup_period=warmup_s)
    client.set_step(step_period)
    client.set_scenario({"electricity_price": cfg["electricity_price"]})

    r = client.last_res
    tz = tout_to_c(r.get("reaTZon_y", 293.15))
    to_cur = tout_to_c(r.get("weaSta_reaWeaTDryBul_y", 283.15))
    qs_cur = float(r.get("weaSta_reaWeaHGloHor_y", 0.0))

    # forecast
    fc = client.get_forecast(
        point_names=["TDryBul", "HGloHor", FORECAST_PRICE_POINT],
        horizon=days * 86400,
        interval=step_period,
    ) or {}
    ta_full = tout_to_c_array(fc.get("TDryBul", []))
    qa_full = np.asarray(fc.get("HGloHor", []), dtype=float)
    pa_full = np.asarray(fc.get(FORECAST_PRICE_POINT, []), dtype=float)

    def pad(a, n, v):
        a = np.asarray(a, dtype=float)
        return a[:n] if len(a) >= n else np.concatenate(
            [a, np.full(n - len(a), a[-1] if len(a) else v)]
        )

    ta_full = pad(ta_full, total_steps, 10.0)
    qa_full = pad(qa_full, total_steps, 0.0)
    pa_full = pad(pa_full, total_steps, 0.25)

    base_date = datetime(2019, 1, 1)

    # --- TCN_T ---
    tcn_t_cfg = cfg["tcn_t"]
    seq_len = tcn_t_cfg["seq_len"]
    model = PureTCN(
        input_dim=TCN_INPUT_DIM,
        tcn_channels=(tcn_t_cfg["ch0"], tcn_t_cfg["ch1"]),
        dropout=tcn_t_cfg.get("dropout", 0.05),
        dt_max=tcn_t_cfg.get("dt_max", 0.8),
    )
    optimizer_t = optim.Adam(
        model.parameters(),
        lr=tcn_t_cfg["lr"],
        weight_decay=tcn_t_cfg["weight_decay"],
    )
    buffer_t = TBuffer(maxlen=50000)
    feat_hist = deque(maxlen=seq_len)

    # --- TCN_U ---
    tcn_u_cfg = cfg["tcn_u"]
    tcnu_seq_len = tcn_u_cfg["seq_len"]
    tcnu_model = TCNU(
        input_dim=TCN_U_INPUT_DIM,
        tcn_channels=(tcn_u_cfg["ch0"], tcn_u_cfg["ch1"]),
        rmax=tcn_u_cfg["rmax"],
        ff_scale=tcn_u_cfg["ff_scale"],
    )
    optimizer_u = optim.Adam(
        tcnu_model.parameters(), lr=tcn_u_cfg["lr"], weight_decay=1e-5
    )
    buffer_u = DualLabelBuf(maxlen=10000)
    tcnu_hist = deque(maxlen=tcnu_seq_len)

    # --- RLS + PID ---
    rls = RLSIdentifier(forgetting=cfg["rls"]["forgetting"])
    phi_prev = None
    theta_now = np.array([0.95, 0.03, 0.05, 0.30, 0.0])
    pid = AdaptivePID()

    # --- state ---
    u_prev_val = 0.0
    u_pid_val = 0.0
    train_t_steps = 0
    train_u_steps = 0

    # --- trajectory CSV ---
    cp = os.path.join(rdir, "details.csv")
    cf = open(cp, "w", newline="")
    cw = csv.DictWriter(cf, fieldnames=CSV_COLS)
    cw.writeheader()

    cold_tot = 0
    hot_tot = 0
    ut_sum = 0.0
    lt_sum = 0.0
    e_sum = 0.0
    t0 = time.time()

    # ================================================================
    # main control loop
    # ================================================================
    for step in range(total_steps):
        ct = start_s + step * step_period
        dt_now = base_date + timedelta(seconds=ct)
        cp_val = float(pa_full[step]) if step < len(pa_full) else 0.25
        to_c = float(ta_full[step]) if step < len(ta_full) else to_cur
        qs_c = float(qa_full[step]) if step < len(qa_full) else qs_cur

        t_low, t_high = paper_comfort_bounds(dt_now)
        occ = is_occupied(dt_now)

        # T_rule (with config-driven offset and price toggle)
        T_rule = paper_trule(
            scenario,
            dt_now,
            tz,
            to_c,
            qs_c,
            cp_val,
            trule_global_offset=cfg.get("trule_global_offset", 0.3),
            enable_price=cfg.get("enable_price", True),
        )

        # --- TCN_T: temperature setpoint correction ---
        fut = compute_future_features(
            dt_now, step, tcn_t_cfg["seq_len"], ta_full, qa_full, pa_full
        )
        delta_T_teacher = high_level_teacher(
            scenario, tz, T_rule, to_c, qs_c, cp_val, occ, fut
        )

        feat = make_tcn_features(
            tz, T_rule, to_c, qs_c, cp_val, occ,
            dt_now.hour + dt_now.minute / 60.0, fut,
        )
        feat_hist.append(feat)
        x_seq_np = build_seq(feat_hist, seq_len, TCN_INPUT_DIM)
        buffer_t.add(x_seq_np, delta_T_teacher)

        delta_T_nn = 0.0
        with torch.no_grad():
            dT_raw = model(
                torch.tensor(x_seq_np[None, :, :], dtype=torch.float32)
            )
            delta_T_nn = max(float(dT_raw.item()), tcn_t_cfg["dt_max_neg"])

        loss_T = None
        if (
            step > 0
            and step % tcn_t_cfg["train_interval"] == 0
            and len(buffer_t) >= tcn_t_cfg["min_buffer"]
        ):
            xseqs_np, yTs_np = buffer_t.sample(tcn_t_cfg["batch_size"])
            xs_t = torch.tensor(xseqs_np, dtype=torch.float32)
            yT = torch.tensor(yTs_np, dtype=torch.float32)
            model.train()
            dTp = model(xs_t)
            loss = torch.nn.functional.mse_loss(dTp, yT)
            optimizer_t.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_t.step()
            model.eval()
            loss_T = float(loss.item())
            train_t_steps += 1

        # β schedule → blend T_rule + TCN_T correction
        beta_val = beta_sched(step, beta_warmup_val, beta_ramp_val, cfg["beta"]["max"])
        lo_clip = 20.0 if scenario == "Typical" else 20.5
        hi_clip = 21.8 if scenario == "Typical" else 22.2
        T_set = float(
            np.clip(T_rule + beta_val * delta_T_nn, lo_clip, hi_clip)
        )

        # --- RLS update ---
        if phi_prev is not None:
            rls.update(phi_prev, tz)
            theta_now = rls.theta.copy()

        # --- TCN_U: PID gain correction + feedforward ---
        rp = 0.0
        ri = 0.0
        rd = 0.0
        u_ff_nn = 0.0
        loss_U = None
        tcnu_feat = make_tcnu_features(
            tz, T_set, T_rule, to_c, qs_c, cp_val, occ,
            dt_now.hour + dt_now.minute / 60.0,
            u_pid_val, 0.0, u_pid_val, u_prev_val,
        )
        tcnu_hist.append(tcnu_feat)
        tcnu_x_seq = build_seq(tcnu_hist, tcnu_seq_len, TCN_U_INPUT_DIM)
        with torch.no_grad():
            r_vec, u_ff_vec = tcnu_model(
                torch.tensor(tcnu_x_seq[None, :, :], dtype=torch.float32)
            )
            rp = float(r_vec[0, 0].item())
            ri = float(r_vec[0, 1].item())
            rd = float(r_vec[0, 2].item())
            u_ff_nn = float(u_ff_vec[0, 0].item())

        # PID step
        e = T_set - tz
        u_pid, u_raw, I_val, D_val, kp_eff, ki_eff, kd_eff = pid.step(
            e, rp, ri, rd
        )
        u_pid_val = u_pid
        e_sum += float(e)

        # TCN_U teacher labels
        gain_label = compute_gain_labels(tz, t_low, t_high, D_val)
        u_ff_teacher = action_teacher_delta_u(
            scenario, tz, T_set, t_low, t_high,
            to_c, qs_c, cp_val, theta_now, u_pid, u_prev_val,
        )
        buffer_u.add(tcnu_x_seq, gain_label, u_ff_teacher)

        # TCN_U training
        if (
            step > 0
            and step % tcn_u_cfg["train_interval"] == 0
            and len(buffer_u) >= tcn_u_cfg["min_buffer"]
        ):
            xseqs_np, gains_np, ffs_np = buffer_u.sample(tcn_t_cfg["batch_size"])
            xs_t = torch.tensor(xseqs_np, dtype=torch.float32)
            gT = torch.tensor(gains_np, dtype=torch.float32)
            fT = torch.tensor(ffs_np, dtype=torch.float32)
            tcnu_model.train()
            rp_vec, uff_vec = tcnu_model(xs_t)
            loss = torch.nn.functional.mse_loss(
                rp_vec, gT
            ) + torch.nn.functional.mse_loss(uff_vec, fT)
            optimizer_u.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(tcnu_model.parameters(), 1.0)
            optimizer_u.step()
            tcnu_model.eval()
            loss_U = float(loss.item())
            train_u_steps += 1

        # --- α_u schedule + safety gates ---
        alpha_u_val = alpha_u_sched(
            step,
            cfg["alpha_u"]["warmup"],
            cfg["alpha_u"]["ramp"],
            cfg["alpha_u"]["max"],
        )
        margin_l = tz - t_low
        margin_h = t_high - tz
        action_gate = 1.0
        if margin_l < 0.6 and u_ff_nn < 0:
            u_ff_nn = 0.0
        if margin_h < 0.6 and u_ff_nn > 0:
            u_ff_nn = 0.0
        if margin_l < 0.3 or margin_h < 0.3:
            action_gate = 0.0
        if u_pid <= 0.02 and u_ff_nn < 0:
            u_ff_nn = 0.0
        if scenario == "Typical" and tz > 23.5 and u_pid <= 0.02:
            action_gate = 0.0
        alpha_u_eff = alpha_u_val * action_gate
        u_final = float(np.clip(u_pid + alpha_u_eff * u_ff_nn, 0.0, 1.0))

        # --- advance simulation ---
        ad = paper_action(float(u_final), T_set)
        rr = client.advance(ad)
        tz_n = tout_to_c(rr.get("reaTZon_y", 293.15))
        to_n = tout_to_c(rr.get("weaSta_reaWeaTDryBul_y", 283.15))
        qs_n = float(rr.get("weaSta_reaWeaHGloHor_y", 0.0))

        uv = 1 if tz_n > t_high else 0
        lv = 1 if tz_n < t_low else 0
        ud = max(0.0, tz_n - t_high) * 0.25
        ld = max(0.0, t_low - tz_n) * 0.25
        cold_tot += lv
        hot_tot += uv
        ut_sum += ud
        lt_sum += ld

        # --- log trajectory ---
        dy = (ct - start_s) / 86400.0
        hr = (ct % 86400) / 3600.0
        cw.writerow(
            {
                k: v
                for k, v in zip(
                    CSV_COLS,
                    [
                        ct, round(dy, 4), round(hr, 4), round(tz_n, 4),
                        round(t_low, 2), round(t_high, 2),
                        round(to_n, 4), round(qs_n, 2), round(cp_val, 6),
                        round(T_rule, 4), round(T_set, 4),
                        round(e, 4), round(I_val, 4), round(D_val, 6),
                        round(u_pid, 6), round(float(u_final), 6),
                        round(u_prev_val, 6),
                        round(delta_T_teacher, 4), round(delta_T_nn, 4),
                        round(beta_val, 3),
                        round(loss_T, 6) if loss_T is not None else "",
                        round(rp, 4), round(ri, 4), round(rd, 4),
                        round(u_ff_nn, 4), round(alpha_u_eff, 3),
                        round(loss_U, 6) if loss_U is not None else "",
                        uv, lv, round(ud, 6), round(ld, 6),
                    ],
                )
            }
        )

        if (step + 1) % spd == 0:
            el = (time.time() - t0) / 3600.0
            print(
                f"  Day {(step + 1) // spd}/{days} Tz={tz_n:.2f} "
                f"Trule={T_rule:.2f} Kp={kp_eff:.3f} FF={u_ff_nn:.4f} "
                f"Tset={T_set:.2f} u={u_final:.3f} "
                f"hot={uv} cold={lv} | {el:.1f}h",
                flush=True,
            )

        phi_now = rls.make_phi(tz, to_c, qs_c, u_final)
        phi_prev = phi_now
        tz, to_cur, qs_cur, u_prev_val = tz_n, to_n, qs_n, u_final

    # ================================================================
    # end of loop — save results
    # ================================================================
    cf.close()
    kpis = client.get_kpis()
    client.stop()
    mean_e = e_sum / total_steps
    elapsed_min = (time.time() - t0) / 60.0
    print(f"\n[Done] {scenario} in {elapsed_min:.1f} min", flush=True)

    # --- KPI JSON ---
    kd = {
        "method": "L3",
        "full_name": "TCN_T + TCN_U full → AdpPID+FF",
        "scenario": scenario,
        "seed": seed,
        "tcn_t_ch0": tcn_t_cfg["ch0"],
        "tcn_t_ch1": tcn_t_cfg["ch1"],
        "tcn_t_lr": tcn_t_cfg["lr"],
        "tcn_t_train_interval": tcn_t_cfg["train_interval"],
        "tcn_u_ch0": tcn_u_cfg["ch0"],
        "tcn_u_ch1": tcn_u_cfg["ch1"],
        "tcn_u_lr": tcn_u_cfg["lr"],
        "tcn_u_rmax": tcn_u_cfg["rmax"],
        "tcn_u_ff_scale": tcn_u_cfg["ff_scale"],
        "beta_warmup": beta_warmup_val,
        "beta_ramp": beta_ramp_val,
        "alpha_u_warmup": cfg["alpha_u"]["warmup"],
        "alpha_u_ramp": cfg["alpha_u"]["ramp"],
        "safety_rules": cfg.get("safety_rules", False),
        "tdis": round(float(kpis.get("tdis_tot", 0)), 4),
        "cost": round(float(kpis.get("cost_tot", 0)), 6),
        "energy": round(float(kpis.get("ener_tot", 0)), 4),
        "emissions": round(float(kpis.get("emis_tot", 0)), 4),
        "mean_e": round(mean_e, 4),
        "upper_tdis": round(ut_sum, 4),
        "lower_tdis": round(lt_sum, 4),
        "upper_n": hot_tot,
        "lower_n": cold_tot,
        "train_t_steps": train_t_steps,
        "train_u_steps": train_u_steps,
        "run_id": rid,
    }
    json_path = os.path.join(raw_dir, f"{rid}_kpi.json")
    with open(json_path, "w") as f:
        json.dump(kd, f, indent=2)

    # --- results CSV ---
    csv_path = os.path.join(tables_dir, "results_kpi.csv")
    csv_cols = [
        "method", "scenario", "seed",
        "tcn_t_ch0", "tcn_t_ch1", "tcn_t_lr", "tcn_t_train_interval",
        "tcn_u_ch0", "tcn_u_ch1", "tcn_u_lr", "tcn_u_rmax", "tcn_u_ff_scale",
        "beta_warmup", "beta_ramp", "alpha_u_warmup", "alpha_u_ramp",
        "tdis", "cost", "energy", "emissions", "mean_e",
        "upper_tdis", "lower_tdis", "upper_n", "lower_n",
        "train_t_steps", "train_u_steps", "run_id",
    ]
    ex = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        if not ex:
            w.writeheader()
        w.writerow({k: kd.get(k, "") for k in csv_cols})

    print(
        f"[KPI] L3 {scenario}: tdis={kd['tdis']:.4f} cost={kd['cost']:.6f} "
        f"energy={kd['energy']:.4f} train_t={train_t_steps} train_u={train_u_steps}",
        flush=True,
    )
    print(f"[Saved] {json_path}", flush=True)
    return kd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run best method (L3 full) on BOPTEST scenarios."
    )
    parser.add_argument(
        "--scenario", default="all",
        choices=["Typical", "Peak", "all"],
        help="Which scenario to run (default: all)"
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG})"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[Config] loaded from {args.config}", flush=True)
    print(f"  seed={cfg['seed']}  tcn_t ch={cfg['tcn_t']['ch0']}/{cfg['tcn_t']['ch1']}  "
          f"tcn_u ch={cfg['tcn_u']['ch0']}/{cfg['tcn_u']['ch1']}  "
          f"rmax={cfg['tcn_u']['rmax']}  ff_scale={cfg['tcn_u']['ff_scale']}", flush=True)

    scenarios = ["Typical", "Peak"] if args.scenario == "all" else [args.scenario]

    all_kpis = {}
    for sn in scenarios:
        all_kpis[sn] = run(sn, cfg)

    # --- summary ---
    print(f"\n{'=' * 60}")
    print("  Results Summary")
    print(f"{'=' * 60}")
    for sn, kd in all_kpis.items():
        print(
            f"  {sn:8s}  tdis={kd['tdis']:8.4f}  cost={kd['cost']:10.6f}  "
            f"energy={kd['energy']:8.4f}  upper_n={kd['upper_n']:3d}  "
            f"lower_n={kd['lower_n']:3d}"
        )
    print(f"{'=' * 60}\nDone.")
