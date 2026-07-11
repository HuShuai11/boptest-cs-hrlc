"""【消融 L2】T_rule + TCN_T(ch=32/64) + TCN_U gain only (rmax=0.25, ff=0) → AdpPID."""
import csv, json, os, sys, time, random
from collections import deque
from datetime import datetime, timedelta
import numpy as np
import torch, torch.nn as nn, torch.optim as optim

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

from common.boptest_client import BoptestClient, tout_to_c, tout_to_c_array, FORECAST_PRICE_POINT
from method.rule_teacher import paper_comfort_bounds, is_occupied, paper_trule, compute_future_features, make_tcn_features, high_level_teacher
from method.networks import PureTCN, TCNU, TBuffer, DualLabelBuf, build_seq
from method.controllers import AdaptivePID
from method.rls import RLSIdentifier
from method.tcnu_teacher import make_tcnu_features, compute_gain_labels, action_teacher_delta_u
from method.scheduler import set_seed, beta_sched
from method.action import paper_action

STEP = 900; SPD = 96; TCN_INPUT_DIM = 23; TCN_U_INPUT_DIM = 16; SEED = 123

FROZEN_TCN = dict(lr=0.003, tcn_ch0=32, tcn_ch1=64, beta_warmup=24, beta_ramp=96,
                  train_interval=4, batch_size=64, weight_decay=1e-5, dt_max_neg=-0.8)
FROZEN_TCNU = dict(lr=0.0001, ch0=16, ch1=32, rmax=0.25, ff=0.0, uw=288, ur=288, ti=4)

CSV_COLS = ["time", "day", "hour", "Tz", "Tlow", "Thigh", "Tout", "Qsol", "price",
            "T_rule", "T_set", "e", "I", "D", "u_pid", "u_final", "u_prev",
            "delta_T_teacher", "delta_T_nn", "beta", "loss_T",
            "rp", "ri", "rd", "u_ff_nn", "alpha_u", "loss_U",
            "upper_violation", "lower_violation", "upper_tdis_step", "lower_tdis_step"]


def run(scenario="Typical", days=14, warmup_hours=168):
    set_seed(SEED)
    start_day = {"Typical": 108, "Peak": 16}[scenario]
    start_s = start_day * 86400; total_steps = days * SPD; warmup_s = warmup_hours * 3600
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rid = f"ablation_L2_{scenario}_{ts}"
    _ABL = os.path.dirname(_HERE)

    client = BoptestClient(step_period=STEP); client.start()
    client.initialize(start_time=start_s, warmup_period=warmup_s)
    client.set_step(STEP); client.set_scenario({"electricity_price": "highly_dynamic"})

    r = client.last_res
    tz = tout_to_c(r.get("reaTZon_y", 293.15)); to_cur = tout_to_c(r.get("weaSta_reaWeaTDryBul_y", 283.15))
    qs_cur = float(r.get("weaSta_reaWeaHGloHor_y", 0.0))

    fc = client.get_forecast(point_names=["TDryBul", "HGloHor", FORECAST_PRICE_POINT],
                             horizon=14 * 86400, interval=STEP) or {}
    ta_full = tout_to_c_array(fc.get("TDryBul", []))
    qa_full = np.asarray(fc.get("HGloHor", []), dtype=float)
    pa_full = np.asarray(fc.get(FORECAST_PRICE_POINT, []), dtype=float)

    def pad(a, n, v):
        a = np.asarray(a, dtype=float)
        return a[:n] if len(a) >= n else np.concatenate([a, np.full(n - len(a), a[-1] if len(a) else v)])
    ta_full = pad(ta_full, total_steps, 10.0); qa_full = pad(qa_full, total_steps, 0.0)
    pa_full = pad(pa_full, total_steps, 0.25)

    base_date = datetime(2019, 1, 1)

    seq_len = 24
    model = PureTCN(input_dim=TCN_INPUT_DIM, tcn_channels=(FROZEN_TCN["tcn_ch0"], FROZEN_TCN["tcn_ch1"]))
    optimizer = optim.Adam(model.parameters(), lr=FROZEN_TCN["lr"], weight_decay=FROZEN_TCN["weight_decay"])
    buffer = TBuffer(maxlen=50000); feat_hist = deque(maxlen=seq_len)

    tcnu_seq_len = 12
    tcnu_model = TCNU(input_dim=TCN_U_INPUT_DIM, tcn_channels=(FROZEN_TCNU["ch0"], FROZEN_TCNU["ch1"]),
                      rmax=FROZEN_TCNU["rmax"], ff_scale=FROZEN_TCNU["ff"])
    tcnu_opt = optim.Adam(tcnu_model.parameters(), lr=FROZEN_TCNU["lr"], weight_decay=1e-5)
    tcnu_buf = DualLabelBuf(maxlen=10000); tcnu_hist = deque(maxlen=tcnu_seq_len)

    rls = RLSIdentifier(forgetting=0.998)
    phi_prev = None; pred_base_prev = None
    theta_now = np.array([0.95, 0.03, 0.05, 0.30, 0.0])

    pid = AdaptivePID()
    u_prev_val = 0.0; u_pid_val = 0.0; train_t = 0; train_u = 0

    tdir = os.path.join(_ABL, "trajectories"); os.makedirs(tdir, exist_ok=True)
    rdir = os.path.join(tdir, rid); os.makedirs(rdir, exist_ok=True)
    cp = os.path.join(rdir, "details.csv"); cf = open(cp, "w", newline="")
    cw = csv.DictWriter(cf, fieldnames=CSV_COLS); cw.writeheader()

    cold_tot = 0; hot_tot = 0; ut_sum = 0.0; lt_sum = 0.0; t0 = time.time()
    e_sum = 0.0

    for step in range(total_steps):
        ct = start_s + step * STEP; dt_now = base_date + timedelta(seconds=ct)
        cp_val = float(pa_full[step]) if step < len(pa_full) else 0.25
        to_c = float(ta_full[step]) if step < len(ta_full) else to_cur
        qs_c = float(qa_full[step]) if step < len(qa_full) else qs_cur

        t_low, t_high = paper_comfort_bounds(dt_now); occ = is_occupied(dt_now)
        T_rule = paper_trule(scenario, dt_now, tz, to_c, qs_c, cp_val)
        T_rule_ref = paper_trule(scenario, dt_now, tz, to_c, qs_c, cp_val)

        fut = compute_future_features(dt_now, step, 24, ta_full, qa_full, pa_full)
        delta_T_teacher = high_level_teacher(scenario, tz, T_rule_ref, to_c, qs_c, cp_val, occ, fut)

        feat = make_tcn_features(tz, T_rule, to_c, qs_c, cp_val, occ, dt_now.hour + dt_now.minute / 60.0, fut)
        feat_hist.append(feat); x_seq_np = build_seq(feat_hist, seq_len, TCN_INPUT_DIM)
        buffer.add(x_seq_np, delta_T_teacher)

        delta_T_nn = 0.0
        with torch.no_grad():
            dT_raw = model(torch.tensor(x_seq_np[None, :, :], dtype=torch.float32))
            delta_T_nn = max(float(dT_raw.item()), FROZEN_TCN["dt_max_neg"])

        loss_T = None
        if step > 0 and step % FROZEN_TCN["train_interval"] == 0 and len(buffer) >= 96:
            xseqs_np, yTs_np = buffer.sample(FROZEN_TCN["batch_size"])
            xs_t = torch.tensor(xseqs_np, dtype=torch.float32); yT = torch.tensor(yTs_np, dtype=torch.float32)
            model.train(); dTp = model(xs_t)
            loss = torch.nn.functional.mse_loss(dTp, yT)
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); model.eval(); loss_T = float(loss.item()); train_t += 1

        if str(scenario).lower() == "typical":
            beta_val = beta_sched(step, FROZEN_TCN["beta_warmup"], FROZEN_TCN["beta_ramp"], 1.0)
        else:
            beta_val = beta_sched(step, 96, 192, 1.0)
        T_set = float(np.clip(T_rule + beta_val * delta_T_nn,
                              20.0 if str(scenario).lower() == "typical" else 20.5,
                              21.8 if str(scenario).lower() == "typical" else 22.2))

        if phi_prev is not None: rls.update(phi_prev, tz); theta_now = rls.theta.copy()

        rp = 0.0; ri = 0.0; rd = 0.0; u_ff_nn = 0.0; loss_U = None
        tcnu_feat = make_tcnu_features(tz, T_set, T_rule, to_c, qs_c, cp_val, occ,
                                       dt_now.hour + dt_now.minute / 60.0,
                                       u_pid_val, 0.0, u_pid_val, u_prev_val)
        tcnu_hist.append(tcnu_feat)
        tcnu_x_seq = build_seq(tcnu_hist, tcnu_seq_len, TCN_U_INPUT_DIM)
        with torch.no_grad():
            r_vec, u_ff_vec = tcnu_model(torch.tensor(tcnu_x_seq[None, :, :], dtype=torch.float32))
            rp = float(r_vec[0, 0].item()); ri = float(r_vec[0, 1].item())
            rd = float(r_vec[0, 2].item()); u_ff_nn = 0.0  # L2: ff forced to 0

        e = T_set - tz
        u_pid, u_raw, I_val, D_val, kp_eff, ki_eff, kd_eff = pid.step(e, rp, ri, rd)
        u_pid_val = u_pid; e_sum += float(e)

        gain_label = compute_gain_labels(tz, t_low, t_high, D_val)
        u_ff_teacher = action_teacher_delta_u(scenario, tz, T_set, t_low, t_high,
                                              to_c, qs_c, cp_val, theta_now, u_pid, u_prev_val)
        tcnu_buf.add(tcnu_x_seq, gain_label, u_ff_teacher)

        if step > 0 and step % FROZEN_TCNU["ti"] == 0 and len(tcnu_buf) >= 96:
            xseqs_np, gains_np, ffs_np = tcnu_buf.sample(FROZEN_TCN["batch_size"])
            xs_t = torch.tensor(xseqs_np, dtype=torch.float32)
            gT = torch.tensor(gains_np, dtype=torch.float32); fT = torch.tensor(ffs_np, dtype=torch.float32)
            tcnu_model.train(); rp_vec, uff_vec = tcnu_model(xs_t)
            loss = torch.nn.functional.mse_loss(rp_vec, gT) + torch.nn.functional.mse_loss(uff_vec, fT)
            tcnu_opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(tcnu_model.parameters(), 1.0)
            tcnu_opt.step(); tcnu_model.eval()
            loss_U = float(loss.item()); train_u += 1

        # L2: NO forward gating, use u_pid directly
        u_final = float(np.clip(u_pid, 0.0, 1.0))
        alpha_u_eff = 0.0

        ad = paper_action(float(u_final), T_set)
        rr = client.advance(ad)
        tz_n = tout_to_c(rr.get("reaTZon_y", 293.15)); to_n = tout_to_c(rr.get("weaSta_reaWeaTDryBul_y", 283.15))
        qs_n = float(rr.get("weaSta_reaWeaHGloHor_y", 0.0))

        uv = 1 if tz_n > t_high else 0; lv = 1 if tz_n < t_low else 0
        ud = max(0.0, tz_n - t_high) * 0.25; ld = max(0.0, t_low - tz_n) * 0.25
        cold_tot += lv; hot_tot += uv; ut_sum += ud; lt_sum += ld

        dy = (ct - start_s) / 86400.0; hr = (ct % 86400) / 3600.0
        cw.writerow({k: v for k, v in zip(CSV_COLS, [
            ct, round(dy, 4), round(hr, 4), round(tz_n, 4), round(t_low, 2), round(t_high, 2),
            round(to_n, 4), round(qs_n, 2), round(cp_val, 6), round(T_rule, 4), round(T_set, 4),
            round(e, 4), round(I_val, 4), round(D_val, 6), round(u_pid, 6), round(float(u_final), 6),
            round(u_prev_val, 6), round(delta_T_teacher, 4), round(delta_T_nn, 4), round(beta_val, 3),
            round(loss_T, 6) if loss_T else "",
            round(rp, 4), round(ri, 4), round(rd, 4), round(u_ff_nn, 4), round(alpha_u_eff, 3),
            round(loss_U, 6) if loss_U else "", uv, lv, round(ud, 6), round(ld, 6)])})

        if (step + 1) % SPD == 0:
            el = (time.time() - t0) / 3600.0
            print(f"  Day {(step + 1) // SPD}/{days} Tz={tz_n:.2f} Trule={T_rule:.2f} "
                  f"Kp={kp_eff:.3f} Tset={T_set:.2f} u={u_final:.3f} dT={delta_T_nn:.3f} "
                  f"hot={uv} cold={lv} | {el:.1f}h", flush=True)

        phi_now = rls.make_phi(tz, to_c, qs_c, u_final)
        pred_base_now = rls.predict(phi_now); phi_prev = phi_now; pred_base_prev = pred_base_now
        tz, to_cur, qs_cur, u_prev_val = tz_n, to_n, qs_n, u_final

    cf.close(); kpis = client.get_kpis(); client.stop()
    mean_e = e_sum / total_steps
    print(f"\n[Done] {scenario} in {(time.time() - t0) / 60:.1f} min")

    rd = os.path.join(_ABL, "raw"); os.makedirs(rd, exist_ok=True)
    kd = {"ablation": "L2", "level": 2, "method": "TCN_T + TCN_U gain-only → AdpPID",
          "scenario": scenario, "seed": SEED,
          "tcn_ch0": FROZEN_TCN["tcn_ch0"], "tcn_ch1": FROZEN_TCN["tcn_ch1"],
          "lr": FROZEN_TCN["lr"], "tcnu_rmax": FROZEN_TCNU["rmax"], "tcnu_ff_scale": 0.0,
          "tdis": round(float(kpis.get("tdis_tot", 0)), 4),
          "cost": round(float(kpis.get("cost_tot", 0)), 6),
          "energy": round(float(kpis.get("ener_tot", 0)), 4),
          "emissions": round(float(kpis.get("emis_tot", 0)), 4),
          "mean_e": round(mean_e, 4),
          "upper_tdis": round(ut_sum, 4), "lower_tdis": round(lt_sum, 4),
          "upper_n": hot_tot, "lower_n": cold_tot,
          "train_t_steps": train_t, "train_u_steps": train_u, "run_id": rid}
    with open(os.path.join(rd, f"{rid}_kpi.json"), "w") as f: json.dump(kd, f, indent=2)

    sp = os.path.join(_ABL, "tables", "ablation_kpi.csv")
    cols = ["ablation", "level", "method", "scenario", "seed",
            "tcn_ch0", "tcn_ch1", "lr", "tcnu_rmax", "tcnu_ff_scale",
            "tdis", "cost", "energy", "emissions", "mean_e",
            "upper_tdis", "lower_tdis", "upper_n", "lower_n",
            "train_t_steps", "train_u_steps", "run_id"]
    os.makedirs(os.path.dirname(sp), exist_ok=True)
    ex = os.path.exists(sp)
    with open(sp, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not ex: w.writeheader()
        w.writerow({k: kd.get(k, "") for k in cols})

    print(f"[KPI] L2 {scenario}: tdis={kd['tdis']:.4f} cost={kd['cost']:.6f}")
    return kd


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Ablation L2: TCN_U gain-only")
    p.add_argument("--scenario", default="Typical", choices=["Typical", "Peak", "all"])
    scs = ["Typical", "Peak"] if vars(p.parse_args())["scenario"] == "all" else [p.parse_args().scenario]
    for sn in scs:
        print(f"\n{'=' * 60}\n  Ablation L2: TCN_U gain-only | {sn}\n{'=' * 60}")
        run(sn)
    print("\nL2 DONE.")
