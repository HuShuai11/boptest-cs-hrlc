"""【消融 L0】T_rule+0.3 + Paper PID (baseline, no neural networks)."""
import csv, json, os, sys, time
from datetime import datetime, timedelta
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

from common.boptest_client import BoptestClient, tout_to_c, tout_to_c_array
from method.rule_teacher import paper_comfort_bounds, paper_trule
from method.controllers import PaperPID
from method.action import paper_action
from method.scheduler import set_seed

STEP = 900; SPD = 96; SEED = 123

CSV_COLS = ["time", "day", "hour", "Tz", "Tlow", "Thigh", "Tout", "Qsol", "price",
            "T_rule", "T_set", "e", "I", "D", "u", "u_raw", "u_prev",
            "upper_violation", "lower_violation", "upper_tdis_step", "lower_tdis_step"]


def run(scenario="Typical", days=14, warmup_hours=168):
    set_seed(SEED)
    start_day = {"Typical": 108, "Peak": 16}[scenario]
    start_s = start_day * 86400; total_steps = days * SPD; warmup_s = warmup_hours * 3600
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rid = f"ablation_L0_{scenario}_{ts}"

    client = BoptestClient(step_period=STEP); client.start()
    client.initialize(start_time=start_s, warmup_period=warmup_s)
    client.set_step(STEP); client.set_scenario({"electricity_price": "highly_dynamic"})

    r = client.last_res
    tz = tout_to_c(r.get("reaTZon_y", 293.15))
    to_cur = tout_to_c(r.get("weaSta_reaWeaTDryBul_y", 283.15))
    qs_cur = float(r.get("weaSta_reaWeaHGloHor_y", 0.0))

    fc = client.get_forecast(point_names=["TDryBul", "HGloHor", "PriceElectricPowerHighlyDynamic"],
                             horizon=14 * 86400, interval=STEP) or {}
    ta_full = tout_to_c_array(fc.get("TDryBul", []))
    qa_full = np.asarray(fc.get("HGloHor", []), dtype=float)
    pa_full = np.asarray(fc.get("PriceElectricPowerHighlyDynamic", []), dtype=float)
    def pad(a, n, v):
        a = np.asarray(a, dtype=float)
        return a[:n] if len(a) >= n else np.concatenate([a, np.full(n - len(a), a[-1] if len(a) else v)])
    ta_full = pad(ta_full, total_steps, 10.0); qa_full = pad(qa_full, total_steps, 0.0)
    pa_full = pad(pa_full, total_steps, 0.25)

    base_date = datetime(2019, 1, 1); pid = PaperPID()
    u_prev_val = 0.0

    _ABL = os.path.dirname(_HERE)
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

        t_low, t_high = paper_comfort_bounds(dt_now)
        T_rule = paper_trule(scenario, dt_now, tz, to_c, qs_c, cp_val)
        T_set = T_rule  # no TCN

        e = T_set - tz; u, u_raw, I_val, D_val = pid.step(e)
        e_sum += float(e)

        ad = paper_action(float(u), T_set)
        rr = client.advance(ad)
        tz_n = tout_to_c(rr.get("reaTZon_y", 293.15))
        to_n = tout_to_c(rr.get("weaSta_reaWeaTDryBul_y", 283.15))
        qs_n = float(rr.get("weaSta_reaWeaHGloHor_y", 0.0))

        uv = 1 if tz_n > t_high else 0; lv = 1 if tz_n < t_low else 0
        ud = max(0.0, tz_n - t_high) * 0.25; ld = max(0.0, t_low - tz_n) * 0.25
        cold_tot += lv; hot_tot += uv; ut_sum += ud; lt_sum += ld

        dy = (ct - start_s) / 86400.0; hr = (ct % 86400) / 3600.0
        cw.writerow({k: v for k, v in zip(CSV_COLS, [
            ct, round(dy, 4), round(hr, 4), round(tz_n, 4), round(t_low, 2), round(t_high, 2),
            round(to_n, 4), round(qs_n, 2), round(cp_val, 6), round(T_rule, 4), round(T_set, 4),
            round(e, 4), round(I_val, 4), round(D_val, 6), round(float(u), 6),
            round(float(u_raw), 6), round(u_prev_val, 6),
            uv, lv, round(ud, 6), round(ld, 6)])})

        if (step + 1) % SPD == 0:
            el = (time.time() - t0) / 3600.0
            print(f"  Day {(step + 1) // SPD}/{days} Tz={tz_n:.2f} Trule={T_rule:.2f} "
                  f"u={u:.3f} hot={uv} cold={lv} | {el:.1f}h", flush=True)

        tz, to_cur, qs_cur, u_prev_val = tz_n, to_n, qs_n, u

    cf.close(); kpis = client.get_kpis(); client.stop()
    mean_e = e_sum / total_steps
    print(f"\n[Done] {scenario} in {(time.time() - t0) / 60:.1f} min")

    rd = os.path.join(_ABL, "raw"); os.makedirs(rd, exist_ok=True)
    kd = {"ablation": "L0", "level": 0, "method": "T_rule+PID (no NN)",
          "scenario": scenario, "seed": SEED,
          "tdis": round(float(kpis.get("tdis_tot", 0)), 4),
          "cost": round(float(kpis.get("cost_tot", 0)), 6),
          "energy": round(float(kpis.get("ener_tot", 0)), 4),
          "emissions": round(float(kpis.get("emis_tot", 0)), 4),
          "mean_e": round(mean_e, 4),
          "upper_tdis": round(ut_sum, 4), "lower_tdis": round(lt_sum, 4),
          "upper_n": hot_tot, "lower_n": cold_tot, "run_id": rid}
    with open(os.path.join(rd, f"{rid}_kpi.json"), "w") as f: json.dump(kd, f, indent=2)

    sp = os.path.join(_ABL, "tables", "ablation_kpi.csv")
    cols = ["ablation", "level", "method", "scenario", "seed",
            "tdis", "cost", "energy", "emissions", "mean_e",
            "upper_tdis", "lower_tdis", "upper_n", "lower_n", "run_id"]
    os.makedirs(os.path.dirname(sp), exist_ok=True)
    ex = os.path.exists(sp)
    with open(sp, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not ex: w.writeheader()
        w.writerow({k: kd.get(k, "") for k in cols})

    print(f"[KPI] L0 {scenario}: tdis={kd['tdis']:.4f} cost={kd['cost']:.6f}")
    return kd


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="Typical", choices=["Typical", "Peak", "all"])
    scs = ["Typical", "Peak"] if vars(p.parse_args())["scenario"] == "all" else [p.parse_args().scenario]
    for sn in scs:
        print(f"\n{'=' * 60}\n  Ablation L0: T_rule+PID | {sn}\n{'=' * 60}")
        run(sn)
    print("\nL0 DONE.")
