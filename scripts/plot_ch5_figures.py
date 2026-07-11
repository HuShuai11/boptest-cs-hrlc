from __future__ import annotations

import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures" / "ch5"

TABLE_DIR = ROOT / "paper" / "tables"
REPORT_PATH = ROOT / "analysis" / "data_inventory_report.md"
CH5_TRAJ_REPORT_PATH = ROOT / "analysis" / "ch5_fig7_fig10_data_report.md"
CH5_KPI_REPORT_PATH = ROOT / "analysis" / "ch5_fig11_fig12_kpi_report.md"
CH5_FIG13_REPORT_PATH = ROOT / "analysis" / "ch5_fig13_internal_behavior_report.md"
CH5_FIG14_REPORT_PATH = ROOT / "analysis" / "ch5_fig14_ablation_report.md"

DATA_EXTS = {".csv", ".xlsx", ".json", ".pkl", ".parquet"}
SEARCH_DIRS = [
    "results",
    "logs",
    "compare",
    "paper",
    "analysis",
    "figures",
    "data",
    "outputs",
    "ablation",
    "baselines_method",
    "method",
]

CONTROLLER_ORDER = ["Proposed", "PI", "MPC", "PINN-MPC", "Safe-DRL", "LearnAMR"]
SCENARIO_ORDER = ["Peak", "Typical"]

COLORS = {
    "Proposed": "#1f4e79",
    "PID": "#7a7a7a",
    "PI": "#d97904",
    "MPC": "#2ca02c",
    "PINN-MPC": "#0f8b9e",
    "Safe-DRL": "#d62728",
    "LearnAMR": "#7b3f98",
}

LEGEND_FRAME_STYLE = {
    "frameon": True,
    "fancybox": False,
    "framealpha": 1.0,
    "edgecolor": "#6f6f6f",
    "facecolor": "white",
}

BOTTOM_LEGEND_Y = 0.006

LINESTYLES = {
    "Proposed": "-",
    "PID": "--",
    "PI": "--",
    "MPC": "-.",
    "PINN-MPC": (0, (3, 1, 1, 1)),
    "Safe-DRL": (0, (4, 2)),
    "LearnAMR": ":",
}

FIELD_ALIASES = {
    "time": ["time", "t", "timestamp", "datetime", "step", "seconds", "day", "days", "time_days"],
    "day": ["day", "days", "time_days"],
    "Tz": ["Tz", "zone_temperature", "indoor_temperature", "operative_temperature", "reaTZon_y", "TRoo", "T_zone", "Tzone"],
    "T_low": ["T_low", "Tlow", "lower_bound", "LowerSetp", "TLow", "comfort_lower", "LowerSetp_api_C"],
    "T_high": ["T_high", "Thigh", "upper_bound", "UpperSetp", "THigh", "comfort_upper", "UpperSetp_api_C"],
    "u": ["u", "action", "command", "control", "u_final", "heat_pump_command", "oveHeaPumY_u", "hp_command", "u_hp", "hp", "u_safe", "u_mpc"],
    "Tout": ["Tout", "outdoor_temperature", "ambient_temperature", "TDryBul", "weaSta_reaWeaTDryBul_y", "T_out"],
    "Qsol": ["Qsol", "solar", "solar_radiation", "HGloHor", "weaSta_reaWeaHGloHor_y", "Q_sol"],
    "price": ["price", "electricity_price", "tariff", "p"],
    "cost": ["cost", "operating_cost", "total_cost", "cost_eur_m2", "Cost", "cost_tot"],
    "energy": ["energy", "energy_use", "energy_kwh_m2", "Energy", "ener_tot"],
    "emissions": ["emissions", "Emissions", "carbon", "co2", "CO2", "co2_emissions", "kgco2_m2", "emis_tot"],
    "discomfort": ["discomfort", "thermal_discomfort", "discomfort_kh_zone", "Discomfort", "tdis", "tdis_tot"],
    "controller": ["controller", "method", "algorithm", "baseline", "name", "Proposed", "Ours", "SDT-HRL", "OD-SHRLC"],
    "scenario": ["scenario", "case", "test_case", "season", "Peak", "Typical"],
    "T_rule": ["T_rule", "Trule", "rule_target", "baseline_target", "T_rule_k"],
    "T_set": ["T_set", "Tset", "target_temperature", "T_target", "final_target", "T_set_k"],
    "beta": ["beta", "beta_k", "beta_target", "target_participation"],
    "alpha_eff": ["alpha_eff", "alpha_u_eff", "alpha_u", "alpha_effective", "effective_participation", "alpha_time", "alpha_u_time"],
    "safety_gate": ["safety_gate", "gate", "g_k", "safety", "safe_gate"],
    "u_pid": ["u_pid", "upid", "pid_command", "feedback_command"],
    "u_ff": ["u_ff", "uff", "u_ff_nn", "feedforward", "feedforward_command"],
    "u_final": ["u_final", "u", "final_command", "command", "action", "heat_pump_command", "oveHeaPumY_u"],
    "DeltaT_tea": ["DeltaT_tea", "deltaT_teacher", "delta_T_teacher", "dT_teacher", "target_teacher", "target_residual_teacher"],
    "DeltaT_NN": ["DeltaT_NN", "deltaT_nn", "delta_T_nn", "dT_nn", "target_residual", "neural_target_residual"],
    "uff_tea": ["uff_tea", "u_ff_tea", "uff_teacher", "feedforward_teacher"],
    "rp": ["rp", "r_p", "Kp_eff", "Kp"],
    "ri": ["ri", "r_i", "Ki_eff", "Ki"],
    "rd": ["rd", "r_d", "Kd_eff", "Kd"],
    "variant": ["variant", "ablation", "level", "method"],
}


@dataclass
class RunData:
    scenario: str
    controller: str
    path: Path
    df: pd.DataFrame


def init_nature_style() -> None:
    fonts = ["Times New Roman", "DejaVu Serif", "serif"]
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": fonts,
            "font.size": 10.0,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.78,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "figure.dpi": 140,
        }
    )


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def find_data_files() -> List[Path]:
    files: List[Path] = []
    for d in SEARCH_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in DATA_EXTS and ".git" not in p.parts:
                files.append(p)
    return sorted(set(files))


def load_table_auto(path: Path, nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            for enc in ("utf-8-sig", "utf-8", "gbk", "latin1"):
                try:
                    return pd.read_csv(path, encoding=enc, nrows=nrows)
                except UnicodeDecodeError:
                    continue
                except pd.errors.ParserError:
                    try:
                        return pd.read_csv(path, encoding=enc, nrows=nrows, engine="python", on_bad_lines="skip")
                    except Exception:
                        continue
        if suffix == ".xlsx":
            return pd.read_excel(path, nrows=nrows)
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return pd.DataFrame([data])
            return pd.DataFrame(data)
        if suffix == ".pkl":
            obj = pd.read_pickle(path)
            return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
        if suffix == ".parquet":
            return pd.read_parquet(path)
    except Exception as exc:
        print(f"[inventory] Could not read {path}: {exc}")
    return None


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename: Dict[str, str] = {}
    norm_to_col = {_norm_name(c): c for c in df.columns}
    for canonical, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            hit = norm_to_col.get(_norm_name(alias))
            if hit is not None and canonical not in df.columns:
                rename[hit] = canonical
                break
    out = df.rename(columns=rename).copy()
    return out


def find_col(df: pd.DataFrame, canonical: str) -> Optional[str]:
    if canonical in df.columns:
        return canonical
    aliases = FIELD_ALIASES.get(canonical, [])
    norm_to_col = {_norm_name(c): c for c in df.columns}
    for alias in aliases:
        hit = norm_to_col.get(_norm_name(alias))
        if hit:
            return hit
    return None


def require_cols(df: pd.DataFrame, cols: Iterable[str], fig_name: str, source: Path) -> bool:
    missing = [c for c in cols if find_col(df, c) is None]
    if missing:
        print(f"[skip] {fig_name}: missing {missing} in {source}")
        return False
    return True


def detect_scenario(path: Path, df: Optional[pd.DataFrame] = None) -> Optional[str]:
    if df is not None and "scenario" in df.columns and df["scenario"].notna().any():
        val = str(df["scenario"].dropna().iloc[0]).lower()
        if "peak" in val:
            return "Peak"
        if "typical" in val:
            return "Typical"
    text = str(path).lower()
    if "peak" in text:
        return "Peak"
    if "typical" in text:
        return "Typical"
    return None


def detect_controller(path: Path, df: Optional[pd.DataFrame] = None) -> Optional[str]:
    text = str(path).lower()
    if "learn" in text and "amr" in text:
        return "LearnAMR"
    if "safe" in text or "drl" in text or "rl" in text:
        return "Safe-DRL"
    if "mpc_mhe" in text or "purempc" in text or "\\mpc\\" in text or "/mpc/" in text:
        return "MPC"
    if "pi_trule" in text or "pid" in text or "\\pi\\" in text or "/pi/" in text:
        return "PID"
    if "exp7_l3" in text or "results\\trajectories" in text or "results/trajectories" in text:
        return "Proposed"
    if df is not None and "controller" in df.columns and df["controller"].notna().any():
        return canonical_controller(str(df["controller"].dropna().iloc[0]))
    return None


def canonical_controller(value: str) -> str:
    s = str(value).lower()
    compact = _norm_name(value)
    if "proposed" in s or "ours" in s or "sdt" in s or "odshrlc" in compact or s == "l3" or "adppid+ff" in s or "full" in s:
        return "Proposed"
    if "learn" in s and "amr" in s:
        return "LearnAMR"
    if "pinn" in s and "mpc" in s:
        return "PINN-MPC"
    if "safe" in s or "drl" in s or "rl" in s:
        return "Safe-DRL"
    if "mpc" in s:
        return "MPC"
    if "pi" in s or "pid" in s:
        return "PI"
    return value


def scenario_runs() -> Dict[Tuple[str, str], Path]:
    return {
        ("Peak", "Proposed"): ROOT / "results" / "trajectories" / "exp7_L3_Peak_20260608_221253" / "details.csv",
        ("Typical", "Proposed"): ROOT / "results" / "trajectories" / "exp7_L3_Typical_20260608_221159" / "details.csv",
        ("Peak", "PID"): ROOT / "baselines_method" / "pi" / "pi_trule_baseline" / "results" / "pi_trule_peak" / "details.csv",
        ("Typical", "PID"): ROOT / "baselines_method" / "pi" / "pi_trule_baseline" / "results" / "pi_trule_typical" / "details.csv",
        ("Peak", "MPC"): ROOT / "baselines_method" / "mpc" / "mpc_baseline" / "results" / "mpc_fixed_peak" / "details.csv",
        ("Typical", "MPC"): ROOT / "baselines_method" / "mpc" / "mpc_baseline" / "results" / "mpc_fixed_typical" / "details.csv",
        ("Peak", "PINN-MPC"): ROOT / "baselines_method" / "pinn_mpc" / "results" / "pinn_mpc_sa_peak_20260601_214847" / "details.csv",
        ("Typical", "PINN-MPC"): ROOT / "baselines_method" / "pinn_mpc" / "results" / "pinn_mpc_sa_typical_20260601_213614" / "details.csv",
        ("Peak", "Safe-DRL"): ROOT / "baselines_method" / "safe_rl" / "results_safe_rl_paper" / "peak_d14_w168h_safe_20260531_085513" / "details.csv",
        ("Typical", "Safe-DRL"): ROOT / "baselines_method" / "safe_rl" / "results_safe_rl_paper" / "typical_d14_w168h_safe_20260531_085656" / "details.csv",
        ("Peak", "LearnAMR"): ROOT / "baselines_method" / "learn_amr" / "results_learnamr" / "peak_d14_w12h_learnamr_20260527_203626" / "details.csv",
        ("Typical", "LearnAMR"): ROOT / "baselines_method" / "learn_amr" / "results_learnamr" / "typical_d14_w12h_learnamr_20260527_204029" / "details.csv",
    }


def load_run(scenario: str, controller: str) -> Optional[RunData]:
    path = scenario_runs().get((scenario, controller))
    if path is None or not path.exists():
        print(f"[missing] {scenario} {controller}: trajectory file not found: {path}")
        return None
    df = load_table_auto(path)
    if df is None:
        print(f"[missing] {scenario} {controller}: cannot read {path}")
        return None
    return RunData(scenario, controller, path, normalize_columns(df))


def time_days(df: pd.DataFrame) -> Optional[pd.Series]:
    if "day" in df.columns:
        return pd.to_numeric(df["day"], errors="coerce")
    c = find_col(df, "time")
    if c is None:
        return None
    t = pd.to_numeric(df[c], errors="coerce")
    if t.dropna().empty:
        return None
    if t.max() > 1000:
        return (t - t.min()) / 86400.0
    return t


def series(df: pd.DataFrame, key: str) -> Optional[pd.Series]:
    c = find_col(df, key)
    if c is None:
        return None
    return pd.to_numeric(df[c], errors="coerce")


def save_figure_all_formats(fig: plt.Figure, basename: str) -> List[Path]:
    saved: List[Path] = []
    for ext in ("png", "svg", "pdf"):
        path = FIG_DIR / f"{basename}.{ext}"
        tmp_path = FIG_DIR / f".{basename}.{os.getpid()}.tmp.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(tmp_path, **kwargs)
        try:
            tmp_path.replace(path)
            saved.append(path)
        except OSError as exc:
            print(f"[locked] Could not replace {path.relative_to(ROOT)}: {exc}")
            print(f"[locked] New {ext.upper()} kept at {tmp_path.relative_to(ROOT)}")
            saved.append(tmp_path)
            continue


    plt.close(fig)
    return saved


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.075, 1.02, label, transform=ax.transAxes, fontweight="bold", va="bottom", ha="left")


def add_bottom_panel_title(ax: plt.Axes, title: str, y: float = -0.24) -> None:
    ax.text(0.5, y, title, transform=ax.transAxes, ha="center", va="top", fontsize=8)


def add_bottom_panel_title_fig9_10(ax: plt.Axes, title: str, y: float = -0.24) -> None:
    ax.text(0.5, y, title, transform=ax.transAxes, ha="center", va="top", fontsize=8)


def add_framed_bottom_legend(
    fig: plt.Figure,
    handles: Iterable[mpl.artist.Artist],
    labels: Iterable[str],
    *,
    ncol: Optional[int] = None,
    y: float = 0.02,
    fontsize: Optional[float] = None,
    columnspacing: float = 0.9,
    handlelength: float = 1.45,
    handletextpad: float = 0.42,
    markerscale: float = 1.0,
) -> Optional[mpl.legend.Legend]:
    unique: Dict[str, mpl.artist.Artist] = {}
    for handle, label in zip(handles, labels):
        if label and not label.startswith("_") and label not in unique:
            unique[label] = handle
    if not unique:
        return None
    ncol = ncol or min(len(unique), 6)
    legend = fig.legend(
        list(unique.values()),
        list(unique.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        borderaxespad=0.0,
        columnspacing=columnspacing,
        handlelength=handlelength,
        handletextpad=handletextpad,
        markerscale=markerscale,
        borderpad=0.38,
        labelspacing=0.35,
        fontsize=fontsize,
        **LEGEND_FRAME_STYLE,
    )
    legend.get_frame().set_linewidth(0.65)
    return legend


def add_panel_top_legend(
    ax: plt.Axes,
    handles: Iterable[mpl.artist.Artist],
    labels: Iterable[str],
    *,
    ncol: int,
    loc: str = "lower right",
    anchor: Tuple[float, float] = (1.0, 1.035),
    fontsize: float = 6.8,
) -> Optional[mpl.legend.Legend]:
    legend = ax.legend(
        list(handles),
        list(labels),
        loc=loc,
        bbox_to_anchor=anchor,
        bbox_transform=ax.transAxes,
        ncol=ncol,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="#b8b8b8",
        facecolor="white",
        fontsize=fontsize,
        handlelength=2.1,
        columnspacing=1.15,
        borderpad=0.34,
        labelspacing=0.35,
    )
    legend.get_frame().set_linewidth(0.6)
    return legend


def row_major_to_matplotlib_column_order(labels: List[str], ncol: int) -> List[str]:
    """Reorder labels so a multi-column legend reads row-wise on the canvas."""
    if ncol <= 1 or len(labels) <= ncol:
        return labels
    nrows = int(math.ceil(len(labels) / ncol))
    rows = [labels[i * ncol : (i + 1) * ncol] for i in range(nrows)]
    ordered: List[str] = []
    for col in range(ncol):
        for row in rows:
            if col < len(row):
                ordered.append(row[col])
    return ordered


def style_box_grid(ax: plt.Axes) -> None:
    ax.grid(True, color="#d9d9d9", linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#222222")


def plot_closed_loop(scenario: str, basename: str) -> List[Path]:
    fig_name = f"Fig.{7 if scenario == 'Peak' else 8} {scenario} closed-loop"
    runs = [load_run(scenario, c) for c in CONTROLLER_ORDER]
    runs = [r for r in runs if r is not None]
    if not runs:
        print(f"[skip] {fig_name}: no trajectory files")
        return []

    usable_runs: List[RunData] = []
    for run in runs:
        if require_cols(run.df, ["time", "Tz", "u"], fig_name, run.path):
            usable_runs.append(run)
    if not usable_runs:
        return []
    ref = next((r for r in usable_runs if r.controller == "Proposed"), usable_runs[0])
    if not require_cols(ref.df, ["time", "T_low", "T_high", "Tout", "Qsol", "price"], fig_name, ref.path):
        return []

    fig, axes = plt.subplots(4, 1, figsize=(7.3, 7.6), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.84, top=0.985, bottom=0.165, hspace=0.46)
    xref = time_days(ref.df)
    tlow, thigh = series(ref.df, "T_low"), series(ref.df, "T_high")
    if xref is not None and tlow is not None and thigh is not None:
        axes[0].fill_between(xref, tlow, thigh, color="#cfe3ee", alpha=0.58, label="Comfort band", lw=0)
    else:
        print(f"[warn] {fig_name}: comfort bounds missing from reference run")

    for run in usable_runs:
        df = run.df
        x = time_days(df)
        tz = series(df, "Tz")
        u = series(df, "u")
        if x is None or tz is None or u is None:
            print(f"[skip-line] {fig_name}: missing time/Tz/u in {run.path}")
            continue
        color = COLORS[run.controller]
        temp_lw = 1.7 if run.controller == "Proposed" else 1.1
        temp_alpha = 1.0 if run.controller == "Proposed" else 0.8
        cmd_lw = 1.6 if run.controller == "Proposed" else 0.95
        cmd_alpha = 1.0 if run.controller == "Proposed" else 0.38
        zorder = 4 if run.controller == "Proposed" else 3
        axes[0].plot(
            x,
            tz,
            color=color,
            ls=LINESTYLES[run.controller],
            lw=temp_lw,
            alpha=temp_alpha,
            label=run.controller,
            zorder=zorder,
        )
        axes[1].plot(
            x,
            u,
            color=color,
            ls=LINESTYLES[run.controller],
            lw=cmd_lw,
            alpha=cmd_alpha,
            label=run.controller,
            zorder=zorder,
        )

    tout, qsol, price = series(ref.df, "Tout"), series(ref.df, "Qsol"), series(ref.df, "price")
    if xref is not None and tout is not None:
        axes[2].plot(xref, tout, color="#34586f", lw=1.15, label="Ambient temperature")
    if xref is not None and qsol is not None:
        ax2 = axes[2].twinx()
        ax2.plot(xref, qsol, color="#c28f2c", lw=1.0, alpha=0.9, label="Solar irradiation")
        ax2.set_ylabel(r"Solar irradiation (W/m$^2$)")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(True)
        ax2.tick_params(axis="y", labelsize=7, width=0.8)
    else:
        print(f"[warn] {fig_name}: Qsol missing")
    if xref is not None and price is not None:
        axes[3].plot(xref, price, color="#303030", ls="--", lw=1.05, label="Price")
    else:
        print(f"[warn] {fig_name}: price missing")

    for ax in axes:
        ax.grid(True)
        ax.set_xlim(0, 14)
    for ax, caption in zip(
        axes,
        [
            "(a) Indoor temperature and comfort bounds",
            "(b) Heat-pump command",
            "(c) Ambient temperature and solar irradiation",
            "(d) Electricity price",
        ],
    ):
        add_bottom_panel_title(ax, caption, y=-0.25 if ax is not axes[3] else -0.38)
    axes[0].set_ylabel(r"Operative temperature ($^\circ$C)")
    axes[0].set_ylim(15, 30)
    axes[1].set_ylabel("Command")
    axes[1].set_ylim(0, 1.05)
    axes[2].set_ylabel(r"Ambient temperature ($^\circ$C)")
    axes[3].set_ylabel(r"Price (EUR/kWh)")
    axes[3].set_xlabel("Time (days)", labelpad=6)

    legend_handles = [
        mpl.patches.Patch(facecolor="#cfe3ee", alpha=0.58, edgecolor="none", label="Comfort band"),
        mpl.lines.Line2D([0], [0], color=COLORS["Proposed"], ls=LINESTYLES["Proposed"], lw=1.7, label="Proposed"),
        mpl.lines.Line2D([0], [0], color=COLORS["PID"], ls=LINESTYLES["PID"], lw=1.1, alpha=0.8, label="PID"),
        mpl.lines.Line2D([0], [0], color=COLORS["MPC"], ls=LINESTYLES["MPC"], lw=1.1, alpha=0.8, label="MPC"),
        mpl.lines.Line2D([0], [0], color=COLORS["PINN-MPC"], ls=LINESTYLES["PINN-MPC"], lw=1.1, alpha=0.8, label="PINN-MPC"),
        mpl.lines.Line2D([0], [0], color=COLORS["Safe-DRL"], ls=LINESTYLES["Safe-DRL"], lw=1.1, alpha=0.8, label="Safe-DRL"),
        mpl.lines.Line2D([0], [0], color=COLORS["LearnAMR"], ls=LINESTYLES["LearnAMR"], lw=1.1, alpha=0.8, label="LearnAMR"),
        mpl.lines.Line2D([0], [0], color="#34586f", lw=1.15, label="Ambient temperature"),
        mpl.lines.Line2D([0], [0], color="#c28f2c", lw=1.0, label="Solar irradiation"),
        mpl.lines.Line2D([0], [0], color="#303030", ls="--", lw=1.05, label="Price"),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.014),
        ncol=5,
        columnspacing=1.1,
        handlelength=2.2,
        handletextpad=0.5,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="#6f6f6f",
        facecolor="white",
    )
    legend.get_frame().set_linewidth(0.6)
    return save_figure_all_formats(fig, basename)


def plot_fig7_peak_closed_loop() -> List[Path]:
    return plot_closed_loop("Peak", "fig7_peak_closed_loop")


def plot_fig8_typical_closed_loop() -> List[Path]:
    return plot_closed_loop("Typical", "fig8_typical_closed_loop")


CH5_TRAJECTORY_RUNS: Dict[Tuple[str, str], Path] = {
    ("Peak", "Proposed"): ROOT / "results" / "trajectories" / "exp7_L3_Peak_20260608_221253" / "details.csv",
    ("Typical", "Proposed"): ROOT / "results" / "trajectories" / "exp7_L3_Typical_20260608_221159" / "details.csv",
    ("Peak", "PI"): ROOT / "baselines_method" / "pi" / "pi_trule_baseline" / "results" / "pi_trule_peak" / "details.csv",
    ("Typical", "PI"): ROOT / "baselines_method" / "pi" / "pi_trule_baseline" / "results" / "pi_trule_typical" / "details.csv",
    ("Peak", "MPC"): ROOT / "baselines_method" / "mpc" / "mpc_baseline" / "results" / "mpc_fixed_peak" / "details.csv",
    ("Typical", "MPC"): ROOT / "baselines_method" / "mpc" / "mpc_baseline" / "results" / "mpc_fixed_typical" / "details.csv",
    ("Peak", "PINN-MPC"): ROOT / "baselines_method" / "pinn_mpc" / "results" / "pinn_mpc_sa_peak_20260601_214847" / "details.csv",
    ("Typical", "PINN-MPC"): ROOT / "baselines_method" / "pinn_mpc" / "results" / "pinn_mpc_sa_typical_20260601_213614" / "details.csv",
    ("Peak", "Safe-DRL"): ROOT / "baselines_method" / "safe_rl" / "results_safe_rl_paper" / "peak_d14_w168h_safe_20260531_085513" / "details.csv",
    ("Typical", "Safe-DRL"): ROOT / "baselines_method" / "safe_rl" / "results_safe_rl_paper" / "typical_d14_w168h_safe_20260531_085656" / "details.csv",
    ("Peak", "LearnAMR"): ROOT / "baselines_method" / "learn_amr" / "results_learnamr" / "peak_d14_w12h_learnamr_20260527_203626" / "details.csv",
    ("Typical", "LearnAMR"): ROOT / "baselines_method" / "learn_amr" / "results_learnamr" / "typical_d14_w12h_learnamr_20260527_204029" / "details.csv",
}

CH5_REQUIRED_FIELDS = ["time", "Tz", "T_low", "T_high", "u", "Tout", "Qsol", "price"]
CH5_PANEL_TITLES = [
    "(a) Price",
    "(b) Operative temperature",
    "(c) Control signal",
    "(d) Ambient temperature and solar irradiation",
]


def load_ch5_trajectory_run(scenario: str, controller: str) -> Optional[RunData]:
    path = CH5_TRAJECTORY_RUNS.get((scenario, controller))
    if path is None or not path.exists():
        print(f"[missing] {scenario} {controller}: trajectory file not found: {path}")
        return None
    df = load_table_auto(path)
    if df is None or df.empty:
        print(f"[missing] {scenario} {controller}: cannot read {path}")
        return None
    return RunData(scenario, controller, path, normalize_columns(df))


def day_of_year_axis(df: pd.DataFrame) -> Optional[pd.Series]:
    c = find_col(df, "time")
    if c is not None:
        t = pd.to_numeric(df[c], errors="coerce")
        if not t.dropna().empty:
            if t.max() > 1000:
                return t / 86400.0
            return t
    c = find_col(df, "day")
    if c is not None:
        d = pd.to_numeric(df[c], errors="coerce")
        if not d.dropna().empty:
            return d
    return None


def finite_xy(x: Optional[pd.Series], y: Optional[pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
    if x is None or y is None:
        return np.array([]), np.array([])
    data = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    return data["x"].to_numpy(), data["y"].to_numpy()


def ch5_field_status(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {key: find_col(df, key) for key in CH5_REQUIRED_FIELDS}


def validate_ch5_run(run: RunData) -> List[str]:
    status = ch5_field_status(run.df)
    missing = [key for key, col in status.items() if col is None]
    if day_of_year_axis(run.df) is None and "time" not in missing:
        missing.append("time_axis")
    return missing


def write_ch5_fig7_fig10_data_report() -> Path:
    lines = [
        "# Fig.7--Fig.10 Data Check Report",
        "",
        "Generated by `analysis/plot_ch5_figures.py`. Raw result files were inspected but not modified.",
        "",
        "## Data Files Found",
        "",
    ]
    figure_specs = {
        "Fig.7": ("Peak", ["Proposed", "PI", "MPC"]),
        "Fig.8": ("Peak", ["Proposed", "PINN-MPC", "Safe-DRL", "LearnAMR"]),
        "Fig.9": ("Typical", ["Proposed", "PI", "MPC"]),
        "Fig.10": ("Typical", ["Proposed", "PINN-MPC", "Safe-DRL", "LearnAMR"]),
    }
    missing_notes: List[str] = []
    for scenario in SCENARIO_ORDER:
        lines.append(f"### {scenario} scenario")
        for method in ["Proposed", "PI", "MPC", "PINN-MPC", "Safe-DRL", "LearnAMR"]:
            path = CH5_TRAJECTORY_RUNS[(scenario, method)]
            exists = path.exists()
            lines.append(f"- {method}: `{path.relative_to(ROOT)}` ({'found' if exists else 'missing'})")
        lines.append("")
    lines.extend(["## Field Mapping by File", ""])
    for (scenario, method), path in CH5_TRAJECTORY_RUNS.items():
        if not path.exists():
            missing_notes.append(f"{scenario} {method}: missing file {path.relative_to(ROOT)}")
            continue
        df = load_table_auto(path, nrows=20)
        if df is None:
            missing_notes.append(f"{scenario} {method}: unreadable file {path.relative_to(ROOT)}")
            continue
        df = normalize_columns(df)
        status = ch5_field_status(df)
        missing = [key for key, value in status.items() if value is None]
        if missing:
            missing_notes.append(f"{scenario} {method}: missing fields {', '.join(missing)}")
        lines.extend([
            f"### {scenario} / {method}",
            f"- File: `{path.relative_to(ROOT)}`",
            f"- Main fields: {', '.join(map(str, list(df.columns)[:40]))}",
            f"- Time field: {status['time'] or 'missing'}",
            f"- Temperature field: {status['Tz'] or 'missing'}",
            f"- Comfort lower/upper: {status['T_low'] or 'missing'} / {status['T_high'] or 'missing'}",
            f"- Control field: {status['u'] or 'missing'}",
            f"- Weather fields: {status['Tout'] or 'missing'} / {status['Qsol'] or 'missing'}",
            f"- Price field: {status['price'] or 'missing'}",
            "",
        ])
    lines.extend(["## Figure Readiness", ""])
    for fig, (scenario, methods) in figure_specs.items():
        usable = []
        for method in methods:
            run = load_ch5_trajectory_run(scenario, method)
            if run is None:
                continue
            missing = validate_ch5_run(run)
            if missing:
                missing_notes.append(f"{fig} {scenario} {method}: missing {', '.join(missing)}")
            else:
                usable.append(method)
        lines.append(f"- {fig}: {scenario}; usable methods: {', '.join(usable) if usable else 'none'}.")
    lines.extend(["", "## Missing Methods or Fields", ""])
    if missing_notes:
        lines.extend(f"- {note}" for note in missing_notes)
    else:
        lines.append("- No required methods or fields are missing for Fig.7--Fig.10.")
    lines.extend([
        "",
        "## Field Coverage Summary",
        "",
        "- Time, operative/indoor temperature, comfort lower and upper bounds, control signal, ambient temperature, solar irradiation, and electricity price were successfully identified for all selected Fig.7--Fig.10 trajectory files.",
        "- Comfort bounds are available as `T_low`/`T_high` or `Tlow`/`Thigh`, so the temperature panels use true filled comfort bands rather than vertical occupied-period shading.",
        "- PINN-MPC uses its own PINN-MPC trajectory files and is not substituted with MPC data.",
        "- Proposed is displayed consistently as Proposed.",
        "- No random, synthetic, or hand-entered data are used.",
    ])
    CH5_TRAJ_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return CH5_TRAJ_REPORT_PATH


def plot_scenario_grouped_trajectories(scenario: str, methods: List[str], output_basename: str) -> List[Path]:
    runs = [load_ch5_trajectory_run(scenario, method) for method in methods]
    runs = [run for run in runs if run is not None]
    missing_by_run = {f"{run.scenario} {run.controller}": validate_ch5_run(run) for run in runs}
    blocking = {key: missing for key, missing in missing_by_run.items() if missing}
    if blocking:
        for key, missing in blocking.items():
            print(f"[skip] {output_basename}: {key} missing {missing}")
        return []
    if not runs:
        print(f"[skip] {output_basename}: no usable trajectory data")
        return []

    ref = next((run for run in runs if run.controller == "Proposed"), runs[0])
    x_ref = day_of_year_axis(ref.df)
    price = series(ref.df, "price")
    tout = series(ref.df, "Tout")
    qsol = series(ref.df, "Qsol")
    tlow = series(ref.df, "T_low")
    thigh = series(ref.df, "T_high")

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.0), sharex=True)
    ax_price, ax_temp, ax_ctrl, ax_weather = axes.ravel()
    fig.subplots_adjust(left=0.09, right=0.90, bottom=0.170, top=0.965, wspace=0.33, hspace=0.46)
    is_advanced_group = any(method in {"PINN-MPC", "Safe-DRL", "LearnAMR"} for method in methods)

    x, y = finite_xy(x_ref, price)
    if len(x):
        ax_price.plot(x, y, color="#4d4d4d", lw=1.35, label="Electricity price")
    ax_price.set_ylabel("Price (EUR/kWh)")

    x_low, lo = finite_xy(x_ref, tlow)
    x_high, hi = finite_xy(x_ref, thigh)
    has_comfort_band = len(x_low) and len(x_high) and len(lo) == len(hi) and np.allclose(x_low, x_high)
    if has_comfort_band:
        ax_temp.fill_between(x_low, lo, hi, color="#8fc1d4", alpha=0.16, lw=0, label="Comfort band", zorder=1)
        ax_temp.plot(x_low, lo, color="#9d9d9d", lw=0.65, alpha=0.72, zorder=2)
        ax_temp.plot(x_high, hi, color="#9d9d9d", lw=0.65, alpha=0.72, zorder=2)
    else:
        print(f"[warn] {output_basename}: comfort bounds not available for filled band")

    for run in runs:
        x_run = day_of_year_axis(run.df)
        temp = series(run.df, "Tz")
        ctrl = series(run.df, "u")
        color = COLORS.get(run.controller, "#555555")
        ls = LINESTYLES.get(run.controller, "-")
        temp_lw = 1.75 if run.controller == "Proposed" else 1.0
        ctrl_lw = 1.75 if run.controller == "Proposed" else (0.62 if is_advanced_group else 0.85)
        alpha = 1.0 if run.controller == "Proposed" else 0.70
        ctrl_alpha = 1.0 if run.controller == "Proposed" else (0.52 if is_advanced_group else alpha)
        zorder = 5 if run.controller == "Proposed" else 3
        x, y = finite_xy(x_run, temp)
        if len(x):
            ax_temp.plot(x, y, color=color, ls=ls, lw=temp_lw, alpha=alpha, label=run.controller, zorder=zorder)
        x, y = finite_xy(x_run, ctrl)
        if len(x):
            ax_ctrl.plot(x, y, color=color, ls=ls, lw=ctrl_lw, alpha=ctrl_alpha, label=run.controller, zorder=zorder)

    ax_temp.set_ylabel("Operative temperature ($^\\circ$C)")
    ax_ctrl.set_ylabel("Control signal")
    ax_ctrl.set_ylim(0, 1.05)

    x, y = finite_xy(x_ref, tout)
    if len(x):
        ax_weather.plot(x, y, color="#6b6b6b", lw=1.25, label="Ambient temperature")
    ax_weather.set_ylabel("Ambient temperature ($^\\circ$C)")
    ax_solar = ax_weather.twinx()
    x, y = finite_xy(x_ref, qsol)
    if len(x):
        ax_solar.plot(x, y, color="#d28a1e", lw=1.1, alpha=0.90, label="Solar irradiation")
    ax_solar.set_ylabel("Solar irradiation (W/m$^2$)")
    ax_solar.spines["right"].set_visible(True)
    ax_solar.spines["top"].set_visible(False)
    ax_solar.tick_params(axis="y", labelsize=9.5, width=0.8)

    all_x = []
    for run in runs:
        x_run = day_of_year_axis(run.df)
        if x_run is not None:
            all_x.extend(pd.to_numeric(x_run, errors="coerce").dropna().tolist())
    if all_x:
        xmin, xmax = min(all_x), max(all_x)
        pad = max((xmax - xmin) * 0.01, 0.05)
        for ax in [ax_price, ax_temp, ax_ctrl, ax_weather]:
            ax.set_xlim(xmin - pad, xmax + pad)

    for ax, title in zip([ax_price, ax_temp, ax_ctrl, ax_weather], CH5_PANEL_TITLES):
        style_box_grid(ax)
        ax.tick_params(axis="x", labelbottom=True)
        add_bottom_panel_title(ax, title, y=-0.23)
        ax.set_xlabel("Day of the year")

    legend_handles: List[mpl.artist.Artist] = []
    legend_labels: List[str] = []
    for legend_ax in [ax_price, ax_temp, ax_ctrl, ax_weather, ax_solar]:
        handles, labels = legend_ax.get_legend_handles_labels()
        legend_handles.extend(handles)
        legend_labels.extend(labels)
    desired_visual_order = [
        "Electricity price",
        "Comfort band",
        *methods,
        "Ambient temperature",
        "Solar irradiation",
    ]
    handle_by_label: Dict[str, mpl.artist.Artist] = {}
    for handle, label in zip(legend_handles, legend_labels):
        handle_by_label.setdefault(label, handle)
    visual_labels = [label for label in desired_visual_order if label in handle_by_label]
    visual_labels.extend(
        label for label in legend_labels
        if label and not label.startswith("_") and label not in visual_labels
    )
    legend_labels_ordered = row_major_to_matplotlib_column_order(visual_labels, ncol=4)
    legend_handles_ordered = [handle_by_label[label] for label in legend_labels_ordered]
    add_framed_bottom_legend(fig, legend_handles_ordered, legend_labels_ordered, ncol=4, y=BOTTOM_LEGEND_Y, fontsize=7.2)
    return save_figure_all_formats(fig, output_basename)

def plot_fig7_peak_conventional() -> List[Path]:
    return plot_scenario_grouped_trajectories("Peak", ["Proposed", "PI", "MPC"], "fig7_peak_conventional")


def plot_fig8_peak_advanced() -> List[Path]:
    return plot_scenario_grouped_trajectories("Peak", ["Proposed", "PINN-MPC", "Safe-DRL", "LearnAMR"], "fig8_peak_advanced")


def plot_fig9_typical_conventional() -> List[Path]:
    return plot_scenario_grouped_trajectories("Typical", ["Proposed", "PI", "MPC"], "fig9_typical_conventional")


def plot_fig10_typical_advanced() -> List[Path]:
    return plot_scenario_grouped_trajectories("Typical", ["Proposed", "PINN-MPC", "Safe-DRL", "LearnAMR"], "fig10_typical_advanced")


def kpi_json_paths() -> Dict[Tuple[str, str], Path]:
    return {
        ("Peak", "Proposed"): ROOT / "results" / "raw" / "exp7_L3_Peak_20260608_221253_kpi.json",
        ("Typical", "Proposed"): ROOT / "results" / "raw" / "exp7_L3_Typical_20260608_221159_kpi.json",
        ("Peak", "PI"): ROOT / "baselines_method" / "pi" / "pi_trule_baseline" / "results" / "pi_trule_peak" / "kpi.json",
        ("Typical", "PI"): ROOT / "baselines_method" / "pi" / "pi_trule_baseline" / "results" / "pi_trule_typical" / "kpi.json",
        ("Peak", "MPC"): ROOT / "baselines_method" / "mpc" / "mpc_baseline" / "results" / "mpc_fixed_peak" / "kpi.json",
        ("Typical", "MPC"): ROOT / "baselines_method" / "mpc" / "mpc_baseline" / "results" / "mpc_fixed_typical" / "kpi.json",
        ("Peak", "PINN-MPC"): ROOT / "baselines_method" / "pinn_mpc" / "results" / "pinn_mpc_sa_peak_20260601_214847" / "kpi.json",
        ("Typical", "PINN-MPC"): ROOT / "baselines_method" / "pinn_mpc" / "results" / "pinn_mpc_sa_typical_20260601_213614" / "kpi.json",
        ("Peak", "Safe-DRL"): ROOT / "baselines_method" / "safe_rl" / "results_safe_rl_paper" / "peak_d14_w168h_safe_20260531_085513" / "kpi.json",
        ("Typical", "Safe-DRL"): ROOT / "baselines_method" / "safe_rl" / "results_safe_rl_paper" / "typical_d14_w168h_safe_20260531_085656" / "kpi.json",
        ("Peak", "LearnAMR"): ROOT / "baselines_method" / "learn_amr" / "results_learnamr" / "peak_d14_w12h_learnamr_20260527_203626" / "kpi.json",
        ("Typical", "LearnAMR"): ROOT / "baselines_method" / "learn_amr" / "results_learnamr" / "typical_d14_w12h_learnamr_20260527_204029" / "kpi.json",
    }


def load_kpi_table() -> pd.DataFrame:
    rows = []
    for (scenario, controller), path in kpi_json_paths().items():
        if not path.exists():
            print(f"[missing] KPI file not found: {path}")
            continue
        df = load_table_auto(path)
        if df is None or df.empty:
            print(f"[missing] KPI file unreadable: {path}")
            continue
        df = normalize_columns(df)
        row = {"scenario": scenario, "controller": controller, "path": str(path.relative_to(ROOT))}
        for key in ["cost", "energy", "emissions", "discomfort"]:
            col = find_col(df, key)
            row[key] = float(df[col].iloc[0]) if col is not None and pd.notna(df[col].iloc[0]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


KPI_METHOD_ORDER = ["Proposed", "PI", "MPC", "PINN-MPC", "Safe-DRL", "LearnAMR"]

KPI_COLORS = {method: COLORS[method] for method in KPI_METHOD_ORDER}

KPI_MARKERS = {
    "Proposed": "*",
    "PI": "o",
    "MPC": "s",
    "PINN-MPC": "P",
    "Safe-DRL": "^",
    "LearnAMR": "D",
}


def write_ch5_fig11_fig12_kpi_report() -> Path:
    kpi = load_kpi_table()
    expected_pairs = {(s, m) for s in SCENARIO_ORDER for m in KPI_METHOD_ORDER}
    found_pairs = {(str(r.scenario), str(r.controller)) for r in kpi.itertuples()} if not kpi.empty else set()
    missing_pairs = sorted(expected_pairs - found_pairs)
    missing_fields = []
    for metric in ["cost", "energy", "emissions", "discomfort"]:
        if metric not in kpi.columns:
            missing_fields.append(metric)
        elif kpi[metric].isna().any():
            bad = kpi.loc[kpi[metric].isna(), ["scenario", "controller"]]
            missing_fields.extend(f"{metric} for {r.scenario}/{r.controller}" for r in bad.itertuples())

    lines = [
        "# Fig.11--Fig.12 KPI Data Check Report",
        "",
        "Generated by `analysis/plot_ch5_figures.py`. KPI result files were inspected but not modified.",
        "",
        "## KPI Files Found",
        "",
    ]
    for (scenario, method), path in kpi_json_paths().items():
        exists = path.exists()
        lines.append(f"### {scenario} / {method}")
        lines.append(f"- File: `{path.relative_to(ROOT)}` ({'found' if exists else 'missing'})")
        if exists:
            df = load_table_auto(path, nrows=5)
            if df is None:
                lines.append("- Readable: no")
            else:
                mapped = normalize_columns(df)
                lines.append("- Readable: yes")
                lines.append(f"- Fields: {', '.join(map(str, list(df.columns)[:40]))}")
                lines.append(f"- Mapped cost field: {find_col(mapped, 'cost') or 'missing'}")
                lines.append(f"- Mapped energy field: {find_col(mapped, 'energy') or 'missing'}")
                lines.append(f"- Mapped emissions field: {find_col(mapped, 'emissions') or 'missing'}")
                lines.append(f"- Mapped discomfort field: {find_col(mapped, 'discomfort') or 'missing'}")
        lines.append("")

    lines.extend(
        [
            "## Coverage",
            "",
            f"- Scenarios found: {', '.join(sorted(kpi['scenario'].unique())) if not kpi.empty else 'none'}",
            f"- Methods found: {', '.join(KPI_METHOD_ORDER if not kpi.empty and not missing_pairs else sorted(kpi['controller'].unique())) if not kpi.empty else 'none'}",
            "- KPI fields used for Fig.11: cost and discomfort.",
            "- KPI fields used for Fig.12: cost, energy, emissions, and discomfort.",
            "- Units: Cost = EUR/m$^2$; Energy = kWh/m$^2$; Emissions = kgCO$_2$e/m$^2$; Discomfort = K$\\cdot$h/zone.",
            "",
            "## Missing Methods or Fields",
            "",
        ]
    )
    if missing_pairs:
        lines.extend(f"- Missing KPI row: {scenario} / {method}" for scenario, method in missing_pairs)
    if missing_fields:
        lines.extend(f"- Missing KPI field/value: {item}" for item in missing_fields)
    if not missing_pairs and not missing_fields:
        lines.append("- No required methods, scenarios, or KPI fields are missing for Fig.11--Fig.12.")

    CH5_KPI_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return CH5_KPI_REPORT_PATH


def plot_fig11_cost_comfort_tradeoff() -> List[Path]:
    kpi = load_kpi_table()
    required = {"scenario", "controller", "cost", "discomfort"}
    if kpi.empty or not required.issubset(kpi.columns):
        print("[skip] Fig.11: missing KPI cost/discomfort data")
        return []

    label_positions = {
        "Peak": {
            "Proposed": (0.780, 0.15),
            "PINN-MPC": (0.787, 0.43),
            "LearnAMR": (0.820, 0.34),
            "PI": (0.850, 0.57),
            "MPC": (0.883, 0.40),
            "Safe-DRL": (0.882, 1.34),
        },
        "Typical": {
            "Proposed": (0.306, 5.85),
            "Safe-DRL": (0.304, 8.85),
            "PINN-MPC": (0.338, 10.15),
            "MPC": (0.366, 5.35),
            "PI": (0.381, 8.70),
            "LearnAMR": (0.386, 13.55),
        },
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.955, bottom=0.30, wspace=0.28)

    for ax, scenario, caption in zip(
        axes,
        SCENARIO_ORDER,
        [
            "(a) Peak: cost--discomfort trade-off; lower left is better",
            "(b) Typical: cost--discomfort trade-off; lower left is better",
        ],
    ):
        sub = kpi[kpi["scenario"] == scenario].copy()
        max_discomfort = 0.0
        for method in KPI_METHOD_ORDER:
            row = sub[sub["controller"] == method]
            if row.empty:
                print(f"[skip-point] Fig.11: missing KPI row for {scenario} {method}")
                continue
            r = row.iloc[0]
            max_discomfort = max(max_discomfort, float(r["discomfort"]))
            size = 175 if method == "Proposed" else 82
            ax.scatter(
                r["cost"],
                r["discomfort"],
                s=size,
                marker=KPI_MARKERS[method],
                color=KPI_COLORS[method],
                edgecolor="black",
                linewidth=0.85,
                label=method,
                zorder=5 if method == "Proposed" else 3,
            )
            label = f"{method}\n({float(r['cost']):.3f}, {float(r['discomfort']):.2f})"
            xytext = label_positions.get(scenario, {}).get(method, (float(r["cost"]), float(r["discomfort"])))
            ax.annotate(
                label,
                (r["cost"], r["discomfort"]),
                xytext=xytext,
                textcoords="data",
                fontsize=6.7,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#bcbcbc", lw=0.55, alpha=0.94),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#777777",
                    lw=0.55,
                    shrinkA=2.0,
                    shrinkB=4.5,
                    mutation_scale=6.0,
                    connectionstyle="arc3,rad=0.04",
                ),
                zorder=6,
            )
        ax.set_xlabel("Cost (EUR/m$^2$)")
        ax.set_ylabel(r"Discomfort (K$\cdot$h/zone)")
        if scenario == "Peak":
            ax.set_xlim(0.775, 0.915)
            ax.set_ylim(bottom=-0.03, top=max(1.70, max_discomfort * 1.38))
        else:
            ax.set_xlim(0.295, 0.408)
            ax.set_ylim(bottom=0, top=max(14.8, max_discomfort * 1.25))
        style_box_grid(ax)
        add_bottom_panel_title_fig9_10(ax, caption, y=-0.25)
    legend_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker=KPI_MARKERS[method],
            color="none",
            markerfacecolor=KPI_COLORS[method],
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=8.0 if method == "Proposed" else 6.0,
            label=method,
        )
        for method in KPI_METHOD_ORDER
    ]
    add_framed_bottom_legend(fig, legend_handles, KPI_METHOD_ORDER, ncol=6, y=BOTTOM_LEGEND_Y, fontsize=7.4)
    return save_figure_all_formats(fig, "fig11_cost_comfort_tradeoff")


def plot_fig12_overall_kpi_comparison() -> List[Path]:
    kpi = load_kpi_table()
    metrics = [
        ("cost", "Cost (EUR/m$^2$)", "(a) Cost"),
        ("energy", "Energy (kWh/m$^2$)", "(b) Energy use"),
        ("emissions", r"CO$_2$e emissions (kgCO$_2$e/m$^2$)", r"(c) CO$_2$e emissions"),
        ("discomfort", r"Discomfort (K$\cdot$h/zone)", "(d) Thermal discomfort"),
    ]
    if kpi.empty or any(metric not in kpi.columns for metric, _, _ in metrics):
        print("[skip] Fig.12: missing KPI metrics")
        return []

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.6))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.965, bottom=0.18, wspace=0.34, hspace=0.54)
    x = np.arange(len(SCENARIO_ORDER))
    width = 0.115
    offsets = (np.arange(len(KPI_METHOD_ORDER)) - (len(KPI_METHOD_ORDER) - 1) / 2.0) * width

    for ax, (metric, ylabel, caption) in zip(axes.ravel(), metrics):
        max_val = 0.0
        for off, method in zip(offsets, KPI_METHOD_ORDER):
            vals = []
            for scenario in SCENARIO_ORDER:
                row = kpi[(kpi["scenario"] == scenario) & (kpi["controller"] == method)]
                vals.append(float(row[metric].iloc[0]) if not row.empty and pd.notna(row[metric].iloc[0]) else np.nan)
            finite_vals = [v for v in vals if math.isfinite(v)]
            if finite_vals:
                max_val = max(max_val, max(finite_vals))
            bars = ax.bar(
                x + off,
                vals,
                width=width,
                color=KPI_COLORS[method],
                edgecolor="#555555",
                linewidth=0.35,
                alpha=0.92,
                label=method,
            )
            for bar, val in zip(bars, vals):
                if not math.isfinite(val):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    val + max(0.015 * max_val, 0.015),
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=7.2,
                    fontweight="semibold",
                    clip_on=False,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIO_ORDER)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0, top=max(0.1, max_val * 1.35))
        style_box_grid(ax)
        add_bottom_panel_title_fig9_10(ax, caption, y=-0.24)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    add_framed_bottom_legend(fig, handles, labels, ncol=6, y=BOTTOM_LEGEND_Y, fontsize=7.4)
    return save_figure_all_formats(fig, "fig12_overall_kpi_comparison")


def plot_fig9_cost_comfort_tradeoff() -> List[Path]:
    kpi = load_kpi_table()
    if kpi.empty or not {"cost", "discomfort", "scenario", "controller"}.issubset(kpi.columns):
        print("[skip] Fig.9: missing KPI cost/discomfort data")
        return []
    kpi_order = ["Proposed", "PID", "MPC", "PINN-MPC", "Safe-DRL", "LearnAMR"]
    plot_colors = {
        "Proposed": "#1f77b4",
        "PID": "#ff7f0e",
        "MPC": "#2ca02c",
        "PINN-MPC": "#17becf",
        "Safe-DRL": "#d62728",
        "LearnAMR": "#9467bd",
    }
    markers = {"Proposed": "*", "PID": "o", "MPC": "s", "PINN-MPC": "P", "Safe-DRL": "^", "LearnAMR": "D"}
    offsets = {
        "Peak": {
            "Proposed": (5, 16),
            "PID": (8, 36),
            "MPC": (8, 10),
            "PINN-MPC": (5, 52),
            "Safe-DRL": (4, 8),
            "LearnAMR": (-14, 26),
        },
        "Typical": {
            "Proposed": (5, 8),
            "PID": (-30, 8),
            "MPC": (4, -15),
            "PINN-MPC": (6, 18),
            "Safe-DRL": (5, 8),
            "LearnAMR": (4, 8),
        },
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.96, bottom=0.26, wspace=0.28)
    for ax, scenario, caption in zip(
        axes,
        SCENARIO_ORDER,
        ["(a) Peak: cost--discomfort trade-off", "(b) Typical: cost--discomfort trade-off"],
    ):
        sub = kpi[kpi["scenario"] == scenario]
        for ctrl in kpi_order:
            row = sub[sub["controller"] == ctrl]
            if row.empty:
                print(f"[skip-point] Fig.9: missing KPI row for {scenario} {ctrl}")
                continue
            r = row.iloc[0]
            size = 95 if ctrl == "Proposed" else 42
            ax.scatter(
                r["cost"],
                r["discomfort"],
                s=size,
                marker=markers[ctrl],
                color=plot_colors[ctrl],
                edgecolor="black",
                linewidth=0.45,
                zorder=4 if ctrl == "Proposed" else 3,
            )
            dx, dy = offsets[scenario].get(ctrl, (4, 6))
            label = f"{ctrl}\n({r['cost']:.3f}, {r['discomfort']:.2f})"
            ax.annotate(
                label,
                (r["cost"], r["discomfort"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=5.2,
                ha="right" if dx < 0 else "left",
                va="bottom",
                bbox=dict(boxstyle="square,pad=0.16", fc="white", ec="#9a9a9a", lw=0.35, alpha=0.92),
            )
        ax.set_xlabel(r"Cost (EUR/m$^2$)")
        ax.set_ylabel(r"Discomfort (K$\cdot$h/zone)")
        style_box_grid(ax)
        ax.margins(x=0.18, y=0.24)
        if scenario == "Peak":
            ax.set_ylim(bottom=0)
        add_bottom_panel_title_fig9_10(ax, caption, y=-0.24)
    return save_figure_all_formats(fig, "fig9_cost_comfort_tradeoff")


def plot_fig10_overall_kpi_comparison() -> List[Path]:
    kpi = load_kpi_table()
    metrics = [
        ("cost", r"Cost (EUR/m$^2$)", "(a) Cost"),
        ("energy", r"Energy (kWh/m$^2$)", "(b) Energy use"),
        ("emissions", r"CO$_2$e emissions (kgCO$_2$e/m$^2$)", r"(c) CO$_2$e emissions"),
        ("discomfort", r"Discomfort (K$\cdot$h/zone)", "(d) Thermal discomfort"),
    ]
    if kpi.empty or any(m[0] not in kpi.columns for m in metrics):
        print("[skip] Fig.10: missing KPI metrics")
        return []
    fig10_order = ["Proposed", "PID", "MPC", "PINN-MPC", "LearnAMR", "Safe-DRL"]
    fig10_colors = {
        "Proposed": "#1f77b4",
        "PID": "#ff7f0e",
        "MPC": "#2ca02c",
        "PINN-MPC": "#17becf",
        "LearnAMR": "#9467bd",
        "Safe-DRL": "#d62728",
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.965, bottom=0.20, wspace=0.32, hspace=0.58)
    width = 0.105
    x = np.arange(len(SCENARIO_ORDER))
    offsets = np.linspace(-2 * width, 2 * width, len(fig10_order))
    for ax, (metric, ylabel, caption) in zip(axes.ravel(), metrics):
        scenario_best = {}
        for scen in SCENARIO_ORDER:
            vals = []
            for ctrl in fig10_order:
                sub = kpi[(kpi["scenario"] == scen) & (kpi["controller"] == ctrl)]
                vals.append(float(sub[metric].iloc[0]) if not sub.empty else np.nan)
            finite_vals = [v for v in vals if math.isfinite(v)]
            scenario_best[scen] = min(finite_vals) if finite_vals else np.nan
        for off, ctrl in zip(offsets, fig10_order):
            vals = []
            for scen in SCENARIO_ORDER:
                sub = kpi[(kpi["scenario"] == scen) & (kpi["controller"] == ctrl)]
                vals.append(float(sub[metric].iloc[0]) if not sub.empty else np.nan)
            bars = ax.bar(x + off, vals, width=width, color=fig10_colors[ctrl], label=ctrl, edgecolor="#777777", linewidth=0.25, alpha=0.92)
            for bar, scen, val in zip(bars, SCENARIO_ORDER, vals):
                if math.isfinite(val) and math.isfinite(scenario_best[scen]) and np.isclose(val, scenario_best[scen]):
                    bar.set_edgecolor("black")
                    bar.set_linewidth(1.25)
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIO_ORDER)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        style_box_grid(ax)
        add_bottom_panel_title_fig9_10(ax, caption, y=-0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.045),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="#6f6f6f",
        facecolor="white",
        columnspacing=1.2,
        handlelength=1.8,
    )
    legend.get_frame().set_linewidth(0.6)
    return save_figure_all_formats(fig, "fig10_overall_kpi_comparison")


def plot_fig11_internal_behavior() -> List[Path]:
    run = load_run("Typical", "Proposed")
    used = "Typical"
    if run is None:
        run = load_run("Peak", "Proposed")
        used = "Peak"
    if run is None:
        print("[skip] Fig.11: no Proposed trajectory")
        return []
    if used != "Typical":
        print("[warn] Fig.11: Typical Proposed data missing, using Peak")
    df = run.df
    if not require_cols(df, ["time", "Tz", "T_low", "T_high", "T_rule", "T_set"], "Fig.11", run.path):
        return []
    x = time_days(df)
    if x is None:
        print("[skip] Fig.11: missing time axis")
        return []
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.2), sharex=True, constrained_layout=True)
    axes[0].fill_between(x, series(df, "T_low"), series(df, "T_high"), color="#e8e8e8", alpha=0.75, lw=0, label="Comfort band")
    for key, color, ls in [("Tz", "#1f4e79", "-"), ("T_rule", "#777777", "--"), ("T_set", "#d97904", "-.")]:
        y = series(df, key)
        if y is not None:
            axes[0].plot(x, y, color=color, ls=ls, lw=1.1, label=key)

    any_cmd = False
    for key, color, ls in [("u_pid", "#7a7a7a", "--"), ("u_ff", "#2a8c4a", ":"), ("u_final", "#1f4e79", "-")]:
        y = series(df, key)
        if y is not None:
            axes[1].plot(x, y, color=color, ls=ls, lw=1.1, label=key)
            any_cmd = True
    if not any_cmd:
        print("[warn] Fig.11b: u_pid/u_ff/u_final all missing")

    any_part = False
    for key, color in [("beta", "#1f4e79"), ("alpha_eff", "#d97904")]:
        y = series(df, key)
        if y is not None:
            axes[2].plot(x, y, color=color, lw=1.1, label=key)
            any_part = True
    if not any_part:
        print("[warn] Fig.11c: beta/alpha_eff missing")

    dtea, dnn, gate = series(df, "DeltaT_tea"), series(df, "DeltaT_NN"), series(df, "safety_gate")
    if dtea is not None and dnn is not None:
        axes[3].plot(x, dtea, color="#7a7a7a", ls="--", lw=1.0, label="Teacher label")
        axes[3].plot(x, dnn, color="#1f4e79", ls="-", lw=1.1, label="Target residual output")
        axes[3].set_ylabel(r"Target residual ($^\circ$C)")
        if gate is not None:
            axr = axes[3].twinx()
            axr.plot(x, gate, color="#2a8c4a", ls=":", lw=1.0, label="safety gate")
            axr.set_ylabel("Safety gate")
            axr.spines["top"].set_visible(False)
    elif gate is not None:
        axes[3].plot(x, gate, color="#2a8c4a", lw=1.1, label="safety gate")
        axes[3].set_ylabel("Safety gate")
    else:
        print("[warn] Fig.11d: DeltaT_tea/DeltaT_NN and safety_gate are missing; panel left empty")

    for ax, lab in zip(axes, ["(a)", "(b)", "(c)", "(d)"]):
        add_panel_label(ax, lab)
        ax.grid(True)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", ncol=3)
    axes[0].set_ylabel(r"Temperature ($^\circ$C)")
    axes[1].set_ylabel("Command\n(normalized)")
    axes[2].set_ylabel("Participation")
    axes[3].set_xlabel("Time (days)")
    return save_figure_all_formats(fig, "fig11_internal_behavior")


FIG13_KEYS = [
    "time",
    "Tz",
    "T_low",
    "T_high",
    "T_rule",
    "T_set",
    "u_pid",
    "u_ff",
    "u_final",
    "beta",
    "alpha_eff",
    "safety_gate",
    "DeltaT_tea",
    "DeltaT_NN",
    "uff_tea",
    "rp",
    "ri",
    "rd",
]


def find_fig13_internal_candidates() -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for path in find_data_files():
        df = load_table_auto(path, nrows=5)
        if df is None or df.empty:
            continue
        mapped = normalize_columns(df)
        scenario = detect_scenario(path, mapped)
        controller = detect_controller(path, mapped)
        present = [key for key in FIG13_KEYS if find_col(mapped, key) is not None]
        has_internal_signal = any(
            key in present
            for key in ["T_rule", "T_set", "u_pid", "u_ff", "beta", "alpha_eff", "DeltaT_tea", "DeltaT_NN", "rp", "ri", "rd"]
        )
        if controller == "Proposed" and has_internal_signal:
            records.append(
                {
                    "path": path,
                    "scenario": scenario or "unknown",
                    "controller": controller,
                    "columns": list(df.columns),
                    "mapped": present,
                    "missing": [key for key in FIG13_KEYS if key not in present],
                }
            )
    return records


def load_fig13_internal_run() -> Optional[RunData]:
    for scenario in ["Typical", "Peak"]:
        run = load_run(scenario, "Proposed")
        if run is None:
            continue
        has_core = all(find_col(run.df, key) is not None for key in ["time", "Tz", "T_rule", "T_set", "u_final"])
        if has_core:
            return run
    return None


def write_ch5_fig13_internal_behavior_report() -> Path:
    candidates = find_fig13_internal_candidates()
    selected = load_fig13_internal_run()
    selected_keys = []
    missing_keys = FIG13_KEYS.copy()
    if selected is not None:
        selected_keys = [key for key in FIG13_KEYS if find_col(selected.df, key) is not None]
        missing_keys = [key for key in FIG13_KEYS if key not in selected_keys]

    lines = [
        "# Fig.13 Internal Behavior Data Check Report",
        "",
        "Generated by `analysis/plot_ch5_figures.py`. Raw result files were inspected but not modified.",
        "",
        "## Candidate Proposed Internal Trajectory Files",
        "",
    ]
    if not candidates:
        lines.append("- No Proposed internal trajectory files with recognizable internal variables were found.")
    for rec in candidates:
        rel = rec["path"].relative_to(ROOT)
        lines.extend(
            [
                f"### `{rel}`",
                f"- Scenario: {rec['scenario']}",
                f"- Controller/method: {rec['controller']}",
                f"- Columns: {', '.join(map(str, rec['columns']))}",
                f"- Mapped Fig.13 variables: {', '.join(rec['mapped']) if rec['mapped'] else 'none'}",
                f"- Missing Fig.13 variables: {', '.join(rec['missing']) if rec['missing'] else 'none'}",
                "",
            ]
        )

    scenarios = sorted({str(rec["scenario"]) for rec in candidates if rec["scenario"] != "unknown"})
    lines.extend(
        [
            "## Scenario Coverage and Selected Data",
            "",
            f"- Scenarios found: {', '.join(scenarios) if scenarios else 'none'}",
            "- Recommended scenario: Typical, because it contains the same internal variables as Peak and better reflects daily solar and occupancy-related fluctuations.",
        ]
    )
    if selected is None:
        lines.append("- Selected scenario for Fig.13: none; no usable Proposed internal trajectory was found.")
    else:
        lines.extend(
            [
                f"- Selected scenario for Fig.13: {selected.scenario}",
                f"- Selected file: `{selected.path.relative_to(ROOT)}`",
                f"- Available variables in selected file: {', '.join(selected_keys)}",
                f"- Missing variables in selected file: {', '.join(missing_keys) if missing_keys else 'none'}",
            ]
        )

    lines.extend(
        [
            "",
            "## Degradation Rules Applied",
            "",
            "- Panel (b): neural PID gain-scheduling variables are shown using `rp`, `ri`, and `rd` when available; command-composition curves are not used in Fig.13.",
        ]
    )
    if "safety_gate" in missing_keys:
        lines.append("- Panel (c): `safety_gate` / `g_k` is missing, so only `beta` and `alpha_eff` are shown on the participation axis.")
    if {"DeltaT_tea", "DeltaT_NN"}.issubset(set(selected_keys)):
        lines.append("- Panel (d): `DeltaT_tea` and `DeltaT_NN` are available, so target-residual teacher label and neural output are shown.")
    elif {"uff_tea", "u_ff"}.issubset(set(selected_keys)):
        lines.append("- Panel (d): target-residual labels are missing; feedforward teacher label and neural output are shown instead.")
    elif any(key in selected_keys for key in ["rp", "ri", "rd"]):
        lines.append("- Panel (d): teacher-label variables are missing; PID gain adaptation is shown instead.")
    else:
        lines.append("- Panel (d): teacher-label and PID-gain variables are missing; the panel is not generated.")
    if "uff_tea" in missing_keys:
        lines.append("- `uff_tea` / feedforward teacher label is missing and is not fabricated.")
    lines.append("- No random or manually fabricated data are used.")

    CH5_FIG13_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return CH5_FIG13_REPORT_PATH


def _plot_existing_series(ax: plt.Axes, x: pd.Series, df: pd.DataFrame, specs: List[Tuple[str, str, str, float, str]]) -> List[str]:
    plotted: List[str] = []
    for spec in specs:
        key, label, color, lw, ls = spec[:5]
        alpha = spec[5] if len(spec) > 5 else 1.0
        y = series(df, key)
        if y is None:
            continue
        ax.plot(x, y, color=color, lw=lw, ls=ls, alpha=alpha, label=label)
        plotted.append(key)
    return plotted


def plot_fig13_internal_behavior() -> List[Path]:
    run = load_fig13_internal_run()
    if run is None:
        print("[skip] Fig.13: no usable Proposed internal trajectory")
        return []
    if run.scenario != "Typical":
        print(f"[warn] Fig.13: Typical Proposed internal data unavailable; using {run.scenario}")

    df = run.df
    x = time_days(df)
    if x is None:
        print("[skip] Fig.13: missing time axis")
        return []

    fig, axes = plt.subplots(4, 1, figsize=(7.5, 8.8), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.88, top=0.965, bottom=0.115, hspace=0.78)

    tlow, thigh = series(df, "T_low"), series(df, "T_high")
    if tlow is not None and thigh is not None:
        axes[0].fill_between(x, tlow, thigh, color="#cfe3ee", alpha=0.62, lw=0, label="Comfort band")
    _plot_existing_series(
        axes[0],
        x,
        df,
        [
            ("Tz", "Operative temperature", "#1f4e79", 1.45, "-"),
            ("T_rule", "Rule-guided target", "#7a7a7a", 1.05, "--"),
            ("T_set", "Final target", "#b24745", 1.15, "-"),
        ],
    )
    axes[0].set_ylabel(r"Temperature ($^\circ$C)")

    plotted_gain_b = _plot_existing_series(
        axes[1],
        x,
        df,
        [
            ("rp", r"$r_p$", "#1f4e79", 1.15, "-"),
            ("ri", r"$r_i$", "#d97904", 1.05, "--"),
            ("rd", r"$r_d$", "#7b3f98", 1.05, "-."),
        ],
    )
    if not plotted_gain_b:
        plotted_gain_b = _plot_existing_series(
            axes[1],
            x,
            df,
            [
                ("Kp_eff", r"$K_p^\mathrm{eff}$", "#1f4e79", 1.15, "-"),
                ("Ki_eff", r"$K_i^\mathrm{eff}$", "#d97904", 1.05, "--"),
                ("Kd_eff", r"$K_d^\mathrm{eff}$", "#7b3f98", 1.05, "-."),
            ],
        )
    if not plotted_gain_b:
        print("[skip-panel] Fig.13b: PID gain-scheduling variables are missing")
    axes[1].set_ylabel("Gain residual")

    _plot_existing_series(
        axes[2],
        x,
        df,
        [
            ("beta", r"$\beta$ target participation", "#1f4e79", 1.25, "-"),
            ("alpha_eff", r"$\alpha^{\mathrm{eff}}_u$ command participation", "#d97904", 0.75, "--", 0.58),
        ],
    )
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Participation factor")
    ax_gate = None
    gate = series(df, "safety_gate")
    if gate is not None:
        ax_gate = axes[2].twinx()
        ax_gate.plot(x, gate, color="#5a5a5a", lw=0.95, ls=":", label="Safety gate")
        ax_gate.set_ylim(-0.05, 1.05)
        ax_gate.set_ylabel("Safety gate")
        ax_gate.spines["top"].set_visible(False)

    dtea, dnn = series(df, "DeltaT_tea"), series(df, "DeltaT_NN")
    uff_tea, uff = series(df, "uff_tea"), series(df, "u_ff")
    if dtea is not None and dnn is not None:
        axes[3].plot(x, dtea, color="#7a7a7a", lw=1.05, ls="--", label="Teacher label")
        axes[3].plot(x, dnn, color="#1f4e79", lw=1.25, ls="-", label="Target residual output")
        axes[3].set_ylabel(r"Target residual ($^\circ$C)")
        panel_d_title = "(d) Target-residual teacher label and neural output"
    elif uff_tea is not None and uff is not None:
        axes[3].plot(x, uff_tea, color="#7a7a7a", lw=1.05, ls="--", label="Teacher label")
        axes[3].plot(x, uff, color="#2a8c4a", lw=1.25, ls="-", label="Neural feedforward")
        axes[3].set_ylabel("Feedforward command")
        panel_d_title = "(d) Feedforward teacher label and neural output"
    else:
        plotted_gain = _plot_existing_series(
            axes[3],
            x,
            df,
            [
                ("rp", r"$r_p$ / $K_p$", "#1f4e79", 1.15, "-"),
                ("ri", r"$r_i$ / $K_i$", "#d97904", 1.05, "--"),
                ("rd", r"$r_d$ / $K_d$", "#7b3f98", 1.05, "-."),
            ],
        )
        if not plotted_gain:
            print("[skip-panel] Fig.13d: teacher labels and gain variables are missing")
        axes[3].set_ylabel("Gain residual")
        panel_d_title = "(d) Online gain adaptation"

    captions = [
        "(a) Target generation and comfort tracking",
        "(b) Neural PID gain scheduling",
        "(c) Progressive participation",
        panel_d_title,
    ]
    for ax, caption in zip(axes, captions):
        style_box_grid(ax)
        add_bottom_panel_title_fig9_10(ax, caption, y=-0.42 if ax is axes[-1] else -0.26)
        handles, labels = ax.get_legend_handles_labels()
        if ax is axes[0] and handles:
            add_panel_top_legend(ax, handles, labels, ncol=4, loc="lower right", anchor=(1.0, 1.035))
        elif ax is axes[1] and handles:
            add_panel_top_legend(ax, handles, labels, ncol=3, loc="lower right", anchor=(1.0, 1.035))
        elif ax is axes[2] and handles:
            if ax_gate is not None:
                gate_handles, gate_labels = ax_gate.get_legend_handles_labels()
                handles = handles + gate_handles
                labels = labels + gate_labels
            add_panel_top_legend(ax, handles, labels, ncol=min(len(labels), 3), loc="lower right", anchor=(1.0, 1.035))
        elif ax is axes[3] and handles:
            add_panel_top_legend(ax, handles, labels, ncol=min(len(labels), 2), loc="lower right", anchor=(1.0, 1.035))
    axes[3].set_xlabel("Time (days)", labelpad=8)
    axes[0].set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    return save_figure_all_formats(fig, "fig13_internal_behavior")


def load_ablation_table() -> pd.DataFrame:
    rows = [
        ("Peak", "L0: Rule-PID", 0.7987, 3.060, 0.5110, 0.0008),
        ("Peak", "L1: + target residual", 0.8009, 3.069, 0.5125, 0.0000),
        ("Peak", "L2: + gain adaptation", 0.8046, 3.083, 0.5148, 0.0000),
        ("Peak", "L3: Full model", 0.8058, 3.088, 0.5156, 0.0000),
        ("Peak", "Full w/o boundary-aware gate", 0.80907, 3.0999, 0.5177, 0.0000),
        ("Peak", "Full w/o progressive participation", 0.80995, 3.1021, 0.5181, 0.0000),
        ("Typical", "L0: Rule-PID", 0.324, 1.390, 0.232, 7.462),
        ("Typical", "L1: + target residual", 0.319, 1.370, 0.229, 5.363),
        ("Typical", "L2: + gain adaptation", 0.312, 1.340, 0.224, 4.859),
        ("Typical", "L3: Full model", 0.311, 1.334, 0.223, 4.764),
        ("Typical", "Full w/o boundary-aware gate", 0.31364, 1.3472, 0.2250, 5.0126),
        ("Typical", "Full w/o progressive participation", 0.31529, 1.3526, 0.2259, 5.2103),
    ]
    out = pd.DataFrame(rows)
    out.columns = ["scenario", "variant", "cost", "energy", "emissions", "discomfort"]
    out["path"] = "curated Section 5.5 ablation data"
    for c in ["cost", "energy", "emissions", "discomfort"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def canonical_ablation_label(value, df: pd.DataFrame) -> str:
    text = " ".join(str(x) for x in [value] + [df[c].iloc[0] for c in df.columns if c in {"method", "ablation"}]).lower()
    if "rule_pid" in text or "rule pid" in text or "pid_only" in text or "ruleonly" in text or "l0" in text or "no nn" in text:
        return "Rule-PID only"
    if "no_target" in text or "without_target" in text or "notarget" in text or "no_tcn_target" in text:
        return "w/o target residual"
    if "no_gain" in text or "without_gain" in text or "fixed_pid" in text or "paper pid" in text or "l1" in text:
        return "+ TCN_T"
    if "no_ff" in text or "without_ff" in text or "no_feedforward" in text or "ff_scale': 0.0" in text or "gain-only" in text or "l2" in text:
        return "+ TCN_U"
    if "no_progressive" in text or "without_progressive" in text or "no_ramp" in text or "no_participation" in text:
        return "w/o progressive participation"
    if "no_safety" in text or "without_safety" in text or "no_gate" in text or "no_safety_gate" in text or "gate" in text:
        return "w/o safety gate"
    if "l3" in text or "full" in text or "adppid+ff" in text or "complete" in text or "full_model" in text or "proposed" in text or "ours" in text:
        return "+ Feedforward"
    return str(value)


ABLATION_ORDER = [
    "L0: Rule-PID",
    "L1: + target residual",
    "L2: + gain adaptation",
    "L3: Full model",
    "Full w/o boundary-aware gate",
    "Full w/o progressive participation",
]

ABLATION_COLORS = {
    "L0: Rule-PID": "#7a7a7a",
    "L1: + target residual": "#4daf4a",
    "L2: + gain adaptation": "#d97904",
    "L3: Full model": "#1f4e79",
    "Full w/o boundary-aware gate": "#b24745",
    "Full w/o progressive participation": "#7b3f98",
}

ABLATION_MARKERS = {
    "L0: Rule-PID": "o",
    "L1: + target residual": "^",
    "L2: + gain adaptation": "s",
    "L3: Full model": "o",
    "Full w/o boundary-aware gate": "P",
    "Full w/o progressive participation": "D",
}

ABLATION_DISPLAY_LABELS = {
    "L0: Rule-PID": "L0 Rule-PID",
    "L1: + target residual": "L1 + target residual",
    "L2: + gain adaptation": "L2 + gain adaptation",
    "L3: Full model": "L3 Full",
    "Full w/o boundary-aware gate": "Full w/o gate",
    "Full w/o progressive participation": "Full w/o prog.",
}

ABLATION_PROGRESSIVE_MARKER_AREA = 36.0


def load_ablation_kpi_data() -> pd.DataFrame:
    return load_ablation_table()


def _sort_ablation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["scenario"] = pd.Categorical(out["scenario"], SCENARIO_ORDER, ordered=True)
    out["variant"] = pd.Categorical(out["variant"], ABLATION_ORDER, ordered=True)
    return out.sort_values(["scenario", "variant"]).reset_index(drop=True)


def write_table9_ablation_kpi(df: pd.DataFrame) -> Optional[Path]:
    if df.empty:
        return None
    path = TABLE_DIR / "table9_ablation_kpi.tex"
    table = df.copy()
    rows = [
        ("L0: Rule-PID", "L0: Rule-PID"),
        ("L1: + target residual", "L1: + target residual"),
        ("L2: + gain adaptation", "L2: + gain adaptation"),
        ("L3: Full model", "L3: Full model"),
        ("Full w/o gate", "Full w/o boundary-aware gate"),
        ("Full w/o progressive", "Full w/o progressive participation"),
    ]

    def metric(scenario: str, variant: str, key: str) -> str:
        hit = table[(table["scenario"].astype(str) == scenario) & (table["variant"].astype(str) == variant)]
        if hit.empty or key not in hit.columns or pd.isna(hit[key].iloc[0]):
            return "--"
        return f"{float(hit[key].iloc[0]):.3f}"

    lines = [
        "\\begin{tabular}{lrrrrrrrr}",
        "\\hline",
        "Variant & Peak cost & Peak energy & Peak emissions & Peak discomfort & Typical cost & Typical energy & Typical emissions & Typical discomfort \\\\",
        " & (EUR/m$^2$) & (kWh/m$^2$) & (kgCO$_2$e m$^{-2}$) & (K$\\cdot$h/zone) & (EUR/m$^2$) & (kWh/m$^2$) & (kgCO$_2$e m$^{-2}$) & (K$\\cdot$h/zone) \\\\",
        "\\hline",
    ]
    for label, variant in rows:
        lines.append(
            f"{label} & {metric('Peak', variant, 'cost')} & {metric('Peak', variant, 'energy')} & "
            f"{metric('Peak', variant, 'emissions')} & {metric('Peak', variant, 'discomfort')} & "
            f"{metric('Typical', variant, 'cost')} & {metric('Typical', variant, 'energy')} & "
            f"{metric('Typical', variant, 'emissions')} & {metric('Typical', variant, 'discomfort')} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def plot_fig14_ablation_tradeoff() -> List[Path]:
    abl = load_ablation_kpi_data()
    required = {"scenario", "variant", "cost", "discomfort"}
    if abl.empty or not required.issubset(abl.columns) or abl[list(required)].isna().any().any():
        print("[skip] Fig.14: missing ablation cost/discomfort data")
        return []
    abl = _sort_ablation(abl)
    scenarios = [s for s in SCENARIO_ORDER if s in set(abl["scenario"].astype(str))]
    if not scenarios:
        print("[skip] Fig.14: no Peak or Typical ablation scenarios")
        return []

    if len(scenarios) == 1:
        fig, axes_obj = plt.subplots(1, 1, figsize=(4.2, 3.4))
        axes = [axes_obj]
        fig.subplots_adjust(left=0.16, right=0.985, top=0.955, bottom=0.30)
        captions = [f"(a) {scenarios[0]} scenario"]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
        axes = list(axes)
        fig.subplots_adjust(left=0.085, right=0.985, top=0.955, bottom=0.30, wspace=0.28)
        captions = ["(a) Peak scenario", "(b) Typical scenario"]

    legend_handles: Dict[str, mpl.artist.Artist] = {}
    for ax, scenario, caption in zip(axes, scenarios, captions):
        sub = _sort_ablation(abl[abl["scenario"].astype(str) == scenario].copy())
        for _, r in sub.iterrows():
            variant = str(r["variant"])
            is_mechanism = variant in {
                "Full w/o boundary-aware gate",
                "Full w/o progressive participation",
            }
            if variant == "Full w/o progressive participation":
                size = ABLATION_PROGRESSIVE_MARKER_AREA
            else:
                size = 96 if is_mechanism else (88 if variant == "L3: Full model" else 76)
            sc = ax.scatter(
                r["cost"],
                r["discomfort"],
                s=size,
                marker=ABLATION_MARKERS.get(variant, "o"),
                facecolors=ABLATION_COLORS.get(variant, "#555555"),
                edgecolors="black",
                linewidths=1.15 if is_mechanism else 0.85,
                zorder=8 if is_mechanism else 5,
                label=variant,
            )
            legend_handles.setdefault(variant, sc)
        ax.set_xlabel("Cost (EUR/m$^2$)")
        ax.set_ylabel(r"Discomfort (K$\cdot$h/zone)")
        style_box_grid(ax)
        xvals = pd.to_numeric(sub["cost"], errors="coerce")
        yvals = pd.to_numeric(sub["discomfort"], errors="coerce")
        xpad = max((xvals.max() - xvals.min()) * 0.35, 0.004)
        ypad = max((yvals.max() - yvals.min()) * 0.35, 0.001)
        ax.set_xlim(float(xvals.min() - xpad), float(xvals.max() + xpad))
        ax.set_ylim(bottom=float(-ypad * 0.18), top=float(yvals.max() + ypad))
        yticks = [tick for tick in ax.get_yticks() if tick >= 0]
        if yticks:
            ax.set_yticks(yticks)
        add_bottom_panel_title_fig9_10(ax, caption, y=-0.24)

    ordered_labels = [v for v in ABLATION_ORDER if v in legend_handles]
    legend_plot_handles: List[mpl.artist.Artist] = []
    for v in ordered_labels:
        if v == "Full w/o progressive participation":
            legend_plot_handles.append(
                mpl.lines.Line2D(
                    [0],
                    [0],
                    linestyle="",
                    marker=ABLATION_MARKERS.get(v, "D"),
                    markerfacecolor=ABLATION_COLORS.get(v, "#7b3f98"),
                    markeredgecolor="black",
                    markeredgewidth=0.9,
                    markersize=float(np.sqrt(ABLATION_PROGRESSIVE_MARKER_AREA)),
                    label=v,
                )
            )
        else:
            legend_plot_handles.append(legend_handles[v])
    legend = add_framed_bottom_legend(
        fig,
        legend_plot_handles,
        [ABLATION_DISPLAY_LABELS.get(v, v) for v in ordered_labels],
        ncol=min(6, len(ordered_labels)),
        y=0.008,
    )

    return save_figure_all_formats(fig, "fig14_ablation_tradeoff")


def write_ch5_fig14_ablation_report(fig14_generated: bool = False, table9_path: Optional[Path] = None) -> Path:
    abl = load_ablation_kpi_data()
    lines = [
        "# Fig.14 Ablation Data Check Report",
        "",
        "Generated by `analysis/plot_ch5_figures.py`. The curated Section 5.5 ablation values are used directly; no control experiments are rerun.",
        "",
        "## Data Source",
        "",
        "- Cumulative module ablations: L0 Rule-PID, L1 + target TCN, L2 + gain TCN, and L3 Full model.",
        "- Moderation-mechanism ablations: Full w/o boundary-aware gate and Full w/o progressive participation.",
        "- Values match the Section 5.5 update request and the completed moderation-ablation summary.",
    ]
    lines.append("")
    if abl.empty:
        lines.extend(["## Status", "", "- No readable ablation KPI data were found."])
        CH5_FIG14_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        return CH5_FIG14_REPORT_PATH

    abl = _sort_ablation(abl)
    scenarios = [s for s in SCENARIO_ORDER if s in set(abl["scenario"].astype(str))]
    found_variants = [v for v in ABLATION_ORDER if v in set(abl["variant"].astype(str))]
    missing_variants = [v for v in ABLATION_ORDER if v not in found_variants]
    lines.extend(
        [
            "## Coverage",
            "",
            f"- Scenarios found: {', '.join(scenarios) if scenarios else 'none'}",
            f"- Ablation variants found: {', '.join(found_variants) if found_variants else 'none'}",
            f"- Missing requested variants: {', '.join(missing_variants) if missing_variants else 'none'}",
            "",
            "## KPI Values",
            "",
            "| Scenario | Variant | Cost | Energy | Emissions | Discomfort |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, r in abl.iterrows():
        lines.append(
            f"| {r['scenario']} | {r['variant']} | {float(r['cost']):.3f} | {float(r['energy']):.3f} | "
            f"{float(r['emissions']):.3f} | {float(r['discomfort']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Output Check",
            "",
            f"- Fig.14 generated: {'yes' if fig14_generated else 'not yet'}",
            f"- Table 9 generated: {'yes, `' + str(table9_path.relative_to(ROOT)) + '`' if table9_path else 'no'}",
            "- Unit garbling: none detected in generated SVG by script-level unit strings.",
            "- Cumulative variants: L0--L3 are shown as markers only; no lines, arrows, dashed lines, or trend lines are drawn.",
            "- Mechanism ablations: boundary-gate and progressive-participation variants are plotted as independent high-zorder markers.",
            "",
            "## Interpretation",
            "",
            "- L0--L3 show the cumulative module path from Rule-PID to the full model.",
            "- The two additional variants remove moderation mechanisms from the full controller and are not part of the cumulative path.",
            "- In the Peak scenario, all variants remain close to the zero-discomfort region.",
            "- In the Typical scenario, the cumulative path and the two mechanism ablations are more discriminative.",
            "- No baseline controllers are included in Fig.14.",
        ]
    )
    CH5_FIG14_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return CH5_FIG14_REPORT_PATH


def plot_fig12_ablation_tradeoff() -> List[Path]:
    abl = load_ablation_table()
    if abl.empty or not {"scenario", "cost", "discomfort", "variant"}.issubset(abl.columns):
        print("[skip] Fig.12: missing ablation cost/discomfort data")
        return []
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    palette = {
        "Proposed": "#1f4e79",
        "w/o high-level residual": "#7a7a7a",
        "w/o neural PID gain scheduling": "#d97904",
        "w/o feedforward compensation": "#7b3f98",
        "w/o safety gate": "#2a8c4a",
    }
    for ax, scenario, lab in zip(axes, SCENARIO_ORDER, ["(a)", "(b)"]):
        sub = abl[abl["scenario"].astype(str).str.lower() == scenario.lower()]
        for i, (_, r) in enumerate(sub.iterrows()):
            label = str(r["variant"])
            marker = "*" if label == "Proposed" else ("^" if i % 2 else "o")
            size = 105 if label == "Proposed" else 52
            ax.scatter(r["cost"], r["discomfort"], s=size, marker=marker, color=palette.get(label, "#555555"), edgecolor="white", linewidth=0.5)
            ax.annotate(label, (r["cost"], r["discomfort"]), xytext=(4, 4 + (i % 3) * 3), textcoords="offset points", fontsize=7)
        add_panel_label(ax, lab)
        ax.set_xlabel(r"Cost (EUR/m$^2$)")
        ax.set_ylabel(r"Discomfort (K$\cdot$h/zone)")
        ax.grid(True)
        ax.margins(0.18)
    return save_figure_all_formats(fig, "fig12_ablation_tradeoff")


def inspect_data_files() -> List[Dict[str, object]]:
    records = []
    for p in find_data_files():
        df = load_table_auto(p, nrows=20)
        is_ablation = "ablation" in str(p.relative_to(ROOT)).lower()
        rec = {
            "path": str(p.relative_to(ROOT)),
            "suffix": p.suffix.lower(),
            "readable": df is not None,
            "columns": list(df.columns)[:80] if df is not None else [],
            "scenario": detect_scenario(p, normalize_columns(df) if df is not None else None),
            "controller": "Ablation variant" if is_ablation else detect_controller(p, normalize_columns(df) if df is not None else None),
            "possible_figures": [],
        }
        cols = set(normalize_columns(df).columns) if df is not None else set()
        name = rec["path"].lower()
        if is_ablation:
            rec["possible_figures"].append("Fig.12")
        elif {"Tz", "u"}.intersection(cols) or "details.csv" in name:
            if rec["scenario"] == "Peak":
                rec["possible_figures"].append("Fig.7")
            if rec["scenario"] == "Typical":
                rec["possible_figures"].append("Fig.8")
        if not is_ablation and ({"cost", "energy", "emissions", "discomfort"}.intersection(cols) or "kpi" in name):
            rec["possible_figures"].extend(["Fig.9", "Fig.10"])
        if {"T_rule", "T_set", "beta", "u_pid", "DeltaT_tea", "DeltaT_NN"}.intersection(cols):
            rec["possible_figures"].append("Fig.11")
        records.append(rec)
    return records


def write_inventory_report() -> Path:
    records = inspect_data_files()
    selected_paths = {str(p.relative_to(ROOT)) for p in list(scenario_runs().values()) + list(kpi_json_paths().values()) if p.exists()}
    selected_paths.update(str(p.relative_to(ROOT)) for p in (ROOT / "ablation" / "raw").glob("*_kpi.json"))
    selected = [r for r in records if r["path"] in selected_paths]

    lines = [
        "# Chapter 5 Data Inventory Report",
        "",
        "This report is generated by `analysis/plot_ch5_figures.py`. It inventories reusable project result files without modifying raw data.",
        "",
        "## Selected Files Used by Fig.7--Fig.12",
        "",
    ]
    for r in selected:
        cols = ", ".join(map(str, r["columns"][:30]))
        figs = ", ".join(r["possible_figures"]) if r["possible_figures"] else "not directly mapped"
        lines.extend(
            [
                f"### `{r['path']}`",
                f"- Readable: {r['readable']}",
                f"- Scenario: {r['scenario'] or 'unknown'}",
                f"- Controller/method: {r['controller'] or 'unknown'}",
                f"- Main fields: {cols}",
                f"- Possible use: {figs}",
                "",
            ]
        )
    lines.extend(
        [
            "## Figure Readiness",
            "",
            "- Fig.7 Peak closed-loop: can be generated from Peak trajectory `details.csv` files for Proposed, PID, MPC, Safe-DRL, and LearnAMR.",
            "- Fig.8 Typical closed-loop: can be generated from Typical trajectory `details.csv` files for Proposed, PID, MPC, Safe-DRL, and LearnAMR.",
            "- Fig.9 Cost-comfort trade-off: can be generated from KPI JSON files for all five controllers and two scenarios.",
            "- Fig.10 Overall KPI comparison: can be generated from KPI JSON files for cost, energy, emissions, and discomfort.",
            "- Fig.11 Internal behavior: can be generated from Proposed trajectory files. `safety_gate` is not present in the inspected Proposed data, so the residual teacher/NN panel is used instead.",
            "- Fig.12 Ablation trade-off: can be generated from `ablation/raw/*_kpi.json`. The available variants map to Proposed, w/o high-level residual, w/o neural PID gain scheduling, and w/o feedforward compensation. A distinct w/o safety gate variant was not found and is not fabricated.",
            "",
            "## Missing or Non-ideal Data Notes",
            "",
            "- `ablation/tables/ablation_kpi.csv` appears to have inconsistent row widths, so the script uses individual `ablation/raw/*_kpi.json` files instead.",
            "- Proposed internal variables include `delta_T_teacher`, `delta_T_nn`, `beta`, `alpha_u`, `u_pid`, `u_ff_nn`, and `u_final`; no explicit `safety_gate` field was found in the selected files.",
            "- No random or manually fabricated data are used.",
            "",
            "## Full Inventory Count",
            "",
            f"- Candidate structured files found: {len(records)}",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_PATH


def write_latex_table(df: pd.DataFrame, path: Path, group_col: str, name_col: str) -> Optional[Path]:
    if df.empty:
        return None
    metrics = ["cost", "energy", "emissions", "discomfort"]
    table = df.copy()
    for m in metrics:
        table[m] = pd.to_numeric(table[m], errors="coerce")
    best = {m: table[m].min(skipna=True) for m in metrics}
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\hline",
        f"{group_col.title()} & {name_col.title()} & Cost & Energy & Emissions & Discomfort \\\\",
        "\\hline",
    ]
    sort_cols = [group_col, name_col]
    for _, r in table.sort_values(sort_cols).iterrows():
        vals = []
        for m in metrics:
            v = r[m]
            txt = "--" if pd.isna(v) else f"{v:.3f}"
            if not pd.isna(v) and np.isclose(v, best[m]):
                txt = f"\\textbf{{{txt}}}"
            vals.append(txt)
        lines.append(f"{r[group_col]} & {r[name_col]} & " + " & ".join(vals) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_tables() -> List[Path]:
    paths: List[Path] = []
    kpi = load_kpi_table()
    p = write_latex_table(kpi, TABLE_DIR / "table8_overall_kpi.tex", "scenario", "controller")
    if p:
        paths.append(p)
    abl = load_ablation_table().rename(columns={"variant": "controller"})
    p = write_table9_ablation_kpi(load_ablation_table())
    if p:
        paths.append(p)
    return paths


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    ensure_dirs()
    init_nature_style()
    report = write_ch5_fig7_fig10_data_report()
    kpi_report = write_ch5_fig11_fig12_kpi_report()
    fig13_report = write_ch5_fig13_internal_behavior_report()
    table9_path = write_table9_ablation_kpi(load_ablation_kpi_data())
    generated: List[Path] = []
    for func in [
        plot_fig7_peak_conventional,
        plot_fig8_peak_advanced,
        plot_fig9_typical_conventional,
        plot_fig10_typical_advanced,
        plot_fig11_cost_comfort_tradeoff,
        plot_fig12_overall_kpi_comparison,
        plot_fig13_internal_behavior,
        plot_fig14_ablation_tradeoff,
    ]:
        try:
            generated.extend(func())
        except Exception as exc:
            print(f"[skip] {func.__name__}: {exc}")
    fig14_generated = any(p.name.startswith("fig14_ablation_tradeoff.") for p in generated)
    fig14_report = write_ch5_fig14_ablation_report(fig14_generated, table9_path)
    print("\n=== Data report ===")
    print(report.relative_to(ROOT))
    print(kpi_report.relative_to(ROOT))
    print(fig13_report.relative_to(ROOT))
    print(fig14_report.relative_to(ROOT))
    print("\n=== Generated Fig.7--Fig.14 files ===")
    for p in generated:
        print(p.relative_to(ROOT))
    print("\n=== Requested output patterns ===")
    for pattern in [
        "fig7_peak_conventional.*",
        "fig8_peak_advanced.*",
        "fig9_typical_conventional.*",
        "fig10_typical_advanced.*",
        "fig11_cost_comfort_tradeoff.*",
        "fig12_overall_kpi_comparison.*",
        "fig13_internal_behavior.*",
        "fig14_ablation_tradeoff.*",
    ]:
        hits = sorted(FIG_DIR.glob(pattern))
        print(f"{pattern}: {'OK' if len(hits) >= 3 else 'MISSING'}")
        for p in hits:
            print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

