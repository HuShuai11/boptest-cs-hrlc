# BOPTEST Cold-Start Hierarchical Residual Learning Control

## Overview

This repository provides the implementation of **CS-HRLC**, a cold-start hierarchical online supervised residual learning controller for hydronic heat-pump heating under dynamic electricity pricing. The controller is evaluated on the Building Optimization Performance Test (BOPTEST) `bestest_hydronic_heat_pump` benchmark.

The controller retains an interpretable rule/PID feedback backbone as the deployable control path and introduces **bounded online residual corrections** around this backbone. High-level residual learning (TCN<sub>T</sub>) refines the target temperature, while low-level residual learning (TCN<sub>U</sub>) provides PID-gain scheduling and a bounded feedforward correction. Progressive participation schedules and comfort-boundary-aware direction modulation limit the authority of learned corrections during early deployment and near comfort limits. An auxiliary one-step RLS thermal predictor is used only for teacher-label construction and is not part of the forward control path.

**Key design principle**: The online control loop does not embed model predictive control (MPC) or reinforcement learning (RL). The controller does not rely on building-specific historical operating trajectories, offline policy pretraining, receding-horizon optimization, or reward-driven exploration.

## Associated Manuscript

> BOPTEST-based evaluation of cold-start online control for hydronic heat-pump heating under dynamic electricity pricing

*Manuscript under review.*

## Benchmark

| Item | Setting |
|------|---------|
| Platform | Building Optimization Performance Test (BOPTEST) |
| Test case | `bestest_hydronic_heat_pump` |
| BOPTEST version | 0.9.0 |
| Electricity tariff | `highly_dynamic` |
| Control interval | 900 s (15 minutes) |
| Warm-up period | 7 days (emulator thermal-state initialization only) |
| Evaluation horizon | 14 days of continuous deployment |
| Peak scenario | Start day-of-year 16 |
| Typical scenario | Start day-of-year 108 |

## Repository Structure

```
boptest-cs-hrlc/
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
├── requirements.txt
├── environment.yml
├── common/
│   │   └── boptest_client.py    # BOPTEST REST API client
│   ├── method/
│   │   ├── rule_teacher.py      # Rule-guided target, comfort bounds, teacher labels
│   │   ├── networks.py          # PureTCN, TCNU, replay buffers
│   │   ├── controllers.py       # PaperPID, AdaptivePID
│   │   ├── rls.py               # RLS one-step thermal predictor
│   │   ├── tcnu_teacher.py      # TCN_U features, gain labels, feedforward teacher
│   │   ├── scheduler.py         # seed, beta_sched, alpha_u_sched
│   │   └── action.py            # Normalized command → BOPTEST actuator mapping
├── config/
│   └── default.yaml           # All controller parameters (single source of truth)
├── scripts/
│   ├── main.py                # Main entry point (Proposed controller)
│   ├── plot_ch5_figures.py    # Plotting script for all result figures
│   └── ablation/
│       ├── L0_baseline.py     # T_rule + PID backbone (no neural)
│       ├── L1_tcn_t.py        # + TCN_T target residual correction
│       ├── L2_tcn_u_gain.py   # + TCN_U gain scheduling
│       └── L3_tcn_u_full.py   # + TCN_U full (gain + feedforward)
├── results/
│   └── example/               # Example KPI outputs
├── figures/                   # Generated figures (created at runtime)
└── docs/
```

## Installation

### Prerequisites

- Python 3.10+
- BOPTEST Docker container running locally
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/HuShuai11/boptest-cs-hrlc.git
cd boptest-cs-hrlc

# Option A: conda
conda env create -f environment.yml
conda activate boptest-cs-hrlc

# Option B: pip
pip install -r requirements.txt
```

## BOPTEST Setup

BOPTEST is an **external dependency**. Follow the official instructions to set it up:

```bash
# Clone BOPTEST
git clone https://github.com/ibpsa/project1-boptest.git
cd project1-boptest

# Start the Docker container
docker compose up -d
```

The BOPTEST REST API will be available at `http://127.0.0.1:5000`. The controller connects to this URL by default (configurable in `common/boptest_client.py`).

## Running Experiments

The main entry point is `scripts/main.py`, which runs the proposed (full) controller:

```bash
cd boptest-cs-hrlc

# Run Topical scenario
python scripts/main.py --scenario Typical

# Run Peak scenario
python scripts/main.py --scenario Peak

# Use a custom config
python scripts/main.py --config config/default.yaml
```

### Ablation Experiments

```bash
# L0: Rule + PID backbone (no neural networks)
python scripts/ablation/L0_baseline.py --scenario Typical

# L1: + TCN_T target correction
python scripts/ablation/L1_tcn_t.py --scenario Typical

# L2: + TCN_U gain scheduling
python scripts/ablation/L2_tcn_u_gain.py --scenario Typical

# L3: + TCN_U full (gain + feedforward) — same as proposed
python scripts/ablation/L3_tcn_u_full.py --scenario Typical
```

Replace `Typical` with `Peak` for the Peak scenario.

## Cold-Start Data Boundary

The controller is evaluated under a strict cold-start protocol:

- The 7-day BOPTEST warm-up period is used **only** for emulator thermal-state initialization.
- Warm-up samples are **not** used for: TCN<sub>T</sub> training, TCN<sub>U</sub> training, RLS identification, teacher-label generation, replay-buffer initialization, parameter tuning, or KPI reporting.
- At the first formal control step, replay buffers are empty and neural modules have no building-specific training.
- The rule-guided target generator and PID feedback backbone provide the initial deployable control behavior.

## Outputs

- **KPI JSON**: `results/` directory (created at runtime)
- **Figures**: `figures/` directory (created at runtime by `plot_ch5_figures.py`)
- **Trajectory CSV**: Detailed per-step logs under `results/trajectories/`

## Key Results

In the BOPTEST `bestest_hydronic_heat_pump` benchmark under the evaluated protocol:

| Scenario | Metric | Proposed (L3) |
|----------|--------|:------------:|
| Peak | Cost (EUR/m²) | 0.806 |
| Peak | Discomfort (K·h/zone) | 0.000 |
| Typical | Cost (EUR/m²) | 0.311 |
| Typical | Discomfort (K·h/zone) | 4.764 |

Cumulative ablation from Rule-PID backbone to full model reduces Typical discomfort by 36.2% and operating cost by approximately 4.0%.

## Reproducibility

| Item | Value |
|------|-------|
| Random seed | 123 |
| Config file | `config/default.yaml` |
| BOPTEST version | 0.9.0 |
| Python | 3.10+ |
| Control interval | 900 s |

## Citation

If you use this code, please cite the associated manuscript and this repository:

```bibtex
@software{boptest_cs_hrlc,
  title = {BOPTEST Cold-Start Hierarchical Residual Learning Control},
  author = {Ma, Le and Hu, Shuai and Zhao, Linhan and Yao, Zhenjie and Du, Guanfeng and Jin, Xu},
  year = {2026},
  url = {https://github.com/HuShuai11/boptest-cs-hrlc},
}
```

See also [CITATION.cff](CITATION.cff).

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Third-Party Software

| Software | License | Source |
|----------|---------|--------|
| BOPTEST | BSD-3-Clause | https://github.com/ibpsa/project1-boptest |
| PyTorch | BSD | https://pytorch.org |
| NumPy | BSD | https://numpy.org |

BOPTEST is an external dependency and is not included in this repository. All baseline controllers (PI, MPC, PINN-MPC, Safe-DRL, LearnAMR) are external to this repository and were configured following their original implementations.

## Acknowledgments

The authors acknowledge the IBPSA BOPTEST project for providing the standardized benchmark environment.
