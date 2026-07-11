# Cold-Start Hierarchical Residual Learning Control for Heat-Pump Heating in BOPTEST

This repository provides the source code, evaluation scripts and configuration files for:

**BOPTEST-based evaluation of cold-start online control for hydronic heat-pump heating under dynamic electricity pricing**, submitted to *Building Simulation*.

The proposed controller (CS-HRLC) is designed for safe online building heating control under dynamic electricity pricing, with a focus on cold-start deployment without historical operating data.

## Overview

This project implements a hierarchical online supervised residual learning controller (CS-HRLC) for heat-pump-driven building heating systems evaluated on the `bestest_hydronic_heat_pump` test case from the BOPTEST platform under a highly dynamic electricity price scenario. The controller combines:

- a rule-guided target generator and PID feedback backbone as the deployable cold-start control path;
- bounded online residual corrections for target-temperature planning and actuator-command refinement;
- progressive participation and comfort-boundary-aware modulation to limit the authority of learned corrections during early deployment and near comfort limits;
- an auxiliary one-step RLS thermal predictor used only for teacher-label construction.

The online control loop does not embed model predictive control (MPC) or reinforcement learning (RL). The controller does not rely on building-specific historical operating trajectories, offline policy pretraining, receding-horizon optimization, or reward-driven exploration.

## Repository Structure

```text
.
├── main.py                          # Main entry point (Proposed controller)
├── config/
│   └── default.yaml                 # Controller parameters (single source of truth)
├── common/
│   └── boptest_client.py            # BOPTEST REST API client
├── method/
│   ├── rule_teacher.py              # Rule-guided target, comfort bounds, teacher labels
│   ├── networks.py                  # PureTCN, TCNU, replay buffers
│   ├── controllers.py               # PaperPID, AdaptivePID
│   ├── rls.py                       # RLS one-step thermal predictor
│   ├── tcnu_teacher.py              # TCN_U features, gain labels, feedforward teacher
│   ├── scheduler.py                 # seed, beta_sched, alpha_u_sched
│   └── action.py                    # Normalized command → BOPTEST actuator mapping
├── ablation/
│   ├── L0_baseline.py               # T_rule + PID backbone (no neural)
│   ├── L1_tcn_t.py                  # + TCN_T target residual correction
│   ├── L2_tcn_u_gain.py             # + TCN_U gain scheduling
│   └── L3_tcn_u_full.py             # + TCN_U full (gain + feedforward)
├── plot_ch5_figures.py              # Plotting script for all result figures
├── results/
│   └── example/                     # Example KPI outputs
├── requirements.txt
├── environment.yml
├── LICENSE
├── CITATION.cff
└── README.md
```

## Installation

**Prerequisites**: Python 3.10+, BOPTEST Docker container running locally.

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

BOPTEST is an **external dependency**. Follow the [official instructions](https://github.com/ibpsa/project1-boptest) to set it up. The BOPTEST REST API will be available at `http://127.0.0.1:5000`.

## Running Experiments

```bash
python main.py                          # run Typical + Peak
python main.py --scenario Typical       # Typical only
python main.py --scenario Peak          # Peak only
python main.py --config config/default.yaml
```

### Ablation

```bash
python ablation/L0_baseline.py --scenario Typical
python ablation/L1_tcn_t.py --scenario Typical
python ablation/L2_tcn_u_gain.py --scenario Typical
python ablation/L3_tcn_u_full.py --scenario Typical
```

## Cold-Start Data Boundary

The 7-day BOPTEST warm-up period is used **only** for emulator thermal-state initialization. Warm-up samples are **not** used for: TCN<sub>T</sub> training, TCN<sub>U</sub> training, RLS identification, teacher-label generation, replay-buffer initialization, parameter tuning, or KPI reporting.

## Key Results

| Scenario | Cost (EUR/m²) | Energy (kWh/m²) | Emissions (kgCO₂e/m²) | Discomfort (K·h/zone) |
|----------|:------------:|:---------------:|:---------------------:|:---------------------:|
| Peak     | 0.806        | 3.088           | 0.516                 | 0.000                 |
| Typical  | 0.311        | 1.334           | 0.223                 | 4.764                 |

Cumulative ablation from the Rule-PID backbone to the full model reduces Typical discomfort by 36.2% and operating cost by approximately 4.0%.

## Citation

If you use this code, please cite the associated manuscript. See [CITATION.cff](CITATION.cff).

```bibtex
@software{boptest_cs_hrlc,
  title = {BOPTEST Cold-Start Hierarchical Residual Learning Control},
  author = {Ma, Le and Hu, Shuai and Zhao, Linhan and Yao, Zhenjie and Du, Guanfeng and Jin, Xu},
  year = {2026},
  url = {https://github.com/HuShuai11/boptest-cs-hrlc},
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Third-Party Software

| Software | License | Source |
|----------|---------|--------|
| BOPTEST | BSD-3-Clause | https://github.com/ibpsa/project1-boptest |
| PyTorch | BSD | https://pytorch.org |

BOPTEST is an external dependency and is not included in this repository.

## Acknowledgments

The authors acknowledge the IBPSA BOPTEST project for providing the standardized benchmark environment.
