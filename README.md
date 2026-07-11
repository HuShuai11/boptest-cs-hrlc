# Cold-Start Hierarchical Residual Learning Control for Heat-Pump Heating in BOPTEST

This repository provides the source code, evaluation scripts and configuration files for:

**BOPTEST-based evaluation of cold-start online control for hydronic heat-pump heating under dynamic electricity pricing**, submitted to *Building Simulation*.

The proposed controller (CS-HRLC) is designed for safe online building heating control under dynamic electricity pricing, with a focus on cold-start deployment without historical operating data.

## Overview

This project implements a hierarchical online supervised residual learning controller (CS-HRLC) for heat-pump-driven building heating systems. The controller combines:

- a rule-guided target generator and PID feedback backbone as the deployable cold-start control path;
- bounded online residual corrections for target-temperature planning and actuator-command refinement;
- progressive participation and comfort-boundary-aware modulation to limit the authority of learned corrections during early deployment and near comfort limits;
- an auxiliary one-step RLS thermal predictor used only for teacher-label construction.

The online control loop does not embed model predictive control (MPC) or reinforcement learning (RL). The controller does not rely on building-specific historical operating trajectories, offline policy pretraining, receding-horizon optimization, or reward-driven exploration.

The experiments are conducted on the `bestest_hydronic_heat_pump` test case from the BOPTEST platform under a highly dynamic electricity price scenario.

## Repository Structure

```text
.
├── main.py
├── plot_ch5_figures.py
├── common/
│   └── boptest_client.py
├── method/
│   ├── rule_teacher.py
│   ├── networks.py
│   ├── controllers.py
│   ├── rls.py
│   ├── tcnu_teacher.py
│   ├── scheduler.py
│   └── action.py
├── ablation/
│   ├── L0_baseline.py
│   ├── L1_tcn_t.py
│   ├── L2_tcn_u_gain.py
│   └── L3_tcn_u_full.py
├── config/
│   └── default.yaml
├── results/
│   └── example/
├── testing/
├── requirements.txt
├── environment.yml
├── releasenotes.md
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

# conda
conda env create -f environment.yml
conda activate boptest-cs-hrlc

# or pip
pip install -r requirements.txt
```

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

## Citation

If you use this code, please cite the associated manuscript and this repository. See [CITATION.cff](CITATION.cff).

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

BOPTEST is an external dependency and is not included in this repository.

## Acknowledgments

The authors acknowledge the IBPSA BOPTEST project for providing the standardized benchmark environment.
