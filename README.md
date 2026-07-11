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
