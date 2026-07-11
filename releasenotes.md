# Release Notes

CS-HRLC depends on BOPTEST as its simulation platform. The first two digits of the CS-HRLC version number match the compatible BOPTEST version. The last digit is reserved for internal edits specific to this repository. See [here](https://github.com/ibpsa/project1-boptest/blob/master/releasenotes.md) for BOPTEST release notes.

## CS-HRLC v0.9.0

Released on 10/07/2026.

- Initial public release compatible with BOPTEST v0.9.0.
- Supports `bestest_hydronic_heat_pump` test case under `highly_dynamic` electricity tariff.
- Includes full proposed controller (L3: TCN_T + TCN_U full) and three ablation variants (L0–L2).
- Provides example KPI outputs for Peak and Typical scenarios.
- Progressive participation and comfort-boundary-aware modulation are included as configurable safety mechanisms.
