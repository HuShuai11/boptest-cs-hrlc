"""
Action mapping: converts normalized control signal u ∈ [0,1] to BOPTEST control overrides.
"""

import numpy as np


def paper_action(u_final, T_set_C):
    """
    Maps u_final ∈ [0,1] to BOPTEST control signals:
    - u > 0.05: heat pump + fan + pump all active, u_fan = u_pum = max(0.2, u)
    - u ≤ 0.05: all off (heat pump off)
    """
    u_hp = float(np.clip(u_final, 0.0, 1.0))
    if u_hp > 0.05:
        u_fan = max(0.2, u_hp)
        u_pum = max(0.2, u_hp)
    else:
        u_hp = 0.0
        u_fan = 0.0
        u_pum = 0.0
    return {
        "oveTSet_activate": 1,
        "oveTSet_u": float(T_set_C + 273.15),
        "oveHeaPumY_activate": 1,
        "oveHeaPumY_u": float(u_hp),
        "oveFan_activate": 1,
        "oveFan_u": float(u_fan),
        "ovePum_activate": 1,
        "ovePum_u": float(u_pum),
    }
