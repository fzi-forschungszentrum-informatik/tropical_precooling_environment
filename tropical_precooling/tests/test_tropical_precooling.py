import pytest

from tropical_precooling.env import TropicalPrecooling

def test_estimate_pmv():
    """
    By definition estimate_pmv should return -0.5 if estimating pmv at
    T_min_comfort and 0.5 for T_max_comfort.
    """
    env = TropicalPrecooling()
    assert(env.estimate_pmv(22.0, 23.0, 25) == -1)
    assert(env.estimate_pmv(23.0, 23.0, 25) == -0.5)
    assert(env.estimate_pmv(24.0, 23.0, 25) == 0)
    assert(env.estimate_pmv(25.0, 23.0, 25) == 0.5)
    assert(env.estimate_pmv(26.0, 23.0, 25) == 1.0)
