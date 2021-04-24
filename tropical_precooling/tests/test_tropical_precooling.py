import pytest
import numpy as np

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
    
def test_no_negative_energy_costs():
    """
    Test that the environment allows no negative energy costs by setting
    the setpoint temperature higher then the room temperature. This would
    correspond to a system that would create electricity by heating up the
    zone.
    """
    env = TropicalPrecooling()
    done = False
    obs = env.reset()
    while not done:
        obs, reward, done, info = env.step(np.ones(156)*10**9)
        assert np.min(obs[4]) >= 0
    
def test_no_freezing_temperatures():
    """
    Verify that the AC system does not cool the zone down to cyrogenic
    temperatures, even if the setpoint is set to crazy low values.
    """
    env = TropicalPrecooling()
    done = False
    obs = env.reset()
    while not done:
        obs, reward, done, info = env.step(np.ones(156)*-10**9)
        assert np.min(obs[0]) > 10
