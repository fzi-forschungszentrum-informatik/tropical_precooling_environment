from datetime import datetime

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
        
def test_data_dates_are_used_correctly():
    """
    Verify that the dates of the data files are interpreted correctly.
    The first data sample is from 3rd of January 2014. In V1 this
    seems to have been interpreted as the 1st of March. 
    """
    env = TropicalPrecooling()
    assert env.building_parameters.index[0] == datetime(2014, 1, 3, 0, 0, 0)
    assert env.measured_data.index[0] == datetime(2014, 1, 3, 4, 2, 30)
