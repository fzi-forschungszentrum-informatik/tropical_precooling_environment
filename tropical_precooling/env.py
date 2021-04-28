import os

import numpy as np
import pandas as pd


class TropicalPrecooling():

    def __init__(self):
        """
        load measurements and parameters for env simulation.
        """
        env_path = os.path.dirname(__file__)
        measurd_data_fnp = os.path.join(
            env_path,
            "data",
            "building_measurements.csv"
        )
        building_parameters_fnp = os.path.join(
            env_path,
            "data",
            "building_parameters.csv"
        )
        self.measured_data = pd.read_csv(
            measurd_data_fnp,
            index_col=0,
            parse_dates=True,
        )
        self.building_parameters = pd.read_csv(
            building_parameters_fnp,
            index_col=0,
            parse_dates=True,
            )

        # Comfort bounds as defined in the paper.
        # in 156 5 minutes slots, 0:36 is the time between 4am and 7am.
        self.T_min_comfort = np.zeros(156)
        self.T_min_comfort[:36] = 25
        self.T_min_comfort[36:] = 23

        self.T_max_comfort = np.zeros(156)
        self.T_max_comfort[:36] = 29
        self.T_max_comfort[36:] = 25

        # Electricity prices in $/kWh as defined in the paper.
        # Off-peak rates apply between 4am and 7am.
        self.e = np.zeros(156)
        self.e[:36] = 9.78
        self.e[36:] = 24.48

        # This is baseline strategy used for computing the performance
        # measure, 27°C for 4am .. 7am and 23.5°C therafter.
        self.T_zSP_baseline = np.zeros(156)
        self.T_zSP_baseline[:36] = None
        self.T_zSP_baseline[36:] = 23.5

        # Define which of the days for which we have data should be used
        # for training and which for testing. Check that we only use
        # those dates which are available in both files (which should be the
        # case for all entries of both files).
        dates_mdata = set(self.measured_data.index.date)
        dates_bparams = set(self.building_parameters.index.date)
        all_dates = sorted(dates_mdata.intersection(dates_bparams))
        self.train_dates = [d for d in all_dates if d.month >= 7]
        self.test_dates = [d for d in all_dates if d.month < 7]

        # Some objects to store outputs of step as this is required
        # to compute the performance measure later.
        self.simulated_dates = []
        self.test_actions = []
        self.test_obs = []

    def simulate_day(self, simulated_date, T_zSP):
        """
        Simulate the zone temperature for one day.

        This starts with the temperature measured at the real building at 4am
        and computes change of temperatue within the time step length of 5
        minutes, by applying equations (3), (4) and (5) from the paper.
        This is repeated until the full day horizon is simulated.

        Parameters
        ----------
        simulated_date : datetime.date
            The date of the day that is simulated. Used to lookup parameters
            and measurements for the equations.
        T_zSP : numpy array with shape (156,)
            Temperature setpoints for every 5 minute slot between 4am and 5pm.

        Returns
        -------
        T_z : numpy array with shape (156,)
            The zone temperature of the simulated day.
        """
        # Extract the building paramters for this day.
        # The .values prevents that the computed values are casted to
        # pandas data types.
        day_selector = self.building_parameters.index.date == simulated_date
        bparams_day = self.building_parameters.loc[day_selector]
        k_a = bparams_day["k_a"].values
        k_o1 = bparams_day["k_o1"].values
        k_o2 = bparams_day["k_o2"].values
        m_so = bparams_day["m_so"].values
        k_c = bparams_day["k_c"].values
        c_pa = bparams_day["c_pa"].values
        C_z = bparams_day["C_z"].values

        # This is the measured data of the simulated day, a pandas df.
        day_selector = self.measured_data.index.date == simulated_date
        mdata_day = self.measured_data.iloc[day_selector]

        # Our container to store the zone temperature.
        T_z = []

        # The measured zone temperature, valid from 04:00:00
        # All other values above are 1d arrays, this has to be an
        # array to to allow building the final array for T_z
        T_z_t = np.asarray([mdata_day.iloc[0]["Zone temperature"]])

        # Make some arrangements to make the notiations below follow
        # the notation in equations given in the paper.
        sim_data = pd.DataFrame(index=mdata_day.index)
        sim_data["T_zSP_t"] = T_zSP
        sim_data["T_s_t"] = mdata_day["Supply air temp"]
        sim_data["T_a_t"] = mdata_day["Outside air temperature"]
        sim_data["theta_CO2_t"] = mdata_day["CO2"]

        # Iterate over rows of sim_data to conveniently get the values
        # for each of the 5 minute blocks.
        for i, row in sim_data.iterrows():
            T_zSP_t = row["T_zSP_t"]
            T_s_t = row["T_s_t"]
            T_a_t = row["T_a_t"]
            theta_CO2_t = row["theta_CO2_t"]

            # Store the current zone temperature first ...
            T_z.append(T_z_t)

            # ... and now compute the delta for the zone temperature of the
            # next timestep.
            #
            m_s_t = m_so + k_c * (T_z_t - T_zSP_t)  # (5)
            # Prevent negative energy prices if the agent sets the setpoint
            # above the current temperature. See for dicussion:
            # https://github.com/fzi-forschungszentrum-informatik/tropical_precooling_environment/issues/1
            m_s_t = np.clip(m_s_t, 0, None)
            Q_cooling_t = c_pa * (m_s_t * (T_s_t - T_z_t))  # (4)

            # Now cooling/heating if AC is switched of.
            if np.isnan(Q_cooling_t):
                Q_cooling_t = 0

            # (5)
            dT_dt = k_a * (T_a_t - T_z_t)
            dT_dt += k_o1 * theta_CO2_t + k_o2
            dT_dt += Q_cooling_t
            dT_dt /= C_z
            dT = dT_dt * 300 # 5 Minutes step length

            T_z_t = T_z_t + dT

        # After asarray T_z has shape (156, 1) we want (156) and flatten thus
        T_z = np.asarray(T_z).flatten()
        return np.asarray(T_z)

    def compute_obs(self, current_date, next_date, T_zSP):
        """
        Generate the content for obs.

        First compute the zone temperature for the current date (which will be
        the previous day for the agent as it would receive this data after day
        has ended). Then look up / compute the remaining data.

        Parameters
        ----------
        current_date : datetime.date
            The date that is used to load the data for obs content 0 to 4.
        next_date : datetime.date
            The date that is used to load the data for obs content 5 and 6.
        T_zSP : numpy array with shape (156,)
            Temperature setpoints for every 5 minute slot between 4am and 5pm.

        Returns
        -------
        obs : numpy array with shape (7, 156)
            Observed data from the simulated building, each quantitiy for
            every 5 minute slot between 4am and 5pm, i.e. 156 values per
            quantity. These are:
                0: The zone temperature of the previous day in °C
                1: The supply air temperature of the previous day in °C.
                2: The ambient temperature of the previous day in °C.
                3: The CO_2 values of the previous day in ppm.
                4: The energy costs of the previous day in $.
                5: The perfect ambient temperature forecast for the current
                   day in °C.
                6: The electricity costs for the current day in cents/kWh.

        """
        # 0: The zone temperature of the previous day in °C
        T_z = self.simulate_day(
            simulated_date=current_date,
            T_zSP=T_zSP,
        )

        # 1, 2, 3: Retrieve the values that are just loaded from measured data.
        # This is the measured data of the simulated day, a pandas df.
        day_selector = self.measured_data.index.date == current_date
        mdata_day = self.measured_data.iloc[day_selector]
        T_s = mdata_day["Supply air temp"]
        T_a = mdata_day["Outside air temperature"]
        theta_CO2 = mdata_day["CO2"]

        # 4: The energy costs of the previous day in $.
        E = self.estimate_energy_costs(
            T_z=T_z,
            T_zSP=T_zSP,
            e=self.e,
            simulated_date=current_date,
        )

        # 5: The perfect ambient temperature forecast for the current day in °C.
        next_day_selector = self.measured_data.index.date == next_date
        mdata_next_day = self.measured_data.iloc[next_day_selector]
        T_a_next_day = mdata_next_day["Outside air temperature"]

        # 6: The electricity costs for the current day in cents/kWh.
        # These never change.
        e_next_day = self.e

        obs = np.asarray([
            T_z,
            T_s,
            T_a,
            theta_CO2,
            E,
            T_a_next_day,
            e_next_day
        ])
        return obs

    def get_training_data(self):
        """
        Returns the training data, i.e. the the (baseline) actions and
        corresponding observations.

        Returns
        -------
        training_actions : list of numpy arrays with shape (156,)
            The actions that have been taken by the baseline agent.
        training_obs : list of numpy arrays with shape (7, 156)
            The observations that have resulted from the actions.
            See step method for details about the content of obs objects.
        """
        training_actions = []
        training_obs = []
        for i in range(0, len(self.train_dates)-1):
            current_date = self.train_dates[i]
            next_date = self.train_dates[i+1]

            T_zSP = self.T_zSP_baseline
            obs = self.compute_obs(
                current_date=current_date,
                next_date=next_date,
                T_zSP=T_zSP,
            )

            training_actions.append(T_zSP)
            training_obs.append(obs)

        return training_actions, training_obs


    def step(self, actions):
        """
        Simulate one day of building operation.

        Parameters
        ----------
        actions : numpy array with shape (156,)
            Temperature setpoints for every 5 minute slot between 4am and 5pm.
            The actual building doesn't support setpoints below 10°C. Setpoints
            can also be set to None which is interpreted as AC off. Setpoints
            above 35°C will be interpreted as AC off.

        Returns
        -------
        obs : numpy array with shape (7, 156)
            Observed data from the simulated building, each quantitiy for
            every 5 minute slot between 4am and 5pm, i.e. 156 values per
            quantity. These are:
                0: The zone temperature of the previous day in °C
                1: The supply air temperature of the previous day in °C.
                2: The ambient temperature of the previous day in °C.
                3: The CO_2 values of the previous day in ppm.
                4: The energy costs of the previous day in $.
                5: The perfect ambient temperature forecast for the current
                   day in °C.
                6: The electricity costs for the current day in cents/kWh.

        reward : None
            This environment emits no reward, as the building doesn't emit one
            either. This field is kept for consistency with OpenAI gym
            conventions.
        done : bool
            True after the last day has been simulated.
        info : dict
            Always an empty dict as no additional information are provided
            for the user of the environment. This field is kept for consistency
            with OpenAI gym conventions.
        """
        # Restrict setpoints to values that are tyical for AC systems.
        # Setting this to crazy high/low values will anyway yield bad
        # performance measures. However, without the cliping some numerical
        # instabilties may occure.
        actions = np.clip(actions, 10, None)
        # Any setpoint then 35°C is interpreted as leave AC of.
        actions[actions>35] = None

        reward = None
        done = False
        info = {}

        # Determine the date of the current day and also check if this is the
        # last day that is simulated.
        current_date = self.current_step_date
        index_current_date = self.test_dates.index(current_date)
        if index_current_date + 2 == len(self.test_dates):
            done = True
        elif index_current_date + 2 > len(self.test_dates):
            raise RuntimeError("Environment is done already.")
        next_date = self.test_dates[index_current_date + 1]

        obs = self.compute_obs(
            current_date=current_date,
            next_date=next_date,
            T_zSP=actions,
        )

        # Store the actions and obs as these are required to compute the
        # performance measure later
        self.simulated_dates.append(current_date)
        self.test_actions.append(actions)
        self.test_obs.append(obs)

        # Increment so next call to step advances in time.
        self.current_step_date = next_date

        return obs, reward, done, info

    def reset(self):
        """
        Reset and init the environment.

        Returns obs for one day following the baseline strategy. Although
        most of this information will not be of worth for the agent, it has
        the advantage that the obs format stays consistent.

        This function also erases the recorded values that might have been
        stored while the agent has interacted with the step function.

        Returns
        -------
        obs : numpy array with shape (7, 156)
            Observed data from the simulated building, each quantitiy for
            every 5 minute slot between 4am and 5pm, i.e. 156 values per
            quantity. These are:
                0: The zone temperature of the previous day in °C
                1: The supply air temperature of the previous day in °C.
                2: The ambient temperature of the previous day in °C.
                3: The CO_2 values of the previous day in ppm.
                4: The energy costs of the previous day in $.
                5: The perfect ambient temperature forecast for the current
                   day in °C.
                6: The electricity costs for the current day in cents/kWh.

        """
        self.simulated_dates = []
        self.test_actions = []
        self.test_obs = []

        current_date = self.test_dates[0]
        next_date = self.test_dates[1]

        T_zSP = self.T_zSP_baseline
        obs = self.compute_obs(
            current_date=current_date,
            next_date=next_date,
            T_zSP=T_zSP,
        )

        self.current_step_date = next_date

        return obs

    def compute_performance_measure(self):
        """
        Compute performance measure as in equation (7) in the paper.

        This loads the recorded data about actions and obs generated by
        the evaluated agent automatically.

        Returns
        -------
        performance_measure : float
        """
        performance_measure = 0

        # Zone temperatures, Energy costs and PMV for the canidate algorthm,
        # These are arrays with shape (len(self.simulated_dates), 156).
        T_z_ca = np.asarray([a[0] for a in self.test_obs])
        E_ca = np.asarray([a[4] for a in self.test_obs])
        PMV_ca = self.estimate_pmv(
            T_z=T_z_ca,
            T_min_comfort=self.T_min_comfort,
            T_max_comfort=self.T_max_comfort,
        )

        # Now compute the corresponding values for the baseline, this is most
        # conveniently done by simulating the test phase with following the
        # baseline strategy.
        env_bl = TropicalPrecooling()
        done = False
        _ = env_bl.reset()
        while not done:
            _, _, done, _ = env_bl.step(actions=env_bl.T_zSP_baseline)

        T_z_bl = np.asarray([a[0] for a in env_bl.test_obs])
        E_bl = np.asarray([a[4] for a in env_bl.test_obs])
        PMV_bl = self.estimate_pmv(
            T_z=T_z_bl,
            T_min_comfort=self.T_min_comfort,
            T_max_comfort=self.T_max_comfort,
        )

        # Apply equation (7)
        performance_measure = 1
        performance_measure -= 0.5 * E_ca.sum() / E_bl.sum()
        performance_measure -= 0.5 * abs(PMV_ca).sum() / abs(PMV_bl).sum()

        return performance_measure

    def estimate_energy_costs(self, T_z, T_zSP, e, simulated_date):
        """
        Compute the estimated energy costs based on equation (6) from the paper.

        Q_cooling_t has already been computed in self.simulate_day. However,
        this method should also work for cases where the zone temperature has
        been measured, especially to compute the performance measure.

        Parameters
        ----------
        T_z : numpy array with shape (156,)
            The zone temperature for one or several days.
        T_zSP : numpy array with shape like T_z
            The zone setpoint temperature aka. actions.
        e : float or numpy array with shape (156,)
            The electricity prices for every 5 minute slot in $/kWh.
        simulated_date : datetime.date
            The date of the day that is simulated. Used to lookup parameters
            and measurements for the equations.

        Returns
        -------
        E : numpy array with shape of T_z
            The energy costs of AC operation for every 5 minute slot.
        """
        # Extract the required building paramters for this day.
        day_selector = self.building_parameters.index.date == simulated_date
        bparams_day = self.building_parameters.loc[day_selector]
        m_so = bparams_day["m_so"].values
        k_c = bparams_day["k_c"].values
        c_pa = bparams_day["c_pa"].values
        COP = bparams_day["COP"].values

        # Get the supply air temperature from measurements.
        day_selector = self.measured_data.index.date == simulated_date
        T_s = self.measured_data.iloc[day_selector]["Supply air temp"].values

        # The variables have no trailing _t (reresenting the (t) in the
        # equations as these are arrays that hold may of these variables.
        m_s = m_so + k_c * (T_z - T_zSP)  # (5)
        m_s = np.clip(m_s, 0, None)
        Q_cooling = c_pa * (m_s * (T_s - T_z))  # (4)

        # Set cooling power to zero if AC was off.
        Q_cooling[np.isnan(Q_cooling)] = 0

        E = - Q_cooling * e / COP
        return E


    def estimate_pmv(self, T_z, T_min_comfort, T_max_comfort):
        """
        Computes an PMV estimate from given min and max comfort temperatures.

        PMV is usually computed as with Fanger's equation as
        (0.303 * e^(-0.036*M) + 0.028) * L
        whereby M is the metabolic rate and L is linear proportional to the
        (indoor) air temperature. The comfort range for PMV is typically
        expected to lay within the range between -0.5 and 0.5. However,
        in our case the comfort range has already been defined by the facility
        manager. Assuming thus that the minimum comfort temperature is
        equivalent to PMV=-0.5 and the maxmimum comfort temperature is
        equivalent to PMV=0.5, we estimate PMV with linear interpolation
        between these points.

        Arguments:
        ----------
        T_z : float or array with shape (156,) or (n, 156).
            The zone temperature for one or several days for which PMV should
            be esimated.
        T_min_comfort : float or array with shape (156,).
            The minimum thermal comfort temperature equivalent to PMV=-0.5.
        T_max_comfort : float or array with shape (156,).
            The maximum thermal comfort temperature equivalent to PMV=0.5.

        Returns:
        --------
        PMV : float or array
            depending on the input of T_zone.
        """
        # This is a simple linear fit through two points.
        PMV = (0.5 - -0.5) / (T_max_comfort - T_min_comfort) * (T_z - T_min_comfort) + -0.5
        return PMV
