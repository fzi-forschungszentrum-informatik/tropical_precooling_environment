"""
These tests verify that the usage examples in the Readme.md file yield
the results that have been reported there.
"""

def test_baseline_performance():
    """
    Just copy/paste apart from the last three lines.
    """
    from tropical_precooling.env import TropicalPrecooling

    env = TropicalPrecooling()

    done = False
    obs = env.reset()

    while not done:
        actions = env.T_zSP_baseline
        obs, reward, done, info = env.step(actions)


    expected_performance_measure = 0.0
    actual_performance_measure = env.compute_performance_measure()
    assert actual_performance_measure == expected_performance_measure

def test_intended_usage():
    import numpy as np

    from tropical_precooling.env import TropicalPrecooling


    class TestedAlgorithm():
        """
        Just copy/paste apart from the last three lines.
        """
    import numpy as np

    from tropical_precooling.env import TropicalPrecooling


    class TestedAlgorithm():

        def fit(self, training_data):
            """
            Replace this function with something more sophisticated, e.g.
            a function that uses training_data to train some machine learning
            models.
            """
            pass

        def get_actions(self, obs):
            """
            A simple hand crafted strategy that exploits the cheaper
            electricity prices before 7am, and starts precooling the
            building at 6am (rather then at 7am as the baseline does).
            """
            actions = np.zeros(156)
            actions[:24] = None
            actions[24:] = 23.5

            return actions


    tested_algorithm = TestedAlgorithm()
    env = TropicalPrecooling()

    # You can play aroun with the training data as long as you like.
    # However, you should assure that all your algorithm engineering,
    # hyper-parameter tuning, etc. is finished for your algorithm,
    # as ...
    training_data = env.get_training_data()
    tested_algorithm.fit(training_data)

    # ... you should execute the following code only once and report
    # the scores afterwards. Running this multiple times will likely
    # result that you tend to improve your algorithm to reach a higher
    # performance measure. This carries over information about the
    # test data into your algorithm configuration and thus biases
    # the performance measure.
    done = False
    obs = env.reset()
    while not done:
        actions = tested_algorithm.get_actions(obs)
        obs, reward, done, info = env.step(actions)


    expected_performance_measure = 0.20163148790976004
    actual_performance_measure = env.compute_performance_measure()
    assert actual_performance_measure == expected_performance_measure
