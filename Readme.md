# The Tropical Precooling Environment

The Tropical Precooling Environment is an program that can be used to evaluate and benchmark building energy optimization algorithms. For more details and background please see the accompanying [paper](https://dl.acm.org/doi/10.1145/3408308.3427614).



## Installation

This is code is written in Python 3. Given that Python and pip are installed, you can install the environment by typing:

```python
git clone https://github.com/fzi-forschungszentrum-informatik/tropical_precooling_environment.git
pip3 install -e .
```



## Usage

After installation you may test the correct operation of the environment by executing the following code:

```python
from tropical_precooling.env import TropicalPrecooling

env = TropicalPrecooling()

done = False
obs = env.reset()

while not done:
    actions = env.T_zSP_baseline
    obs, reward, done, info = env.step(actions)

print("Performance was: %s" % env.compute_performance_measure())
```

This should should generate the following output:

```
Performance was: 0.0
```



Please see the paper, especially section 5.1, for more details about the environment concepts. Further documentation about implementation and how to interact with the environment are provided within [tropical_precooling/env.py](tropical_precooling/env.py).



The intended usage of the environment is the evaluate or benchmark algorithms for building energy optimization in the following way:

```python
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

print("Performance was: %s" % env.compute_performance_measure())
```

The code above outputs the following, which means that tested algorithm achieves a ~20% improvement in thermal comfort and energy costs compared to the baseline strategy.

```
Performance was: 0.20432590993516558
```



## Building Data

This environment is designed to represent an existing office building in Townsville Australia, as described in section 5.1.1 of the [paper](https://dl.acm.org/doi/10.1145/3408308.3427614). More details can be found in this [paper](https://dl.acm.org/doi/10.1145/3077839.3077847), in particular in section 3.

We are grateful for the support of the Townsville City Council, that not only allowed us to to use measured data from the represented building to implement our environment, but also to publish the raw data. The raw measurements, that have been recorded for the full year of 2014, can be found in [raw_building_data/buildingdata.xlsx](raw_building_data/buildingdata.xlsx).  



## Citation

Please consider citing us if this environment and/or the accompanying [paper](https://dl.acm.org/doi/10.1145/3408308.3427614) was useful for your scientific work. You can use the following BibTex entry:

```latex
@inproceedings{10.1145/3408308.3427614,
author = {W\"{o}lfle, David and Vishwanath, Arun and Schmeck, Hartmut},
title = {A Guide for the Design of Benchmark Environments for Building Energy Optimization},
year = {2020},
isbn = {9781450380614},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3408308.3427614},
doi = {10.1145/3408308.3427614},
booktitle = {Proceedings of the 7th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation},
pages = {220–229},
numpages = {10},
keywords = {building energy management, smart building, reinforcement learning, building control, evaluation, environment, benchmark},
location = {Virtual Event, Japan},
series = {BuildSys '20}
}
```

Other reference formats are provided [here](https://dl.acm.org/doi/10.1145/3408308.3427614) (export citation button).



## Changelog

| Version | Description                                              |
| ------- | -------------------------------------------------------- |
| V1      | Initial version presented and published on BuildSys '20. |



## Contact

Please feel free to contact [David Wölfle](https://www.fzi.de/en/about-us/organisation/detail/address/david-woelfle/) for all inquiries.



## Copyright and license

Code is copyright to the FZI Research Center for Information Technology and released under the MIT [license](./LICENSE).
