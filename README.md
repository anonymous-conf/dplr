Differentially Private Linear Regression
========================================

This repository contains algorithms for performing differentially private
simple linear regression. We also provide an example usage of the main algorithms.

Families of Algorithms
----------------------

There are three families of algorithms we've implemented for
differentially private (DP) simple linear regression:

1. `DPGradDescent`: A DP mechanism that uses differentially private gradient descent
to solve the convex optimization problem that defines OLS (Ordinary Least Squares).

2. `DPTheilSen`: A DP version of Theil-Sen, a robust linear regression estimator that
computes the point estimate for every pair of points and outputs the median of
these estimates. We consider some variants of this algorithm that use different DP median
algorithms: `DPExpTheilSen`, `DPWideTheilSen`, and `DPSSTheilSen`.

3. `NoisyStats`: A DP mechanism that perturbs the sufficient statistics for OLS. It has
two main advantages: it is no less efficient than its non-private analogue, and it allows us to
release DP versions of the sufficient statistics without any extra privacy cost.


Implementations
---------------

1. `DPGradDescent`: [link to main code file](https://github.com/anonymous-conf/dplr/code/DPGradDescent.py)

2. `dpMedTS_exp`: computes `DPExpTheilSen` [link to main code file](https://github.com/anonymous-conf/dplr/code/DPTS_algorithms.py)

3. `dpMedTS_exp_wide`: computes `DPWideTheilSen` [link to main code file](https://github.com/anonymous-conf/dplr/code/DPTS_algorithms.py)

4. `dpMedTS_ss_ST_no_split`: computes `DPSSTheilSen` with a smooth sensitivity calculation based on
the student's T distribution [link to main code file](https://github.com/anonymous-conf/dplr/code/DPTS_algorithms.py)

5. `NoisyStats`: [link to main code file](https://github.com/anonymous-conf/dplr/code/NoisyStats.py)

Example Usage
-------------

In [example.py](https://github.com/anonymous-conf/dplr/code/example.py),
we show how to run each method.

Experimental Evaluation in Main Paper
-------------------------------------

1. `Opportunity Insights Data`

2. `Washington, DC Bikeshare UCI Dataset`

3. `Carbon Nanotubes UCI Dataset`

4. `Stock Exchange UCI Dataset`

5. `Synthetic Datasets`

