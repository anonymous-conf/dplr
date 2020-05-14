Differentially Private Linear Regression
========================================

This repository contains algorithms and sample data files for performing differentially private
simple linear regression.

Families of Algorithms
----------------------

There are three families of algorithms we've implemented for
differentially private (DP) simple linear regression:

1. `DPGradDescent`: A DP mechanism that uses differentially private gradient descent
to solve the convex optimization problem that defines OLS (Ordinary Least Squares).

2. `DPTheilSen`: A DP version of Theil-Sen, a robust linear regression estimator that
computes the point estimate for every pair of points, and outputs the median of
these estimates. We consider some variants of this algorithm that use different DP median
algorithms: `DPExpTheilSen`, `DPSSTheilSen`, and `DPWideTheilSen`.

3. `NoisyStats`: A DP mechanism that perturbs the sufficient statistics for OLS. It has
two main advantages: it is no less efficient than its non-private analogue, and it allows us to
release DP vrsions of the sufficient statistics without any extra privacy cost.


Implementations
---------------

1. `DPGradDescent`: `link to code here`

2. `DPExpTheilSen`: `link to code here`

3. `DPSSTheilSen`: `link to code here`

4. `DPWideTheilSen`: `link to code here`

5. `NoisyStats`:  [link to main code file](https://github.com/anonymous-conf/dplr/code/NoisyStats.py)

Data Files for Experimental Evaluation
--------------------------------------

1. `Opportunity Insights Data`

2. `Washington, DC Bikeshare UCI Dataset`

3. `Carbon Nanotubes UCI Dataset`

4. `Stock Exchange UCI Dataset`

5. `Synthetic Datasets`

