import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.linalg import lstsq
import numpy as np
import math
import sys
import statistics

eps=8
min_point = -.5 # arbitrary bounds for estimates in [0,1]
max_point = 1.5
proj = 0.001 # Use if we want to project negative |s| values to nearly 0

# --- SMOOTH SENSITIVITY NOISE DISTRIBUTIONS ---------------
# Sample from Laplace log-normal distribution
def laplaceLogNormalRV(sigma):
    """
    :param sigma: LLN parameter
    :returns: random variable sampled from LLN(sigma)
    """
    X = np.random.laplace()
    Y = np.random.normal()
    Z = X * math.exp(sigma * Y)
    return Z

# Sample from uniform log-normal distribution
def uniformLogNormalRV(sigma):
    """
    :param sigma: ULN parameter
    :returns: random variable sampled from ULN(sigma)
    """
    X = np.random.uniform(-1, 1)
    Y = np.random.normal()
    Z = X * math.exp(sigma * Y)
    return Z

# Sample from Student's T distribution with d degrees of freedom
def studentsTRV(d):
    """
    :param d: degrees of freedom (should be >= 1)
    :returns: random variable sampled from T(d)
    """
    X = np.random.normal(size=d+1)
    Z = X[0] / math.sqrt((sum(X[1:]**2))/d)
    return Z

# Sample from arsinh-normal distribution with d degrees of freedom
def arsinhNormalRV(sigma):
    """
    :param sigma: arsinh normal parameter
    :returns: random variable sampled from arsinh-normal(sigma)
    """
    Y = np.random.normal()
    Z = (1/sigma)*math.sinh(sigma*Y)
    return Z

# Compute beta-smooth sensitivity of the median (copied from DPmedian.py)
def smooth_sens_median_slow(x, beta, lower_bound, upper_bound):
    """
    Quadratic-time implementation of smooth sensitivity.
    To be used for testing faster algorithm.
    :param x: numpy array of real numbers. Not necessarily sorted.
    :param beta: Smoothing paramter (positive float)
    :param lower_bound: lower bound for data values in x
    :param upper_bound: upper bound for data values in x
    :returns: beta-smooth sensitivity of the median under insertion/removal, evaluated at x
    """
    # Constrain value to bounds provided as arguments.
    x = np.clip(x, lower_bound, upper_bound)
    # Now bookend x with lower and upper bounds for the data
    x = np.concatenate([[lower_bound], x, [upper_bound]])
    x.sort()
    n = len(x)
    m = int(np.floor(len(x)/2))
    max_so_far = -np.inf
    best_pair = []
    for i in range(m+1):
        for j in range(max(m, i+1), n):
            this_value = (x[j] - x[i]) * np.exp(-beta * (j - i -1)) / 2
            if this_value > max_so_far:
                max_so_far = this_value
                best_pair = [i - 1, j - 1]
    return max_so_far, best_pair

# --- SMOOTH SENSITIVITY MEDIAN ALGORITHMS ---------------
# Compute .5eps^2-CDP smooth sens median for bounded data using laplace log-normal noise distribution.
def dpMedianLLN(x, epsilon, beta, smooth_sens):
    """
    :param x: numpy array of real numbers. Not necessarily sorted.
    :param epsilon: parameter for CDP
    :param beta: Smoothing parameter (positive float)
    :param smooth_sens: beta-smooth sensitivity of the median
    :returns: 1/2 epsilon^2 CDP median
    """
    true_median = np.median(x)
    sigma = max(2*beta/epsilon, 1/2)
    Z = laplaceLogNormalRV(sigma)
    s = math.exp(-(3/2)*(sigma**2)) * (epsilon - (abs(beta)/abs(sigma)))
    res = true_median + (1/s)*smooth_sens*Z

    #print("LLN:: sigma: " + str(sigma) + ", Z: " + str(Z) + ", s: " + str(s) + ", res: " + str(res))

    return res 

# Compute .5eps^2-CDP smooth sens median for bounded data using uniform log-normal noise distribution.
def dpMedianULN(x, epsilon, beta, smooth_sens):
    """
    :param x: numpy array of real numbers. Not necessarily sorted.
    :param epsilon: parameter for CDP
    :param beta: Smoothing parameter (positive float)
    :param smooth_sens: beta-smooth sensitivity of the median
    :returns: 1/2 epsilon^2 CDP median
    """
    true_median = np.median(x)
    sigma = math.sqrt(2)
    Z = laplaceLogNormalRV(sigma)
    s = math.exp(-(3/2)*(sigma**2)) * math.sqrt(math.pi * sigma**2 / 2) * (epsilon - (abs(beta)/abs(sigma)))
    res = true_median + (1/s)*smooth_sens*Z

    #print("ULN:: sigma: " + str(sigma) + ", Z: " + str(Z) + ", s: " + str(s) + ", res: " + str(res))

    return res

# Compute .5eps^2-CDP smooth sens median for bounded data using student's T noise distribution.
def dpMedianStudentsT(x, epsilon, beta, smooth_sens, d):
    """
    :param x: numpy array of real numbers. Not necessarily sorted.
    :param epsilon: parameter for CDP
    :param beta: Smoothing parameter (positive float)
    :param smooth_sens: beta-smooth sensitivity of the median
    :param d: degrees of freedom for Student's T distribution
    :returns: 1/2 epsilon^2 CDP median
    """
    true_median = np.median(x)
    Z = studentsTRV(d)
    s = 2 * math.sqrt(d) * (epsilon - abs(beta) * (d+1)) / (d+1)
    res = true_median + (1/s)*smooth_sens*Z

    #print("ST:: d: " + str(d) + ", Z: " + str(Z) + ", s: " + str(s) + ", res: " + str(res))

    return res

# Compute .5eps^2-CDP smooth sens median for bounded data using arsinh-normal noise distribution.
def dpMedianArsinhNormal(x, epsilon, beta, smooth_sens):
    """
    :param x: numpy array of real numbers. Not necessarily sorted.
    :param epsilon: parameter for CDP
    :param beta: Smoothing parameter (positive float), 
    :param smooth_sens: beta-smooth sensitivity of the median
    :returns: 1/2 epsilon^2 CDP median
    """
    true_median = np.median(x)
    sigma = 2/math.sqrt(3)
    Z = arsinhNormalRV(sigma)
    s = (6 * sigma / (4 + 3 * sigma**2)) * (epsilon -  math.sqrt(abs(beta) * ((abs(beta)/(sigma**2)) + (1/sigma) + 2)))
    res = true_median + (1/s)*smooth_sens*Z

    #print("AN:: sigma: " + str(sigma) + ", Z: " + str(Z) + ", s: " + str(s) + ", res: " + str(res))

    return res

# --- EXPONENTIAL MECHANISM MEDIAN ALGORITHMS ---------------
# Computes eps-differentially private median for bounded data.
def dpMedianExp(x, lower_bound=min_point, upper_bound=max_point, epsilon=eps):
    """
    :param x: List of real numbers
    :param lower_bound: Lower bound on values in x 
    :param upper_bound: Upper bound on values in x 
    :param epsilon: Desired value of epsilon for (epsilon,0)-differential privacy
    :return: eps-DP approximate median
    """
    # First, sort the values
    z = x.copy() #making a working copy of the data. 
    z.sort()
    n = len(z)
    # if n is even, add a copy of the true median. [Adam: why do this?] 
    if n % 2 == 0:
        z.insert(math.ceil(n/2), statistics.median(z))
        n=n+1
    # Bookend z with lower bound and upper bound
    # z[0] = lower_bound, z[n+1] = upper_bound
    z.insert(0,lower_bound) 
    z.append(upper_bound) 
     
    # print(z)
    # Iterate through z, assigning scores to each interval given by adjacent indices
    # currentMax and currentInt keep track of highest score and corresponding interval
    currentMax = float("-inf")
    currentInt = -1
    for i in range(1, n+2):
        start = z[i-1]
        end = z[i]
        # Compute length of interval on logarithmic scale
        length = end-start
        loglength = float("-inf") if (length <= 0) else math.log(length)
        # The rungheight is the score of each individual point in the interval
        rungheight = math.ceil(abs((i-1/2)-(n+1)/2))
        # The score has two components:
        # (1) Distance from index to median (closer -> higher score)
        # (2) Length of the interval on a logarithmic scale (larger -> higher score)
        # We include this since all values in the interval have the same score.
        score = -(epsilon/2) * rungheight + loglength         
                # Adam: The old code was wrong–it samples from a different distribution. 
                # Need Gumbel noise to sample from exponential mechanism 
                # See https://pdfs.semanticscholar.org/2782/e47c5b0c8a2ce14eae8713b6f2db864f07c8.pdf
        # Add noise scaled to global sensitivity using exponential mechanism 
        noisyscore = score + np.random.gumbel(loc=0.0, scale=1.0)
        if (noisyscore > currentMax): #This should be always satisfied when i=1. 
            currentInt = i
            currentMax = noisyscore
        #print("hi", i, start, end, length, loglength, rungheight, score, noisyscore)
    # Select uniformly from the highest scoring interval given by currentInt
    #print(currentInt)
    #print(z[currentInt-1], z[currentInt])
    return np.random.uniform(low=z[currentInt-1], high=z[currentInt])

# Computes eps-differentially private median for bounded data.
def dpMedianExpWide(x, lower_bound=min_point, upper_bound=max_point, epsilon=eps, width=0.01):
    """
    :param x: List of real numbers
    :param lower_bound: Lower bound on values in x 
    :param upper_bound: Upper bound on values in x 
    :param epsilon: Desired value of epsilon for (epsilon,0)-differential privacy
    :return: eps-DP approximate median
    """
    # First, sort the values
    z = x.copy() #making a working copy of the data. 
    z.sort()
    n = len(z)
    # if n is even, add a copy of the true median. [Adam: why do this?] 
    if n % 2 == 0:
        z.insert(math.ceil(n/2), statistics.median(z)) # Can remove math.ceil since n is even
        n=n+1
    # insert a second copy of the median
    z.insert(math.ceil(n/2), statistics.median(z)) 
    n = n+1
    # The goal here is to put a buffer around the median in the exponential mechanism. The parameter fat controls how much of a buffer.
    # We do this by moving all points fat away from the true median.
    for i in range(math.floor(n/2)): # Can remove math.floor since n is even
        z[i]=max(lower_bound,z[i]-width)
        z[n-i-1]=min(z[n-i-1]+width, upper_bound)
    # Bookend z with lower bound and upper bound
    z.insert(0,lower_bound) 
    z.append(upper_bound) 
    # print(z)
    # Iterate through z, assigning scores to each interval given by adjacent indices
    # currentMax and currentInt keep track of highest score and corresponding interval
    currentMax = float("-inf")
    currentInt = -1
    for i in range(1, n+2):
        start = z[i-1]
        end = z[i]
        # Compute length of interval on logarithmic scale
        length = end-start
        loglength = float("-inf") if (length <= 0) else math.log(length)
        # The rungheight is the score of each individual point in the interval
        rungheight = math.ceil(abs((i-1/2)-(n+1)/2))
        # The score has two components:
        # (1) Distance from index to median (closer -> higher score)
        # (2) Length of the interval on a logarithmic scale (larger -> higher score)
        # We include this since all values in the interval have the same score.
        score = -(epsilon/2) * rungheight + loglength         
                # Adam: The old code was wrong–it samples from a different distribution. 
                # Need Gumbel noise to sample from exponential mechanism 
                # See https://pdfs.semanticscholar.org/2782/e47c5b0c8a2ce14eae8713b6f2db864f07c8.pdf
        # Add noise scaled to global sensitivity using exponential mechanism 
        noisyscore = score + np.random.gumbel(loc=0.0, scale=1.0)
        if (noisyscore > currentMax): #This should be always satisfied when i=1. 
            currentInt = i
            currentMax = noisyscore
        #print("hi", i, start, end, length, loglength, rungheight, score, noisyscore)
    # Select uniformly from the highest scoring interval given by currentInt
    #print(currentInt)
    #print(z[currentInt-1], z[currentInt])
    return np.random.uniform(low=z[currentInt-1], high=z[currentInt])

