import scipy.stats as st
from scipy.linalg import lstsq
import numpy as np
import math
import sys
import statistics
import DPmedian

import All_DP_median_algorithms as DPM

min_point = -.5 # arbitrary bounds for p25 and p75 estimates
max_point = 1.5
min_slope = -50 # arbitrary bounds for slopes
max_slope = 50
default_beta = 9.0
default_beta_prop = 0.5
default_d = 3
default_delta = 10.0**(-6.0)
default_k=3 #k=-1 results in the full TS


'''
DPTheilSen Main Methods:
1) dpMedTS_exp: computes DPExpTheilSen
2) dpMedTS_exp_wide: computes DPWideTheilSen
3) dpMedTS_ss_ST_no_split: computes SSTheilSen with a smooth sensitivity calculation
   based on the student's T distribution.
'''

# ----- HELPER CODE -------
# Compute nC2 estimates for each point in xnew
def computeAllEsts(x, y, n, xnew, min_est=min_point, max_est=max_point):
    # initialize array of arrays for xnew estimates
    xnew_ests = [[] for i in range(len(xnew))]

    # compute pairwise estimates
    for p in range(n):
        for q in range(p+1, n):
            x_delta = float(x[q]-x[p])
            if x_delta != 0: # instead of setting x_delta to 0.001, just don't compute slope if x_delta is 0
                slope = float(y[q]-y[p])/ float(x_delta) # compute slope between two points
                xmean = (x[q]+x[p])/2 # use xmean and ymean to compute xnew estimates
                ymean = (y[q]+y[p])/2
                for i in range(len(xnew)):
                    xnew_ests[i].append(slope*np.array(xnew[i])+(ymean-slope*xmean))

    return xnew_ests

# Compute n/2 estimates for each point in xnew
def computeHalfEsts(x, y, n, xnew, min_est=min_point, max_est=max_point):
    # initialize array of arrays for xnew estimates
    xnew_ests = [[] for i in range(len(xnew))]

    # use for random matching
    z = np.arange(n)
    z = np.random.permutation(z)

    # compute n/2 estimates
    for i in range(0, n-1, 2):
        p = z[i]
        q = z[i+1]
        x_delta = float(x[q]-x[p])
        if x_delta != 0: # instead of setting x_delta to 0.001, just don't compute slope if x_delta is 0
            slope = float(y[q]-y[p])/ float(x_delta) # compute slope between two points
            xmean = (x[q]+x[p])/2
            ymean = (y[q]+y[p])/2
            for i in range(len(xnew)):
                xnew_ests[i].append(slope*np.array(xnew[i])+(ymean-slope*xmean))

    return xnew_ests

# Clip estimates
def clipEsts(xnew_ests, min_est=min_point, max_est=max_point):
    # clip each list of estimates 
    for i in range(len(xnew_ests)):
        xnew_ests[i] = list(np.clip(np.array(xnew_ests[i]), min_est, max_est))

    return xnew_ests

# Compute clipped TS estimates and new epsilon
def prepForDPMedian(x, y, n, eps, xnew, half):
    # Compute n/2 or nC2 estimates and set new_eps accordingly
    if half:
        # compute n/2 estimates
        xnew_ests = computeHalfEsts(x, y, n, xnew)
        # compute new epsilon; each data point affects 1 point
        new_eps = eps
    else:
        # compute nC2 (pairwise) estimates
        xnew_ests = computeAllEsts(x, y, n, xnew)
        # compute new epsilon; each data point affects n-1 points
        new_eps = float(eps)/float(n-1)

    # clip estimates
    xnew_ests = clipEsts(xnew_ests, min_est=min_point, max_est=max_point)

    return xnew_ests, new_eps

# Translate from (eps,delta)-DP to 1/2 new-eps^2 -CDP 
def convertApproxDPtoCDP(eps, delta):
    new_eps = math.sqrt(2)*(math.sqrt(math.log10(1.0/float(delta)) + eps) - math.sqrt(math.log10(1.0/float(delta))))
    return new_eps


# ----- TS ALGORITHMS ------
# DP TS using exponential median 
def dpMedTS_exp(x, y, xm, ym, n, eps, xnew, half=False):
    """
    :param x: List of x values
    :param y: List of y values
    :param xm: Mean of x values
    :param ym: Mean of y values
    :param n: Length of x,y
    :param eps: Privacy loss parameter
    :param xnew: Percentiles to give estimates; usually [.25, .75]
    :param half: If true, compute n/2 TS estimates; if false, compute all nC2 estimates
    :return: (eps,0)-DP estimates at points in xnew
    """   
    # get clipped TS estimates and new epsilon     
    xnew_ests, new_eps = prepForDPMedian(x, y, n, eps, xnew, half)
    # compute DP median of TS estimates at each point in xnew
    xnew_dp_ests = []
    for i in range(len(xnew)):
        xnew_dp_ests.append(DPM.dpMedianExp(xnew_ests[i], lower_bound=min_point, upper_bound=max_point, epsilon=new_eps))
    return xnew_dp_ests

# DP TS using wide exponential median of TS slopes
def dpMedTS_exp_wide(x, y, xm, ym, n, eps, xnew, width=0.01, half=False):
    """
    :param x: List of x values
    :param y: List of y values
    :param xm: Mean of x values
    :param ym: Mean of y values
    :param n: Length of x,y
    :param eps: Privacy loss parameter
    :param xnew: Percentiles to give estimates; usually [.25, .75]
    :param width: granularity for widened exponential median
    :param half: If true, compute n/2 TS estimates; if false, compute all nC2 estimates
    :return: (eps,0)-DP estimates at points in xnew
    """
    # get clipped TS estimates and new epsilon     
    xnew_ests, new_eps = prepForDPMedian(x, y, n, eps, xnew, half)
    # compute DP median of TS estimates at each point in xnew
    xnew_dp_ests = []
    for i in range(len(xnew)):
        xnew_dp_ests.append(DPM.dpMedianExpWide(xnew_ests[i], lower_bound=min_point, upper_bound=max_point, epsilon=new_eps, width=width))
    return xnew_dp_ests

# DP TS using smooth sensitivity median with arsinh normal noise addition
def dpMedTS_ss_AN(x, y, xm, ym, n, eps, xnew, smooth_sens, beta=default_beta, half=False):
    """
    :param x: List of x values
    :param y: List of y values
    :param xm: Mean of x values
    :param ym: Mean of y values
    :param n: Length of x,y
    :param eps: Privacy loss parameter
    :param xnew: Percentiles to give estimates; usually [.25, .75]
    :param smooth_sens: beta-smooth sensitivity of the median. Must be included.
    :param beta: Smoothing parameter (positive float)
    :param half: If true, compute n/2 TS estimates; if false, compute all nC2 estimates
    :return: (eps,10^-6)-DP estimates at points in xnew
    """
    assert beta >= 0.0
    # get clipped TS estimates and new epsilon     
    xnew_ests, new_eps = prepForDPMedian(x, y, n, eps, xnew, half)
    # translate from (eps,10^-6)-DP to 1/2 new-eps^2 -CDP
    new_eps = convertApproxDPtoCDP(eps, default_delta)
    # compute DP median of TS estimates at each point in xnew
    xnew_dp_ests = []
    for i in range(len(xnew)):
        xnew_dp_ests.append(DPM.dpMedianArsinhNormal(xnew_ests[i], epsilon=new_eps, beta=beta, smooth_sens=smooth_sens))
    return xnew_dp_ests

# DP TS using smooth sensitivity median with student's T noise addition
def dpMedTS_ss_ST(x, y, xm, ym, n, eps, xnew, beta_prop=default_beta_prop, d=default_d, half=False):
    """
    :param x: List of x values
    :param y: List of y values
    :param xm: Mean of x values
    :param ym: Mean of y values
    :param n: Length of x,y
    :param eps: Privacy loss parameter
    :param xnew: Percentiles to give estimates; usually [.25, .75]
    :param smooth_sens: beta-smooth sensitivity of the median. Must be included.
    :param beta_prop: Proportion of epsilon used to compute smoothing parameter beta
    :param d: degrees of freedom for Student's T distribution
    :param half: If true, compute n/2 TS estimates; if false, compute all nC2 estimates
    :return: (eps,10^-6)-DP estimates at points in xnew
    """
    assert beta_prop < 1.0 
    # get clipped TS estimates and new epsilon     
    xnew_ests, new_eps = prepForDPMedian(x, y, n, eps, xnew, half)
    beta = float(new_eps)* beta_prop /float(d+1)
    # compute DP median of TS estimates at each point in xnew
    xnew_dp_ests = []
    for i in range(len(xnew)):
        smooth_sens = DPmedian.smooth_sens_median(xnew_ests[i], beta, min_point, max_point)[0]
        xnew_dp_ests.append(DPM.dpMedianStudentsT(xnew_ests[i], epsilon=new_eps, beta=beta, smooth_sens=smooth_sens, d=d))
    return xnew_dp_ests


#-----------------------------------------------------------------------------------------------------------------------------
# Sample from Student's T distribution with d degrees of freedom
def studentsTRV(d):
    """
    :param d: degrees of freedom (should be >= 1)
    :returns: random variable sampled from T(d)
    """
    X = np.random.normal(size=d+1)
    Z = X[0] / math.sqrt((sum(X[1:]**2))/d)
    return Z

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
    s = float(epsilon*math.sqrt(d)) / float(d+1)
    res = true_median + (1/s)*smooth_sens*Z

    #print("ST:: d: " + str(d) + ", Z: " + str(Z) + ", s: " + str(s) + ", res: " + str(res))

    return res

# Compute nC2 estimates for each point in xnew
def computekMatchEsts(x, y, n, xnew, k, min_est=min_point, max_est=max_point):
    # initialize array of arrays for xnew estimates
    xnew_ests = [[] for i in range(len(xnew))]
    
    # use for random matching
    z = np.arange(n)
    z = np.random.permutation(z)
    
    a = np.random.choice(n-1, k, replace=False)
    # compute n/2 estimates
    
    for j in a:
        p = z[j]
        q = z[n-1]
        x_delta = float(x[q]-x[p])
        if x_delta != 0: # instead of setting x_delta to 0.001, just don't compute slope if x_delta is 0
            slope = float(y[q]-y[p])/ float(x_delta) # compute slope between two points
            xmean = (x[q]+x[p])/2
            ymean = (y[q]+y[p])/2
            for m in range(len(xnew)):
                xnew_ests[m].append(slope*np.array(xnew[m])+(ymean-slope*xmean))
                
        for i in range(1,int((n-1)/2+1)):
            p = z[(j-i)%(n-1)]
            q = z[(j+i)%(n-1)]
            x_delta = float(x[q]-x[p])
            if x_delta != 0: # instead of setting x_delta to 0.001, just don't compute slope if x_delta is 0
                slope = float(y[q]-y[p])/ float(x_delta) # compute slope between two points
                xmean = (x[q]+x[p])/2
                ymean = (y[q]+y[p])/2
                for m in range(len(xnew)):
                    xnew_ests[m].append(slope*np.array(xnew[m])+(ymean-slope*xmean))
    
    return xnew_ests

"""
Quadratic-time implementation of smooth sensitivity without splitting eps/beta.
To be used for testing faster algorithm.
:param n: length of original dataset
:param x: numpy array of real numbers. Not necessarily sorted.
:param beta: Smoothing paramter (positive float)
:param lower_bound: lower bound for data values in x
:param upper_bound: upper bound for data values in x
:returns: beta-smooth sensitivity of the median under insertion/removal, evaluated at x
"""
#distance (n-1) Smooth sensitivity
def smooth_sens_median_no_split(n, x, beta, lower_bound, upper_bound, k):
    # Constrain value to bounds provided as arguments.
    x = np.clip(x, lower_bound, upper_bound)
    # Now bookend x with lower and upper bounds for the data
    x = np.concatenate([[lower_bound], x, [upper_bound]])
    x.sort()
    m = int(np.floor(len(x)/2))
    max_so_far = -np.inf
    best_pair = []

    #taking care of l=0
    j=min(m, len(x)-1)
    i=max(m-k, 0)
    this_value = (x[j] - x[i]) 
    if this_value > max_so_far:
        max_so_far = this_value
        best_pair = [i, j]
    j=min(m+k, len(x)-1)
    i=max(m, 0)
    this_value = (x[j] - x[i]) 
    if this_value > max_so_far:
        max_so_far = this_value
        best_pair = [i, j]
    
    #taking care of l>0
    for l in range(1, n+1):
        for t in range(l*k+k):
            j = min(m + t, len(x) - 1)
            i = max(m - (l*k+k) + t, 0)
            this_value = (x[j] - x[i]) * np.exp(-beta * l)
            if this_value > max_so_far:
                max_so_far = this_value
                best_pair = [i, j]
    return max_so_far, best_pair

"""
wrapper function that computes
DP TS using smooth sensitivity median with student's T noise addition
set k=-1 if you want to run the full TheilSen.

:param x: List of x values
:param y: List of y values
:param xm: Mean of x values
:param ym: Mean of y values
:param n: Length of x,y
:param eps: Privacy loss parameter
:param xnew: Percentiles to give estimates; usually [.25, .75]
:param smooth_sens: beta-smooth sensitivity of the median. Must be included.
:param beta_prop: Proportion of epsilon used to compute smoothing parameter beta
:param d: degrees of freedom for Student's T distribution
:param half: If true, compute n/2 TS estimates; if false, compute all nC2 estimates
:return: (eps,10^-6)-DP estimates at points in xnew
"""
def dpMedTS_ss_ST_no_split(x, y, xm, ym, n, eps, xnew, beta_prop=default_beta_prop, d=default_d, half=False, k=default_k):
    if k<0:
        k=len(x)-1
    else:
        k=k

    assert beta_prop < 1.0 
    # get clipped TS estimates and new epsilon    
    xnew_ests = computekMatchEsts(x, y, n, xnew, k, min_point, max_point)
    xnew_ests = np.clip(xnew_ests, min_point, max_point)
    beta = float(eps)* beta_prop /float(d+1)
    
    # compute DP median of TS estimates at each point in xnew
    xnew_dp_ests = []
    for i in range(len(xnew)):
        smooth_sens, smooth_sens_pair = smooth_sens_median_no_split(n, xnew_ests[i], beta, min_point, max_point, k)
        xnew_dp_ests.append(dpMedianStudentsT(xnew_ests[i], epsilon=eps, beta=beta, smooth_sens=smooth_sens, d=d))
    return xnew_dp_ests
