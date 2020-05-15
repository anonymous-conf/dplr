import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.linalg import lstsq
import numpy as np
import math
import sys
import statistics

eps = 0.5


# Computes eps-differentially private median for bounded data. This algorithm is the exponential mechanism with a "granularity" parameter around the true median.
def dpMedian(x, lower_bound, upper_bound, epsilon = eps, granularity = 0.01):
    """
    :param x: List or numpy array of real numbers
    :param lower_bound: Lower bound on values in x. Defaults to 0. 
    :param upper_bound: Upper bound on values in x Defaults to 1.
    :param epsilon: Desired value of epsilon for (epsilon,0)-differential privacy
    :param granularity: This parameter helps avoid pathological behavior when values are tightly contrated (e.g all equal). It is a lower bound on the algorithm's accuracy.
    :return: eps-DP approximation to median
    """

    # Now add bookends and copies of the true median
    # to get correct intervals for exponential mechanism.
    # Add one copy of lower_bound and upper_bound
    # If length is even, add two copies of median; if n is odd, add one copy.
    true_median = statistics.median(x)
    if len(x) % 2 == 0:
        z = np.concatenate([x, [true_median, true_median, lower_bound, upper_bound]])
    if len(x) % 2 == 1:
        z = np.concatenate([x, [true_median, lower_bound, upper_bound]])
    # NB: Now length is even.
    
    z.sort()
    n = len(z) 

    # Now we clip the data to the interval specified by upper_bound and lower_bound, and we put a buffer of width equal to 'granularity' around the median in the exponential mechanism. We do this by moving all points granularity away from the true median.
    for i in range(math.floor(n/2)+1): #i ranges over the first half of the array indices 
        z[i] = max(lower_bound , z[i] - granularity)
        z[n-i-1] = min(z[n-i-1] + granularity, upper_bound)

    # Iterate through z, assigning scores to each interval given by adjacent indices
    # currentMax and currentInt keep track of highest score and corresponding interval
    currentMax = -np.inf
    currentInt = -1
    for i in range(1, n):
        start = z[i-1]
        end = z[i]
        # Compute length of interval on logarithmic scale
        length = end-start
        if (length <= 0):
            loglength = -np.inf
        else:
            loglength = math.log(length)
        # The rungheight is the score of each individual point in the interval
        rungheight =  abs(i - n/2) # simplified from math.ceil(abs((i-1/2)-(n-1)/2))
        # The score has two components:
        # (1) Distance from index to median (closer -> higher score)
        # (2) Length of the interval on a logarithmic scale (larger -> higher score)
        # We include this since all values in the interval have the same score.
        score = -(epsilon/2) * rungheight + loglength         
                # Add Gumbel noise to sample from exponential mechanism 
                # See https://pdfs.semanticscholar.org/2782/e47c5b0c8a2ce14eae8713b6f2db864f07c8.pdf
        # Add noise scaled to global sensitivity using exponential mechanism 
        noisyscore = score + np.random.gumbel(loc=0.0, scale=1.0)
        if (noisyscore > currentMax): #This should be always satisfied when i=1. 
            currentInt = i
            currentMax = noisyscore
    # Select uniformly from the highest scoring interval given by currentInt
    return np.random.uniform(low=z[currentInt-1], high=z[currentInt])

def median_extension(x, Delta, center = 0.0):
    """
    :param x: numpy array of real numbers. Not necessarily sorted.
    :param Delta: Lipschitz constant to enforce
    :param center: value of the median of an empty list.
    :return: value of Cummings-Durfee median Lipschitz extension
    """
    if len(x) == 0:
        return center
    already_sorted = all(x[i] <= x[i+1] for i in range(len(x)-1))
    if not(already_sorted):
        # Make a copy and sort it so as not to mess up input
        x = x.copy()
        x.sort()
        
    def _recursive_median_ext(i,j):
        # Finds the median extension for x[i:j]
        # print(i, j)
        if j <= i:
            # base case -- median of empty set
            return center
        else:
            m1 = int(np.floor((i + j - 1)/2))
            m2 = int(np.ceil((i + j - 1)/2))
            true_median = ( x[m1] + x[m2]) /2
            if true_median > center: 
                return min(true_median, 
                           _recursive_median_ext(i, j-1) + Delta)
            else:
                return max(true_median,
                          _recursive_median_ext(i+1, j) - Delta)        
        
    return _recursive_median_ext(0, len(x))

            
def local_sens_median(x):
    """
    :param x: numpy array of real numbers. Not necessarily sorted.
    :returns: local sensitivity of the median under insertion/removal, evaluated at x
    """
    if len(x) <= 1:
        return np.inf
    already_sorted = all(x[i] <= x[i+1] for i in range(len(x)-1))
    if not(already_sorted):
        # Make a copy and sort it so as not to mess up input
        x = x.copy()
        x.sort()
        
    if len(x) % 2 ==0: # len(x) is even
        m = int(len(x)/2)
        return (x[m] - x[m-1])/2
    else:
        m = int((len(x) - 1)/2)
        return max(x[m+1] - x[m], x[m] - x[m-1]) / 2

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

def smooth_sens_median(x, beta, lower_bound, upper_bound):
    """
    :param x: numpy array of real numbers. Not necessarily sorted.
    :param beta: Smoothing paramter (positive float)
    :param lower_bound: lower bound for data values in x
    :param upper_bound: upper bound for data values in x
    :returns: beta-smooth sensitivity of the median under insertion/removal, evaluated at x
    """
    x = np.clip(x, lower_bound, upper_bound)
    # Now bookend x with lower and upper bounds for the data
    x = np.concatenate([[lower_bound], x, [upper_bound]])
    x.sort()
    n = len(x)
    m = int(np.floor(len(x)/2))


    # These values store global state for the recursive alg
    j_star_list = np.full(m+1, -np.inf, dtype=np.dtype(int))
    max_for_i_list = np.full(m+1, -np.inf)
    
    # Now we define the core recursive function
    def _J_list(a,c,L,U):
        # Computes j*(a),...,j*(c-1)
        # where j*(i) = argmax (x[j]-x[i]) exp(-beta(j-i-1))
        # and j ranges from ceil(n/2) to n.
        # Stores results in j_star_list and max_for_i_list
        # Invariant: for all i s.t. a <= i < c
        #            we have L <= j*(i) < U
        # (In particular, L < U.)
        if c <= a:
            return #Nothing to do
        else:
            b = int(np.floor((a+c)/2))
            # Now compute j*(b) by searching in {L,...,U-1}
            # max_so_far = -np.inf
            # for j in range(max(L, b+1) , U):
            #     this_value = (x[j] - x[b]) \
            #                 * np.exp(-beta * (j - b -1)) / 2
            #     if this_value > max_so_far:
            #         max_so_far = this_value
            #         best_j_so_far = j
            # j_star_list[b] = best_j_so_far
            # max_for_i_list[b] = max_so_far
            
            #Alternate, vectorized computation of j*(b) 
            L_prime = max(L, b+1)
            gaps = x[L_prime:U] - x[b] # so gaps[k]==x[L'+k] - x[b]
            scaling = np.exp(-beta*np.arange(L_prime - b - 1, U - b - 1)) / 2
            # Both len(gaps) == len(scaling) == U - L_prime.
            # Now rescale gaps and take argmax
            scaled_gaps = gaps * scaling
            # Note: Add back in L_prime to get indices right
            j_star_list[b] = L_prime + np.argmax(scaled_gaps)
            max_for_i_list[b] = np.max(scaled_gaps)

            # Now make recursive calls for the two halves of {a...c}
            _J_list(a, b, L, j_star_list[b] + 1)
            _J_list(b+1, c, j_star_list[b], U)
            return 

    _J_list(0, m+1, m, n)
                
    i_star = np.argmax(max_for_i_list)
    j_star = j_star_list[i_star]
    
    return (np.max(max_for_i_list),
            [i_star - 1, j_star - 1],
           j_star_list,
           max_for_i_list)
    
def smooth_sens_median_slow_j_star(x, beta, lower_bound, upper_bound):
    """
    Quadratic-time implementation of smooth sensitivity.
    Also computes j_star for all inputs
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
    global_max = -np.inf
    j_star_list = np.full(m+1, -np.inf, dtype=np.dtype(int))
    max_for_i_list = np.full(m+1, -np.inf)
    for i in range(m+1):
        for j in range(max(m, i+1), n):
            this_value = (x[j] - x[i]) * np.exp(-beta * (j - i -1)) / 2
            if this_value > max_for_i_list[i]: #update j*(i)
                j_star_list[i] = j
                max_for_i_list[i] = this_value
                if this_value > global_max:
                    global_max = this_value
                    other_best = [i-1, j-1]
    
    i_star = np.argmax(max_for_i_list)
    j_star = j_star_list[i_star]
    best_pair = [i_star -1, j_star-1]
    return global_max, best_pair, other_best, j_star_list, max_for_i_list 

