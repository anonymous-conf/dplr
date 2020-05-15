import numpy as np
import DPGradDescent as DPGD
import DPTS_algorithms as DPTS
import NoisyStats as NS
import math

'''
Example use of methods for differentially private simple linear regression.
'''

def main():
    eps = 1.0 # privacy parameter
    n = 1000 # no. of (x,y) pairs to generate
    varx = 0.05 # variance of x
    barx = 0.5 # mean of x
    vare = 0.005 # conditioned on x, variance of y
    slope = 0.5 # true slope of (x, y)
    intercept = 0.2 # true intercept of (x, y)

    # generate data
    x = []
    y = []
    for i in range(n):
        x.append(np.random.normal(barx, math.sqrt(varx)))
        y.append(slope*x[i] + intercept + np.random.normal(0, math.sqrt(vare)))
    x = np.array(x)
    y = np.array(y)
    xm = np.mean(x)
    ym = np.mean(y)
    xnew = [0.25, 0.75] # goal is to compute predictions (y value) at x = 0.25 and x = 0.75

    # predictions at 0.25, 0.75 are:
    # 0.25 * 0.5 + 0.2 = 0.325
    # 0.75 * 0.5 + 0.2 = 0.575
    num_trials = 10
    p25results = []
    p75results = []
    for i in range(num_trials):
        p25result = []
        p75result = []
        for method in [DPGD.DPGradDescent,
                       DPTS.dpMedTS_exp,
                       DPTS.dpMedTS_exp_wide,
                       DPTS.dpMedTS_ss_ST_no_split,
                       NS.NoisyStats]:
            p25, p75 = method(x, y, xm, ym, n, eps, xnew)
            if p25 is not None: p25result.append(p25)
            if p75 is not None: p75result.append(p75)
        p25results.append(p25result)
        p75results.append(p75result)
    p25results = np.array(p25results)
    p75results = np.array(p75results)

    print("True predictions at 0.25, 0.75: 0.325, 0.575")
    print("DPGradDescent average prediction: ", np.mean(p25results[:,0]), np.mean(p75results[:,0]))
    print("DPExpTheilSen average prediction: ", np.mean(p25results[:,1]), np.mean(p75results[:,1]))
    print("DPWideTheilSen average prediction: ", np.mean(p25results[:,2]), np.mean(p75results[:,2]))
    print("DPSSTheilSen average prediction: ", np.mean(p25results[:,3]), np.mean(p75results[:,3]))
    print("NoisyStats average prediction: ", np.mean(p25results[:,4]), np.mean(p75results[:,4]))

if __name__ == "__main__":
    main()
