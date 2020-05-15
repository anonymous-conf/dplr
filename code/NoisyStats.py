import numpy as np

'''
NoisyStats: 
an eps-DP algorithm that perturbs the OLS sufficient statistics for
simple linear regression.

x, y: numpy arrays containing data
xm, ym: means of x and y respectively
eps: privacy parameter
xnew: target x value
      (e.g. xnew = [0.25, 0.75] for point estimates at x = 0.25, 0.75)
'''
def NoisyStats(x, y, xm, ym, n, eps, xnew):
    eps = eps/3.0

    A = np.dot(x-xm, y-ym)
    B = np.dot(x-xm, x-xm)

    Delta = (1-1.0/n)
    l1 = np.random.laplace(0., Delta/eps, 1)
    l2 = np.random.laplace(0., Delta/eps, 1)

    if B + l2 <= 0: return None
    alphapriv = (A + l1)/(B + l2)

    l3 = np.random.laplace(0., 1./(eps*n) * (1 + abs(alphapriv)), 1)
    betapriv = (ym - alphapriv*xm) + l3

    return alphapriv*xnew+betapriv
