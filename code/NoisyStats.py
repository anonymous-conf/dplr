import numpy as np

### Add documentation
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
