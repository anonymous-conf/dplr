import numpy as np

'''
DPGradDescent:
an eps^2/2-zCDP algorithm that computes noisy gradients
(of the OLS convex optimization equation) for a certain number of iterations.

x, y: numpy arrays containing data
xm, ym: means of x and y respectively
eps: privacy parameter
xnew: target x value
      (e.g. xnew = [0.25, 0.75] for point estimates at x = 0.25, 0.75)
'''
def DPGradDescent(x, y, xm, ym, n, eps, xnew):
    x = np.array(x)
    y = np.array(y)
    assert(len(x) == len(y))

    my_rho = eps**2 / 2 #Convert xnew to rho value for zCDP

    results = adaptive_NGD(x, y, rho = my_rho, T=80, clip_range=1, clip_type = 'square')
    # results are predictions at 0.25 and 0.75
    # Convert these to predictions at xnew
    # yhat = 2*f1*(3/4 - x) + 2*f2*(x - 1/4)

    xnew = np.array(xnew)
    predictions = 2*results[0]*(3/4 - xnew) + 2*results[1]*(xnew - 1/4)
    return predictions

'''
Returns the non-noisy clipped gradient of the OLS loss.
x: array of x values in [0,1]
y: array of y values in [0,1]
clip_range: individual contributions to the gradient
are clipped to a box [-a,a]^2 where a = clip_range
p25: current prediction at 0.25
p75: current prediction at 0.75
'''
def grad(x, y, clip_range, p25, p75, clip_type='square'):
    assert(len(x) == len(y))

    yhat = 2 * (p25 * (3/4 - x) + p75 *  (x-1/4))
    gradients = 2 * np.column_stack((yhat - y, yhat - y)) * np.column_stack((3/4 - x, x - 1/4))
    if (clip_type == 'square'):
        # Clip to Linfty ball of radius clip_range
        clipped_gradients = np.clip(gradients, - clip_range, clip_range)
    elif(clip_type =='ball'):
        # Clip to L2 ball of radius clip_range
        # First get vector of l2_norms
        norms = np.sqrt(np.sum( gradients**2 , axis=1))
        factors = np.minimum ( np.ones(len(norms)), clip_range /norms)
        # Now scale down each row of 'gradients' by corresponding entry of 'factors'.
        clipped_gradients = gradients * (np.column_stack((factors, factors)))

    total_clipped_gradient = np.sum(clipped_gradients, axis =0)

    return total_clipped_gradient

'''
Returns the noisy clipped gradient of the OLS loss.
The result is rho-CDP (think rho=epsilon^2/2)
x: array of x values in [0,1]
    y: array of y values in [0,1]
clip_range: individual contributions to the gradient
are clipped to a box [-a,a]^2 where a = clip_range
p25: current prediction at 0.25
p75: current prediction at 0.75
rho: CDP parameter (for one gradient computation)
clip_type: either 'sqaure' (for linfty ball) or 'ball' (for L2 ball)
'''
def noisyGrad(x, y, clip_range, p25, p75, rho , clip_type='square'):
    assert(len(x) == len(y))

    yhat = 2 * (p25 * (3/4 - x) + p75 *  (x-1/4)) #vector of predicted values
    gradients = 2 * np.column_stack((yhat - y, yhat - y)) * np.column_stack((3/4 - x, x - 1/4))

    # Now clip
    if (clip_type == 'square'):
        # Clip to Linfty ball of radius clip_range
        clipped_gradients = np.clip(gradients, - clip_range, clip_range)
        # Compute l2_sensitivity of clipped gradient
        l2_sensitivity = 2*np.sqrt(2)*clip_range
    elif(clip_type =='ball'):
        # Clip to L2 ball of radius clip_range
        # First get vector of l2_norms
        norms = np.sqrt(np.sum( gradients**2 , axis=1))
        factors = np.minimum ( np.ones(len(norms)), clip_range /norms)
        # Now scale down each row of 'gradients' by corresponding entry of 'factors'.
        clipped_gradients = gradients * (np.column_stack((factors, factors)))
        # Compute l2_sensitivity of clipped gradient
        l2_sensitivity = 2*clip_range

    # Get total gradient by summing over data points
    total_clipped_gradient = np.sum(clipped_gradients, axis =0)

    # Now add noise
    # Recall that rho = (l2_sens^2 / 2 sigma^2).
    # So we set sigma = l2_sens / sqrt(2*rho).
    noisy_total_clipped_gradient = \
                total_clipped_gradient \
                + np.random.normal(loc=0.0, scale = l2_sensitivity / np.sqrt(2*rho), size = 2)

    return noisy_total_clipped_gradient

'''
Noisy GD with Gaussian noise and rho evenly divided.
Shrink eta based on root sum of sqaured norms of gradients so far.
Assums n is public
Returns estimates at 0.25 and 0.75
'''
def adaptive_NGD(x, y, rho,
                 clip_range = None,
                 clip_type = 'square',
                 T = 5,
                 eta = None):
    n = len(x)
    if clip_range == None:
        clip_range = 10
    if eta!=None:
        this_eta = eta
    ests = np.array([0.5, 0.5])
    iterates = np.zeros((T,2))
    sum_gradient_norms = 0
    for t in range(T):
        # Divide budget evenly between iterations
        this_rho = rho/T 
        this_grad = noisyGrad(x, y, clip_range, ests[0], ests[1], this_rho,
                                  clip_type = clip_type)
        # Set eta unless user provided it. 
        if eta==None:
            sum_gradient_norms += np.sum(this_grad**2)
            this_eta = 1/(np.sqrt(sum_gradient_norms))
        ests += - this_eta * this_grad
        iterates[t,:] = ests
    #also average over some window of iterates
    return np.average(iterates[int(np.floor(T/2)):,:], axis=0)

