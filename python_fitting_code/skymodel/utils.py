# Analytic Sampling of Sky Models
# Authors: [removed for review purposes]
# This file contains utilities for the fitting process

import sys

# Numpy
import numpy as np

# Curve fitting with scipy
from scipy.optimize import curve_fit
from scipy import integrate
from scipy import special

from skopt import gp_minimize

import matplotlib.pyplot as plt
from matplotlib import cm


# Our function to fit is going to be a sum of two-dimensional Gaussians
# Normalized 2D Gaussian with weight for GMM
def gaussian(x, y, x0, y0, xalpha, yalpha, A):
    return A * (1.0 / (2.0 * np.pi * xalpha * yalpha)) * np.exp( -0.5 * (((x-x0)/xalpha)**2 + ((y-y0)/yalpha)**2))

# Truncated Normalized 2D Gaussian with weight for GMM
# Can be used to fit a *truncated" gaussian
def gaussian_truncated(x, y, x0, y0, xalpha, yalpha, A, lbx=-np.inf, ubx=np.inf, lby=-np.inf, uby=np.inf):
    return (A * (1.0 / (2.0 * np.pi * xalpha * yalpha)) * np.exp( -0.5 * (((x-x0)/xalpha)**2 + ((y-y0)/yalpha)**2))) / ((approx_cdf(ubx, x0, xalpha) - approx_cdf(lbx, x0, xalpha)) * (approx_cdf(uby, y0, yalpha) - approx_cdf(lby, y0, yalpha)))

# Currenlty we are fitting 2D Gaussians so this is difficult to change
MODEL_PARAM_COUNT = 5

# Cost function (default)
# This is the callable that is passed to curve_fit. M is a (2,N) array
# where N is the total number of data points in Z, which will be ravelled
# to one dimension.
def gaussian_cost(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    for i in range(len(args)//MODEL_PARAM_COUNT):
       arr += gaussian(x, y, *args[i*MODEL_PARAM_COUNT:i*MODEL_PARAM_COUNT + MODEL_PARAM_COUNT])
    return arr

# Cost function (weights must sum to 1)
def gaussian_penalized_cost(M, *args):
    sum = 0.0
    for i in range(len(args)//MODEL_PARAM_COUNT):
       sum += args[i*MODEL_PARAM_COUNT + 4]

    penalization = abs(1.-sum)*10000
    x, y = M
    arr = np.zeros(x.shape)
    for i in range(len(args)//MODEL_PARAM_COUNT):
       arr += gaussian(x, y, *args[i*MODEL_PARAM_COUNT:i*MODEL_PARAM_COUNT + MODEL_PARAM_COUNT]) + penalization
    return arr

# Lightweight abstraction over a 2D gaussian
class Gaussian2D:
    def __init__(self, meanx, meany, sigmax, sigmay, weight):
        self.meanx  = meanx
        self.meany  = meany
        self.sigmax = sigmax
        self.sigmay = sigmay
        self.weight = weight

    def __str__(self):
        return "[Gaussian2D] Weight : " + '{:.10f}'.format(self.weight)  + " Mean X : " + '{:.10f}'.format(self.meanx) + " Mean Y : " + '{:.10f}'.format(self.meany) + " Sigma X : " + '{:.10f}'.format(self.sigmax) + " Sigma Y : " + '{:.10f}'.format(self.sigmay)

    def __repr__(self):
        return "[Gaussian2D] Weight : " + '{:.3f}'.format(self.weight) + " Mean X : " + '{:.10f}'.format(self.meanx) + " Mean Y : " + '{:.10f}'.format(self.meany) + " Sigma X : " + '{:.10f}'.format(self.sigmax) + " Sigma Y : " + '{:.10f}'.format(self.sigmay)

    def eval_n(self):
        N = 400
        X = np.linspace(-4.0 * np.pi, 4.0 * np.pi, 256)
        Y = np.linspace(-4.0 * np.pi, 4.0 * np.pi, 64)
        X, Y = np.meshgrid(X, Y)
        return gaussian(X, Y, self.meanx, self.meany, self.sigmax, self.sigmay, self.weight)

    def eval(self, x, y):
        return gaussian(x, y, self.meanx, self.meany, self.sigmax, self.sigmay, self.weight)

    def plot(self, factor, centered=False):
        N = 400
        if centered:
          X = np.linspace(self.meanx - 4 * self.sigmax, self.meanx + 4 * self.sigmax, N)
          Y = np.linspace(self.meany - 4 * self.sigmay, self.meany + 4 * self.sigmay, N)
        else:
          X = np.linspace(-4 * self.sigmax, 4 * self.sigmax, N)
          Y = np.linspace(-4 * self.sigmay, 4 * self.sigmay, N)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros(X.shape)
        Z = gaussian(X, Y, self.meanx, self.meany, self.sigmax, self.sigmay, self.weight)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z, cmap='plasma')
        #ax.plot_surface(X, Y, Z, alpha=0.2)
        ax.set_zlim(0, factor)
        if centered:
          ax.set_ylim(self.meany - 4 * self.sigmay, self.meany + 4 * self.sigmay)
          ax.set_xlim(self.meanx - 4 * self.sigmax, self.meanx + 4 * self.sigmax)
        else:
          ax.set_ylim(-4 * self.sigmay, 4 * self.sigmay)
          ax.set_xlim(-4 * self.sigmax, 4 * self.sigmax)
          
        print('MAX ', gaussian(self.meanx, self.meany, self.meanx, self.meany, self.sigmax, self.sigmay, self.weight))

        ax.set_zlim(0, gaussian(self.meanx, self.meany, self.meanx, self.meany, self.sigmax, self.sigmay, self.weight))
        #ax.set_ylim(0, 1.0 * np.pi)
        #ax.set_xlim(0, 4.0 * np.pi)
        plt.show()

    def sample(self, factor, centered=False, bounds=[]):
        x = gauss_sample(1, self.meanx, self.sigmax, 0, 2.0 * np.pi)
        #y = gauss_sample_sin_weighted(1, self.meany, self.sigmay, 0, 0.5 * np.pi)
        y = gauss_sample(1, self.meany, self.sigmay, 0, 0.5 * np.pi)

        return x, y

    def plot_hot(self, factor, centered=False, bounds=[]):
        N = 200
        xmin = -4 * self.sigmax
        xmax =  4 * self.sigmax
        ymin = -4 * self.sigmay
        ymax =  4 * self.sigmay
        if centered:
            xmin = self.meanx - 4 * self.sigmax
            xmax = self.meanx + 4 * self.sigmax
            ymin = self.meany - 4 * self.sigmay
            ymax = self.meany + 4 * self.sigmay

        if len(bounds) > 0:
            xmin = bounds[0]
            xmax = bounds[1]
            ymin = bounds[2]
            ymax = bounds[3]
      
        #xmin = np.minimum(xmin, ymin)
        #ymin = np.minimum(xmin, ymin)
        #xmax = np.maximum(xmax, ymax)
        #ymax = np.maximum(xmax, ymax)

        X = np.linspace(xmin, xmax, N)
        Y = np.linspace(ymin, ymax, N)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros(X.shape)
        Z = gaussian(X, Y, self.meanx, self.meany, self.sigmax, self.sigmay, self.weight)

        fig = plt.figure()
        #ax = fig.gca(projection='3d')
        ax = fig.gca()

        ax.contourf(X, Y, Z, zdir='z', offset=-10.15, cmap=cm.viridis)
        ax.axhline(y=0.5 * np.pi, color='r', linestyle='-')
        ax.axhline(y=0, color='r', linestyle='-')
        #ax.plot_surface(X, Y, Z, cmap='plasma')
        #ax.plot_surface(X, Y, Z, alpha=0.2)
        #ax.set_zlim(0, factor)
        
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
           
        print('MAX ', gaussian(self.meanx, self.meany, self.meanx, self.meany, self.sigmax, self.sigmay, self.weight))

        x = np.random.uniform(xmin, xmax, N)
        y = np.random.uniform(ymin, ymax, N)
        #ax.scatter(x, y, s=4, color=(0,0,0))
        # Plot
        x = np.random.normal(self.meanx, self.sigmax, N)
        y = np.random.normal(self.meany, self.sigmay, N)
        #ax.scatter(x, y, s=4, color=(0,0,0))
        #x, y = BoxMuller(self.meanx, self.sigmax, self.meany, self.sigmay, N)
        #y = BoxMuller(self.meanx, self.sigmax, N)
        x = gauss_sample(N, self.meanx, self.sigmax, 0, 2.0 * np.pi)
        y = gauss_sample(N, self.meany, self.sigmay, 0, 0.5 * np.pi)
        ax.scatter(x, y, s=4, color=(0,0,0))
        #ax.set_zlim(0, gaussian(self.meanx, self.meany, self.meanx, self.meany, self.sigmax, self.sigmay, self.weight))
        #ax.set_ylim(0, 1.0 * np.pi)
        #ax.set_xlim(0, 4.0 * np.pi)
        plt.show()

    def volume(self):
        return 2.0 * np.pi * (1.0 / (2.0 * np.pi * self.sigmax * self.sigmay)) * self.sigmax * self.sigmay

    def volume2(self, lbx=-np.inf, ubx=np.inf, lby=-np.inf, uby=np.inf):
        return integrate.dblquad(lambda x, y: gaussian(x, y, self.meanx, self.meany, self.sigmax, self.sigmay, 1), lby, uby, lambda u: lbx, lambda v: ubx)

    def volume_approx(self, lbx=-np.inf, ubx=np.inf, lby=-np.inf, uby=np.inf):
        return (approx_cdf(ubx, self.meanx, self.sigmax) - approx_cdf(lbx, self.meanx, self.sigmax)) * (approx_cdf(uby, self.meany, self.sigmay) - approx_cdf(lby, self.meany, self.sigmay))

    def cdf(self, X, Y, lbx=-np.inf, ubx=np.inf, lby=-np.inf, uby=np.inf):
        return approx_cdf(X, self.meanx, self.sigmax) * approx_cdf(Y, self.meany, self.sigmay)

    def truncated_cdf(self, X, Y, lbx=-np.inf, ubx=np.inf, lby=-np.inf, uby=np.inf):
        Z = self.volume_approx(lbx, ubx, lby, uby)
        Fx = approx_cdf(X, self.meanx, self.sigmax) - approx_cdf(lbx, self.meanx, self.sigmax)
        Fy = approx_cdf(Y, self.meany, self.sigmay) - approx_cdf(lby, self.meany, self.sigmay)
        return (Fx * Fy) / Z

def approx_cdf(x, mu=0, sigma=1):
    return 0.5 * (1.0 + approx_erf2((x - mu) / (sigma * np.sqrt(2))))

def approx_cdf_tanh(x):
    return 0.5 + 0.5 * np.tanh((np.pi * x) / (2.0 * np.sqrt(3)))

def approx_cdf_scipy(x):
    return 0.5 * (1.0 + special.erf((x) / (np.sqrt(2))))

def approx_quantile_tanh(x):
    return - (2.0 * np.sqrt(3) * np.arctanh(1.0 - 2.0 * x)) / (np.pi)

def approx_quantile_scipy(x):
    return np.sqrt(2) * special.erfinv(2.0 * x - 1.0)

# https://ckrao.wordpress.com/2014/09/15/a-good-simple-approximation-of-the-complementary-error-function/
def approx_erfc_positive(x):
    c1 = 1.09500814703333
    c2 = 0.75651138383854
    return np.exp( -(c1 * x + c2 * x * x) )

def approx_erfc_negative(x):
    c1 = 1.09500814703333
    c2 = 0.75651138383854
    return 2.0 - np.exp( c1 * x - c2 * x * x )


def approx_erfc(x): 
    return np.where(x < 0, approx_erfc_negative(x), approx_erfc_positive(x))

def approx_erf(x): 
    return 1.0 - approx_erfc(x)

# http://people.math.sfu.ca/~cbm/aands/page_299.htm
def approx_erf2(x):
    p = 0.47047
    a1 = 0.34802
    a2 = -0.09587
    a3 = 0.74785
    t = 1.0 / ( 1.0 + p * x * np.sign(x) )
    tt = t * t
    ttt = tt * t
    xx = x * x
    return np.sign(x) * ( 1.0 - ( a1 * t + a2 * tt + a3 * ttt) * np.exp(-xx) )

def ncdf(x, mu=0, sigma=1):
    return 0.5 * (1.0 + approx_erf((x - mu) / (sigma * np.sqrt(2))))

def leftb(w):
    w = w - 2.5
    p = 2.81022636e-08
    p = 3.43273939e-07 + p * w
    p = -3.5233877e-06 + p * w
    p = -4.39150654e-06 + p * w
    p = 0.00021858087 + p * w
    p = -0.00125372503 + p * w
    p = -0.00417768164 + p * w
    p = 0.246640727 + p * w
    p = 1.50140941 + p * w

    return p

def rightb(w):
    w = np.sqrt(w) - 3.0
    p = -0.000200214257
    p = 0.000100950558 + p * w
    p = 0.00134934322 + p * w
    p = -0.00367342844 + p * w
    p = 0.00573950773 + p * w
    p = -0.0076224613 + p * w
    p = 0.00943887047 + p * w
    p = 1.00167406 + p * w
    p = 2.83297682 + p * w    

    return p

def MGilesInverseError(x):
    w = -np.log((1.0 - x)*(1.0 + x));
 
    p = np.where(w < 0.5, leftb(w), rightb(w))

    return p * x

def approx_quantile_leftb(p):
    # Coefficients in rational approximations
    a = np.array([ -39.696830, 220.946098, -275.928510, 138.357751, -30.664798, 2.506628 ])
    
    b = np.array([ -54.476098, 161.585836, -155.698979, 66.801311, -13.280681 ])
    
    c = np.array([ -0.007784894002, -0.32239645, -2.400758, -2.549732, 4.374664, 2.938163 ])
    
    d = np.array([ 0.007784695709, 0.32246712, 2.445134, 3.754408 ])
    
    q = np.sqrt(-2 * np.log(p))
    return ( ( ( ( ( c[ 0 ] * q + c[ 1 ] ) * q + c[ 2 ] ) * q + c[ 3 ] ) * q + c[ 4 ] ) * q + c[ 5 ] ) / ( ( ( ( d[ 0 ] * q + d[ 1 ] ) * q + d[ 2 ] ) * q + d[ 3 ] ) * q + 1 )

def approx_quantile_rightb(p):
    # Coefficients in rational approximations
    a = np.array([ -39.696830, 220.946098, -275.928510, 138.357751, -30.664798, 2.506628 ])
    
    b = np.array([ -54.476098, 161.585836, -155.698979, 66.801311, -13.280681 ])
    
    c = np.array([ -0.007784894002, -0.32239645, -2.400758, -2.549732, 4.374664, 2.938163 ])
    
    d = np.array([ 0.007784695709, 0.32246712, 2.445134, 3.754408 ])
    
    q = p - 0.5;
    r = q * q;
    return ( ( ( ( ( a[ 0 ] * r + a[ 1 ] ) * r + a[ 2 ] ) * r + a[ 3 ] ) * r + a[ 4 ] ) * r + a[ 5 ] ) * q / ( ( ( ( ( b[ 0 ] * r + b[ 1 ] ) * r + b[ 2 ] ) * r + b[ 3 ] ) * r + b[ 4 ] ) * r + 1 );
    

def approx_quantile(p, mu=0, sigma=1):
    plow = 0.02425
    phigh = 1.0 - plow
    
    return mu + sigma * np.where(p < plow, approx_quantile_leftb(p), approx_quantile_rightb(p))

def gauss_sample_sin_weighted(N, mu=0, sigma=1, low=-5, high=5):
    # Define break-points.
    Fa = approx_cdf(low, mu, sigma)
    Fb = approx_cdf(high, mu, sigma)

    x = np.arcsin(np.random.uniform(size=N)) / (np.pi * 0.5)
    d = np.zeros(N)

    for i in range(N):
        v = Fa + (Fb - Fa) * x[i]
        d[i] = approx_quantile(v, mu, sigma)

    return d


def gauss_sample(N, mu=0, sigma=1, low=-5, high=5):
    # Define break-points.
    Fa = approx_cdf(low, mu, sigma)
    Fb = approx_cdf(high, mu, sigma)

    x = np.random.uniform(size=N)
    d = np.zeros(N)

    for i in range(N):
        v = Fa + (Fb - Fa) * x[i]
        d[i] = approx_quantile(v, mu, sigma)

    return d

def sample(N):
    # Define break-points.
    Fa = approx_cdf(-1.5)
    Fb = approx_cdf(1.5)

    x = np.random.uniform(size=N)
    d = np.zeros(N)

    for i in range(N):
        v = Fa + (Fb - Fa) * x[i]
        d[i] = approx_quantile(v)

    return d

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def BoxMuller(mux, sigmax, muy, sigmay, N):
    epsilon = 0.000001
    two_pi = 2.0*np.pi
    N=500
    low=0
    high=np.exp(-(1) / 2.0)

    #rands = np.random.uniform(low=low, high=high, size=N)
    rands = np.random.uniform(0.0, 0.65, size=N)
    #rands = np.linspace(0, 1, 20)
    randstheta = np.random.uniform(low=0, high=(np.pi * 0.5) / (np.pi * 2.0), size=N)
    #randstheta = np.full((N), 0.60)
    out = np.zeros(N)
    out2 = np.zeros(N)

    for i in range(N):
        u1 = rands[i]
        #u1 = 0.60
        u2 = randstheta[i]

        out[i] = np.sqrt(-2.0 * np.log(u1)) * np.cos(two_pi * u2) * sigmax + mux
        out2[i] = np.sqrt(-2.0 * np.log(u1)) * np.sin(two_pi * u2) * sigmay + muy

    #print(out)
    #print(out2)

    return out, out2


def fit_gmm_cost(norm_factor, skymap_image, ncomponents = 5, init_params=[], func=gaussian, cost=gaussian_cost):
    ncols, nrows = skymap_image.size
    skymap_array = np.array(skymap_image.getdata()).reshape((nrows, ncols))
    factor = np.max(skymap_array) * norm_factor[0]
    skymap_array /= factor

    xmin, xmax, nx = 0.0, np.pi * 2.0, ncols
    ymin, ymax, ny = 0.0, np.pi / 2.0, nrows
    x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    # Flatten the initial guess parameter list.
    if (len(init_params) > 0):
        guess_params = [param for component_index in range(ncomponents) for param in init_params]
    else:
        guess_params = [param for component_index in range(ncomponents) for param in (1, 1, 0.5, 0.5, 1.0 / ncomponents)]
  
    lower_bounds = tuple([bound for bounds in range(0,ncomponents) for bound in (0, 0, 0, 0, 0)])
    upper_bounds = tuple([bound for bounds in range(0,ncomponents) for bound in (2.0 * np.pi, 0.5 * np.pi, 6, 6, 1)])
    bounds = (lower_bounds, upper_bounds)

    # We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
    xdata = np.vstack((X.ravel(), Y.ravel()))

    # Do the fit, using our custom 'cost' function which understands our
    # flattened (ravelled) ordering of the data points.
    try:
        popt, pcov = curve_fit(cost, xdata, skymap_array.ravel(), guess_params, method='trf', bounds=bounds)
    except RuntimeError:
        return 1000000.0
    
    # Reconstruct data using fitted parameters
    skymap_fit = np.zeros(skymap_array.shape)
    for i in range(ncomponents):
        skymap_fit += func(X, Y, *popt[i*MODEL_PARAM_COUNT:i*MODEL_PARAM_COUNT+MODEL_PARAM_COUNT])
    
    # [Vorba14]
    factored_err = np.sum((1.0 - (skymap_fit / skymap_array))**2)
    # RMSE
    factored_rmse = np.sqrt(np.mean(factor * skymap_array - factor * skymap_fit)**2)
    # MAE
    factored_mae = np.mean(np.absolute(factor * skymap_array - factor * skymap_fit))
    factored_max = np.max(np.absolute(factor * skymap_array - factor * skymap_fit))

    return factored_mae

def fit_gmm(norm_factor, skymap_image, ncomponents = 5, init_params=[], func=gaussian, cost=gaussian_cost):
    ncols, nrows = skymap_image.size
    skymap_array = np.array(skymap_image.getdata()).reshape((nrows, ncols))
    factor = np.max(skymap_array) * norm_factor[0]
    skymap_array /= factor

    xmin, xmax, nx = 0.0, np.pi * 2.0, ncols
    ymin, ymax, ny = 0.0, np.pi / 2.0, nrows
    x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    # Flatten the initial guess parameter list.
    if (len(init_params) > 0):
        guess_params = [param for component_index in range(ncomponents) for param in init_params]
    else:
        guess_params = [param for component_index in range(ncomponents) for param in (1, 1, 0.5, 0.5, 1.0 / ncomponents)]
  
    lower_bounds = tuple([bound for bounds in range(0,ncomponents) for bound in (0, 0, 0, 0, 0)])
    upper_bounds = tuple([bound for bounds in range(0,ncomponents) for bound in (2.0 * np.pi, 0.5 * np.pi, 6, 6, 1)])
    bounds = (lower_bounds, upper_bounds)

    # We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
    xdata = np.vstack((X.ravel(), Y.ravel()))

    # Do the fit, using our custom 'cost' function which understands our
    # flattened (ravelled) ordering of the data points.
    try:
        popt, pcov = curve_fit(cost, xdata, skymap_array.ravel(), guess_params, method='trf', bounds=bounds)
    except RuntimeError:
        return 1000000.0

    # Reconstruct data using fitted parameters
    skymap_fit = np.zeros(skymap_array.shape)
    for i in range(ncomponents):
        skymap_fit += func(X, Y, *popt[i*MODEL_PARAM_COUNT:i*MODEL_PARAM_COUNT+MODEL_PARAM_COUNT])

    # [Vorba14]
    factored_err = np.sum((1.0 - (skymap_fit / skymap_array))**2)
    # RMSE
    factored_rmse = np.sqrt(np.mean(factor * skymap_array - factor * skymap_fit)**2)
    # MAE
    factored_mae = np.mean(np.absolute(factor * skymap_array - factor * skymap_fit))
    factored_max = np.max(np.absolute(factor * skymap_array - factor * skymap_fit))

    gaussians = []
    for i in range(ncomponents):
        gaussians.append(Gaussian2D(*popt[i*MODEL_PARAM_COUNT:i*MODEL_PARAM_COUNT+MODEL_PARAM_COUNT]))

    return gaussians, factored_mae, factored_rmse, factored_max, factored_err

def fit_gmm_bo(skymap_image, ncomponents = 5, init_params=[], func=gaussian, cost=gaussian_cost):
    def f(x):
        return fit_gmm_cost(x, skymap_image, ncomponents, init_params, func, cost)

    res = gp_minimize(f, [(0.1, 20.0)], n_calls=10)
    print(res.x[0], res.fun)

    return fit_gmm(res.x, skymap_image, ncomponents, init_params, func, cost)

def fit_gmm_linear(skymap_image, ncomponents = 5, init_params=[], func=gaussian, cost=gaussian_cost):
    ncols, nrows = skymap_image.size

    #Image is column major so we transpose it

    # Plotting actual PDF
    boundsx = [0, 2.0 * np.pi]
    boundsy = [0, np.pi / 2.0]
    xmin, xmax, nx = boundsx[0], boundsx[1], ncols
    ymin, ymax, ny = boundsy[0], boundsy[1], nrows
    x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    # Flatten the initial guess parameter list.
    guess_params = [param for component_index in range(ncomponents) for param in (1, 1, 0.5, 0.5, 1.0 / ncomponents)]
    if len(init_params) > 0:
        guess_params = [param for component_index in range(ncomponents) for param in (1, 1, 0.5, 0.5, 1.0 / ncomponents)]

    lower_bounds = tuple([bound for bounds in range(0,ncomponents) for bound in (0, 0, 0, 0, 0)])
    upper_bounds = tuple([bound for bounds in range(0,ncomponents) for bound in (2.0 * np.pi, 0.5 * np.pi, 6, 6, 1)])
    bounds = (lower_bounds, upper_bounds)

    # We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
    xdata = np.vstack((X.ravel(), Y.ravel()))
    optimal_params = []
    min_mae = 1000000.0
    min_rmse = 1000000.0
    min_factor = -1
    min_max = 100000.0
    min_error = 100000.0
    range_upper = 1.5
    range_lower = 1
    range_step  = 1
    range_intervals = int((range_upper - range_lower) / range_step) 
    index = 0
    for norm_factor in np.arange(range_lower, range_upper, range_step):
        index = index + 1
        print("Fitting", end='\r', flush=True)
        sys.stdout.write("Fitting %d of %d (%.2f%%)\r" %(index, range_intervals+1, 100* index/(range_intervals+1)))
        sys.stdout.flush()

        skymap_array = np.array(skymap_image.getdata()).reshape((nrows, ncols))
        factor = np.max(skymap_array) * norm_factor
        skymap_array /= factor

        # Do the fit, using our custom _gaussian function which understands our
        # flattened (ravelled) ordering of the data points.
        try:
            popt, pcov = curve_fit(cost, xdata, skymap_array.ravel(), guess_params, method='trf', bounds=bounds)
        except RuntimeError:
            continue
    
        # Reconstruct data using fitted parameters
        skymap_fit = np.zeros(skymap_array.shape)
        for i in range(ncomponents):
            skymap_fit += func(X, Y, *popt[i*MODEL_PARAM_COUNT:i*MODEL_PARAM_COUNT+MODEL_PARAM_COUNT])
    
        factored_err = np.sum((1.0 - (skymap_fit / skymap_array))**2)
        factored_rmse = np.sqrt(np.mean((factor * skymap_array - factor * skymap_fit)**2))
        factored_mae = np.mean(np.absolute(factor * skymap_array - factor * skymap_fit))
        factored_mbe = np.mean(factor * skymap_array - factor * skymap_fit)
        factored_max = np.max(np.absolute(factor * skymap_array - factor * skymap_fit))

        #print('Sampling error ', factored_err)
        #print('Sampling rmse ', factored_rmse)
        #print('Sampling mae ', factored_mae)
        
        if factored_mae < min_mae:
        #if factored_max < min_max :
        #if factored_err < min_error :
            min_mae = factored_mae
            min_error = factored_err
            min_factor = norm_factor
            optimal_params = popt
            min_rmse = factored_rmse
            min_max = factored_max

    gaussians = []
    for i in range(ncomponents):
        gaussians.append(Gaussian2D(*optimal_params[i*MODEL_PARAM_COUNT:i*MODEL_PARAM_COUNT+MODEL_PARAM_COUNT]))

    return gaussians, min_mae, min_rmse, min_max, min_factor
    
def gen_sample(gaussians, lbx=-np.inf, ubx=np.inf, lby=-np.inf, uby=np.inf):
    x = 0.0
    y = 0.0
    FaX = 0.0
    FbX = 0.0
    FaY = 0.0
    FbY = 0.0
    
    for gauss in gaussians:
        FaX += approx_cdf(lbx, gauss.meanx, gauss.sigmax)
        FbX += approx_cdf(ubx, gauss.meanx, gauss.sigmax)
        
        FaY += approx_cdf(lby, gauss.meany, gauss.sigmay)
        FbY += approx_cdf(uby, gauss.meany, gauss.sigmay)

    r = np.random.uniform()
    v = FaX + (FbX - FaX) * r
    x = approx_quantile(v, gauss.meanx, gauss.sigmax)
    
    r = np.random.uniform()
    v = FaY + (FbY - FaY) * r
    y = approx_quantile(v, gauss.meany, gauss.sigmay)
    
    return x, y

