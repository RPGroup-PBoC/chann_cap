# -*- coding: utf-8 -*-
"""
Title:
    stasts.py
Last update:
    2019-11-06
Author(s):
    Manuel Razo-Mejia
Purpose:
    This file compiles all of the functions necessary for statistical
    analysis of the data relevant for the channel capacity project
"""

import numpy as np


# USEFUL GENERAL FUNCTIONS
def ecdf(data):
    """
    Computes the empirical cumulative distribution function (ECDF)
    of a given set of 1D data.

    Parameters
    ----------
    data : 1d-array
        Data from which the ECDF will be computed.

    Returns
    -------
    x, y : 1d-arrays
        The sorted data (x) and the ECDF (y) of the data.
    """

    return np.sort(data), np.arange(len(data))/len(data)


def hpd(trace, mass_frac):
    """
    Returns highest probability density region given by
    a set of samples.
    Parameters
    ----------
    trace : array
        1D array of MCMC samples for a single variable
    mass_frac : float with 0 < mass_frac <= 1
        The fraction of the probability to be included in
        the HPD.  For hreple, `massfrac` = 0.95 gives a
        95% HPD.

    Returns
    -------
    output : array, shape (2,)
        The bounds of the HPD

    Notes
    -----
    We thank Justin Bois (BBE, Caltech) for developing this function.
    http://bebi103.caltech.edu/2015/tutorials/l06_credible_regions.html
    """
    # Get sorted list
    d = np.sort(np.copy(trace))

    # Number of total samples taken
    n = len(trace)

    # Get number of samples that should be included in HPD
    n_samples = np.floor(mass_frac * n).astype(int)

    # Get width (in units of data) of all intervals with n_samples samples
    int_width = d[n_samples:] - d[:n - n_samples]

    # Pick out minimal interval
    min_int = np.argmin(int_width)

    # Return interval
    return np.array([d[min_int], d[min_int + n_samples]])


def gauss_kernel(t):
    """
    Gaussian kernel.
    """
    return np.exp(-t**2 / 2.0)


def nw_kernel_smooth(x_0, x, y, lam, kernel_fun=gauss_kernel):
    """
    Gives smoothed data at points x_0 using a Nadaraya-Watson kernel 
    estimator.  The data points are given by NumPy arrays x, y.
        
    kernel_fun must be of the form
        kernel_fun(t), 
    where t = |x - x_0| / lam
    
    This is not a fast way to do it, but it simply implemented!
    """
    
    # Function to give estimate of smoothed curve at single point.
    def single_point_estimate(x_0_single):
        """
        Estimate at a single point x_0_single.
        """
        t = np.abs(x_0_single - x) / lam
        return np.dot(kernel_fun(t), y) / kernel_fun(t).sum()
    
    # If we only want an estimate at a single data point
    if np.isscalar(x_0):
        return single_point_estimate(x_0)
    else:  # Get estimate at all points
        y_smooth = np.empty_like(x_0)
        for i in range(len(x_0)):
            y_smooth[i] = single_point_estimate(x_0[i])
        return y_smooth

# Bootstrap estimates
def bootstrap_estimate(data, function, n_estimates=100000):
    '''
    Function to compute a bootstrap estimate of any function applied to a data
    array.
    Parameters
    ----------
    data : array-like.
        raw data values on which to perform the bootstrap resampling.
    function : function
        function (such as mean or standard deviation) that will be aplied on
        the data array. This function must return a single scalar value.
    n_estimates : int. Default = 100000
        number of bootstrap estimates to be performed over the data

    Returns
    -------
    bootstrap : array-like.
        bootstrap estimates.
    '''
    # Define number of data points
    n = len(data)

    # Initialize array to save bootstrap estimates
    bootstrap = np.zeros(n_estimates)

    # Perform estimates
    for i in range(n_estimates):
        # resample data
        d = np.random.choice(data, size=n, replace=True)
        bootstrap[i] = function(d)

    return bootstrap
    