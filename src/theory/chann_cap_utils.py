# -*- coding: utf-8 -*-
"""
evolution_bits_utils

This file is a compilation of the funtions developed for the channel 
capacity project. Most of the functions found here can also be found
in different iPython notebooks, but in order to break down those notebooks
into shorter and more focused notebooks it is necessary to call some functions
previously defined.
By importing these utils you will have available all important functions
defined in the project.
"""

#==============================================================================
# Our numerical workhorses
import numpy as np
import scipy.optimize
import scipy.special
import scipy.integrate
from sympy import mpmath
import pandas as pd

# Import libraries to parallelize processes
from joblib import Parallel, delayed

# Import random library to make random sampling of parameters
import random

# Import plotting utilities
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

# Seaborn, useful for graphics
import seaborn as sns

from fit_bivariate_gaussian_astroML import *

#==============================================================================
# Generic themrodynamic functions
#============================================================================== 
def p_act(C, ka, ki, epsilon=4.5, logC=False):
    '''
    Returns the probability of a lac repressor being in the active state, i.e.
    able to bind the promoter as a function of the ligand concentration.
    
    Parameters
    ----------
    C : array-like.
        concentration(s) of ligand at which evaluate the function.
    ka, ki : float.
        dissociation constants for the active and inactive states respectively
        in the MWC model of the lac repressor.
    epsilon : float.
        energetic barrier between the inactive and the active state.
    logC : Bool.
        boolean indicating if the concentration is given in log scale
    
    Returns
    -------
    p_act : float. 
        The probability of the repressor being in the active state.
    '''
    C = np.array(C)
    if logC:
        C = 10**C
        
    return (1 + C / ka)**2 / \
            ((1 + C / ka)**2 + np.exp(-epsilon) * (1 + C / ki)**2)

#==============================================================================
# chemical_master_eq_analytic_mRNA
#==============================================================================

def kon_fn(epsilon, k0=2.7E-3):
    '''
    Returns the value of the kon rate constant as a function of the difussion
    limited constant k0 and the binding energy of the thermodynamic model
    for simple repression
    Parameters
    ----------
    epsilon : float.
        value of the binding energy in the thermodynamic model
    k0 : float.
        value of the difussion limited rate constant
    '''
    return 1.66 / 2 * k0 * 4.6E6 * np.exp(epsilon)

#=============================================================================== 

# define a np.frompyfunc that allows us to evaluate the sympy.mp.math.hyp1f1
np_log_hyp= np.frompyfunc(lambda x, y, z: \
mpmath.ln(mpmath.hyp1f1(x, y, z, zeroprec=1000)), 3, 1)

def log_p_m_mid_C(C, mRNA, rep, ki, ka, epsilon, kon, k0, gamma, r_gamma,
                 logC=False):
    '''
    Computes the log conditional probability lnP(m|C,R), 
    i.e. the probability of having m mRNA molecules given 
    an inducer concentration C and a repressor copy number R.
    
    Parameters
    ----------
    C : float.
        Concentration at which evaluate the probability. if logC=True, then
        this array is defined as log10(C).
    mRNA : float.
        mRNA copy number at which evaluate the probability.        
    repressor : float.
        repressor copy number per cell.
    ki, ka : float.
        dissociation constants for the inactive and active states respectively
        in the MWC model of the lac repressor.
    epsilon : float.
        energetic barrier between the inactive and the active state.
    kon : float.
        rate of activation of the promoter in the chemical master equation
    k0 : float.
        diffusion limited rate of a repressor binding the promoter
    gamma : float.
        half-life time for the mRNA.
    r_gamma : float.
        average number of mRNA in the unregulated promoter.
    logC : Bool.
        boolean indicating if the concentration is given in log scale
    
    Returns
    -------
    log probability lnP(m|c,R)
    '''
    # Convert C and mRNA into np.arrays
    C = np.array(C)
    mRNA = np.array(mRNA)
    if logC:
        C = 10**C
    
    # Calculate the off rate including the MWC model
    koff = k0 * rep * p_act(C, ka, ki, epsilon)

    # Compute the probability
    lnp = scipy.special.gammaln(kon / gamma + mRNA) \
    - scipy.special.gammaln(mRNA + 1) \
    - scipy.special.gammaln((koff + kon) / gamma + mRNA) \
    + scipy.special.gammaln((koff + kon) / gamma) \
    - scipy.special.gammaln(kon / gamma) \
    + mRNA * np.log(r_gamma) \
    + np_log_hyp(kon / gamma + mRNA,
            (koff + kon) / gamma + mRNA, -r_gamma)
    
    return lnp.astype(float)

#==============================================================================
# chemical_masater_eq_analytic_protein
#============================================================================== 

# define a np.frompyfunc that allows us to evaluate the sympy.mp.math.hyp1f1
np_log_gauss_hyp = np.frompyfunc(lambda a, b, c, z: \
mpmath.ln(mpmath.hyp2f1(a, b, c, z,  maxprec=10000)).real, 4, 1)

def log_p_p_mid_C(C, protein, rep, ka, ki, epsilon, kon, k0, gamma_m, r_gamma_m,
                  gamma_p, r_gamma_p, logC=False):
    '''
    Computes the log conditional probability lnP(p|C,R), 
    i.e. the probability of having p proteins given 
    an inducer concentration C and a repressor copy number R.
    
    Parameters
    ----------
    C : array-like.
        Concentration at which evaluate the probability.
    protein : array-like.
        protein copy number at which evaluate the probability.        
    repressor : float.
        repressor copy number per cell.
    ki, ka : float.
        dissociation constants for the inactive and active states respectively
        in the MWC model of the lac repressor.
    epsilon : float.
        energetic barrier between the inactive and the active state.
    kon : float.
        rate of activation of the promoter in the chemical master equation
    k0 : float.
        diffusion limited rate of a repressor binding the promoter
    gamma_m : float.
        half-life time for the mRNA.
    r_gamma_m : float.
        average number of mRNA in the unregulated promoter.
    gamma_p : float.
        half-life time for the protein.
    r_gamma_p : float.
        average number of protein per mRNA in the unregulated promoter.
    logC : Bool.
        boolean indicating if the concentration is given in log scale. If True
        C = 10**C
    Returns
    -------
    log probability lnP(p|c,R)
    '''
    # Convert C and mRNA into np.arrays
    C = np.array(C)
    protein = np.array(protein)
    # Convert from log if necessary
    if logC:
        C = 10**C
 
    # Calculate the off rate including the MWC model
    koff = k0 * rep * p_act(C, ka, ki, epsilon)

    # compute the variables needed for the distribution
    a = r_gamma_m * gamma_m / gamma_p # r_m / gamma_p
    b = r_gamma_p * gamma_p / gamma_m # r_p / gamma_m
    gamma = gamma_m / gamma_p
    Kon = kon / gamma_p
    Koff = koff / gamma_p
    
    phi = np.sqrt((a + Kon + Koff)**2 - 4 * a * Kon)
    
    alpha = 1 / 2 * (a + Kon + Koff + phi)
    beta = 1 / 2 * (a + Kon + Koff - phi)
    
    # Compute the probability
    lnp = scipy.special.gammaln(alpha + protein) \
    + scipy.special.gammaln(beta + protein) \
    + scipy.special.gammaln(Kon + Koff) \
    - scipy.special.gammaln(protein + 1) \
    - scipy.special.gammaln(alpha) \
    - scipy.special.gammaln(beta) \
    - scipy.special.gammaln(Kon + Koff + protein) \
    + protein * (np.log(b) - np.log(1 + b)) \
    + alpha * np.log(1 - b / (1 + b)) \
    + np_log_gauss_hyp(alpha + protein, Kon + Koff - beta,
            Kon + Koff + protein, b / (1 + b))
    return lnp.astype(float)

#=============================================================================== 

def log_p_p_mid_C_spline(C, p_range, step, rep, ka, ki, omega, 
                         kon, k0, gamma_m, r_gamma_m, gamma_p, r_gamma_p,
                         norm_check=False, tol=0.01):
    '''
    Computes the log conditional probability lnP(p|C,R), 
    i.e. the probability of having p proteins given 
    an inducer concentration C and a repressor copy number R.
    This function performs an interpolation with n_points uniformly
    distributed in p_range
    
    Parameters
    ----------
    C : array-like.
        Concentration at which evaluate the probability.
    p_range : array-like.
        Protein copy number range at which evaluate the probability.        
    step : int.
        Step size to take between values in p_range.
    repressor : float.
        Repressor copy number per cell.
    ki, ka : float.
        Dissociation constants for the inactive and active states respectively
        in the MWC model of the lac repressor.
    omega : float.
        Energetic barrier between the inactive and the active state.
    kon : float.
        Rate of activation of the promoter in the chemical master equation
    k0 : float.
        Diffusion limited rate of a repressor binding the promoter
    gamma_m : float.
        Half-life time for the mRNA.
    r_gamma_m : float.
        Average number of mRNA in the unregulated promoter.
    gamma_p : float.
        Half-life time for the protein.
    r_gamma_p : float.
        Average number of protein per mRNA in the unregulated promoter.
    norm_check : bool.
        Check if the returned distribution is normalized, and if not perform
        the full evaluation of the analytical expression.
    tol : float.
        +- Tolerance allowed for the normalization. The distribution is considered
        normalized if it is within 1+-tol
    Returns
    -------
    log probability lnP(p|c,R)
    '''
    # Convert C and the protein range into np.arrays
    C = np.array(C)
    protein = np.arange(p_range[0], p_range[1], step)
    protein = np.append(protein, p_range[1])
    # Compute the probability
    lnp = log_p_p_mid_C(C, protein, rep, ka, ki, omega, 
                        kon, k0, gamma_m, r_gamma_m, gamma_p, r_gamma_p)
    
    # Perform the cubic spline interpolation
    lnp_spline = scipy.interpolate.interp1d(protein, lnp, kind='cubic')
    # return the complete array of proteins evaluated with the spline
    p_array = np.arange(p_range[0], p_range[1])
    
    lnp = lnp_spline(p_array)
    # If ask to check the normalization of the distribution
    if norm_check:
        if (np.sum(np.exp(lnp)) <= 1 + tol) and (np.sum(np.exp(lnp)) >= 1 - tol):
            return lnp
        else:
            print('Did not pass the normalization test. Re-doing calculation')
            protein = np.arange(p_range[0], p_range[1])
            return log_p_p_mid_C(C, protein, rep, ka, ki, omega, 
                        kon, k0, gamma_m, r_gamma_m, gamma_p, r_gamma_p)
    else:
        return lnp

#=============================================================================== 

def log_p_p_mid_C_spline(C, p_range, step, rep, ka, ki, omega, 
                         kon, k0, gamma_m, r_gamma_m, gamma_p, r_gamma_p,
                         norm_check=False, tol=0.01):
    '''
    Computes the log conditional probability lnP(p|C,R), 
    i.e. the probability of having p proteins given 
    an inducer concentration C and a repressor copy number R.
    This function performs an interpolation with n_points uniformly
    distributed in p_range
    
    Parameters
    ----------
    C : array-like.
        Concentration at which evaluate the probability.
    p_range : array-like.
        Protein copy number range at which evaluate the probability.        
    step : int.
        Step size to take between values in p_range.
    repressor : float.
        Repressor copy number per cell.
    ki, ka : float.
        Dissociation constants for the inactive and active states respectively
        in the MWC model of the lac repressor.
    omega : float.
        Energetic barrier between the inactive and the active state.
    kon : float.
        Rate of activation of the promoter in the chemical master equation
    k0 : float.
        Diffusion limited rate of a repressor binding the promoter
    gamma_m : float.
        Half-life time for the mRNA.
    r_gamma_m : float.
        Average number of mRNA in the unregulated promoter.
    gamma_p : float.
        Half-life time for the protein.
    r_gamma_p : float.
        Average number of protein per mRNA in the unregulated promoter.
    norm_check : bool.
        Check if the returned distribution is normalized, and if not perform
        the full evaluation of the analytical expression.
    tol : float.
        +- Tolerance allowed for the normalization. The distribution is considered
        normalized if it is within 1+-tol
    Returns
    -------
    log probability lnP(p|c,R)
    '''
    # Convert C and the protein range into np.arrays
    C = np.array(C)
    protein = np.arange(p_range[0], p_range[1], step)
    protein = np.append(protein, p_range[1])
    # Compute the probability
    lnp = log_p_p_mid_logC(C, protein, rep, ka, ki, omega, 
                        kon, k0, gamma_m, r_gamma_m, gamma_p, r_gamma_p)
    
    # Perform the cubic spline interpolation
    lnp_spline = scipy.interpolate.interp1d(protein, lnp, kind='cubic')
    # return the complete array of proteins evaluated with the spline
    p_array = np.arange(p_range[0], p_range[1])
    
    lnp = lnp_spline(p_array)
    # If ask to check the normalization of the distribution
    if norm_check:
        if (np.sum(np.exp(lnp)) <= 1 + tol) and (np.sum(np.exp(lnp)) >= 1 - tol):
            return lnp
        else:
            print('Did not pass the normalization test. Re-doing calculation')
            protein = np.arange(p_range[0], p_range[1])
            return log_p_p_mid_C(C, protein, rep, ka, ki, omega, 
                        kon, k0, gamma_m, r_gamma_m, gamma_p, r_gamma_p)
    else:
        return lnp

#==============================================================================
# blahut_arimoto_channel_capacity
#=============================================================================== 

def channel_capacity(QmC, epsilon=1E-3, info=1E4):
    '''
    Performs the Blahut-Arimoto algorithm to compute the channel capacity
    given a channel QmC.

    Parameters
    ----------
    QmC : array-like 
        definition of the channel with C inputs and m outputs.
    epsilon : float.
        error tolerance for the algorithm to stop the iterations. The smaller
        epsilon is the more precise the rate-distortion function is, but also
        the larger the number of iterations the algorithm must perform
    info : int.
        Number indicating every how many cycles to print the cycle number as
        a visual output of the algorithm.
    Returns
    -------
    C : float.
        channel capacity, or the maximum information it can be transmitted 
        given the input-output function.
    pc : array-like.
        array containing the discrete probability distribution for the input 
        that maximizes the channel capacity
    '''
    # initialize the probability for the input.
    pC = np.repeat(1 / QmC.shape[0], QmC.shape[0])
        
    # Initialize variable that will serve as termination criteria
    Iu_Il = 1
    
    loop_count = 0
    # Perform a while loop until the stopping criteria is reached
    while Iu_Il > epsilon:
        if (loop_count % info == 0) & (loop_count != 0):
            print('loop : {0:d}, Iu - Il : {1:f}'.format(loop_count, Iu_Il))
        loop_count += 1
        # compute the relevant quantities. check the notes on the algorithm
        # for the interpretation of these quantities
        # cC = exp(∑_m Qm|C log(Qm|C / ∑_c pC Qm|C))
        sum_C_pC_QmC = np.sum((pC * QmC.T).T, axis=0)
        QmC_log_QmC_sum_C_pC_QmC = QmC * np.log(QmC / sum_C_pC_QmC)
        # check for values that go to -inf because of 0xlog0
        QmC_log_QmC_sum_C_pC_QmC[np.isnan(QmC_log_QmC_sum_C_pC_QmC)] = 0
        QmC_log_QmC_sum_C_pC_QmC[np.isneginf(QmC_log_QmC_sum_C_pC_QmC)] = 0
        cC = np.exp(np.sum(QmC_log_QmC_sum_C_pC_QmC, axis=1))
       
        # I_L log(∑_C pC cC)
        Il = np.log(np.sum(pC * cC))
        
        # I_U = log(max_C cC)
        Iu = np.log(cC.max())
        
        # pC = pC * cC / ∑_C pC * cC
        pC = pC * cC / np.sum(pC * cC)
        
        Iu_Il = Iu - Il
        
    # convert from nats to bits
    Il = Il / np.log(2)
    return Il, pC, loop_count

#=============================================================================== 

def trans_matrix(C, m, logPmC_fun, param, tol=0.01,
                 spline=False, verbose=False):
    '''
    Computes the transition matrix P(m|C) for a promoter given a series of 
    concentrations C and molecules (either mRNA or protein) where the transition 
    matrix is  built according to the logPmC_fun.
    The function can take the spline approximation to the logPmC_fun. If so the
    param dictionary should include the step and tol parameters for this function.
    
    IMPORTANT : We have noticed that for some high repressor copy numbers the 
    calculation at high repressor copy number diverges. This problem is solved
    by taking a smaller range of proteins in the calculation. That's why the 
    function automatically checks that the distributions are normalized, and
    corrects for the ones that are not.

    Parameters
    ----------
    C : array-like.
        Concentration discretized values that are used to compute the
        distributions. logC is assumed to be uniform over the log of the
        values given in the array.
    m : array-like.
        molecule (either mRNA or protein) copy number per cell.
    logPmC_fun : function.
        Function to determine the conditional distribution logP(m|C). This in
        general will be one of the versions of the chemical master equation
        solutions, but it can be extended to any input-outpu function one
        wants to use.
        NOTE: When applied to C and m it must return an array of the same
        length.
    param : dictionary.
        Dictionary containing the parameters for the function that computes
        the probability P(m|C). Look for the help on the specific logPmC_fun
        used to know the list of parameters required.
    epsilon : float.
        Error tolerance for the algorithm to stop the iterations. The smaller
        epsilon is the more precise the rate-distortion function is, but also
        the larger the number of iterations the algorithm must perform
    info : int.
        Number to indicate every how many steps to print the step that
        the algorithm is running.
    tol : float.
        +- Tolerance allowed for the normalization. The distribution is considered
        normalized if it is within 1+-tol
    spline : bool.
        Indicate ff the function used to compute the probability uses the 
        spline fitting approximation.
        NOTE : if this is the case the m array should be length 2 with the min
        and the max of the range to be used and the param dictionary should 
        contain an entry step for the step size.
    verbose : bool.
        Print the concentrations which distributions are being calculated at
        the moment or not.
    Returns
    -------
    Chan_cap : float.
        Channel capacity, or the maximum information it can be transmitted 
        given the input-output function.
    pc : array-like.
        Array containing the discrete probability distribution for the input 
        that maximizes the channel capacity
    '''
    # Check if the spline approximation is called
    if spline:
        param['tol'] = tol
        # Since we'll need 2D integrals, make a mesh of variables
        CC, mm = np.meshgrid(C, np.arange(m[0], m[1]))
        # initialize the matrix to save the probabilities
        QmC = np.zeros_like(CC)
        # Loop through the concentrations
        for i, c in enumerate(C):
            if verbose:
                print('C = {:f}'.format(c))
            # Build P(m|C) the input-output transition matrix
            prob = np.exp(logPmC_fun(c, m, **param))
            # Check that the distribution is normalized, otherwise perform
            # the calculation using a smaller range of m molecules to evaluate
            # since this solves the numerical problem that high number of 
            # repressor copy numbers tend to have.
            if (np.sum(prob) <= 1 + tol) & (np.sum(prob) >= 1 - tol):
                QmC[:, i] = prob
            else:
                print('Dist not normalized, re-computing with smaller range.')
                # If not normalize compute up to 1/3 of the original range
                mnew = np.floor(m[1] / 3)
                QmC[m[0]:mnew, i] = np.exp(logPmC_fun(c, [m[0], mnew], **param))
        
    # If not spline approximation is called perform the regular computation
    # But also checking that the distribution is normalized
    else:
        # Since we'll need 2D integrals, make a mesh of variables
        CC, mm = np.meshgrid(C, m)
        # Build P(m|C) the input-output transition matrix
        # Initialize the matrix
        QmC = np.zeros_like(CC)
        for i, c in enumerate(C):
            if verbose:
                print('C = {:f}'.format(c))
            # Build P(m|C) the input-output transition matrix
            prob = np.exp(logPmC_fun(c, m, **param))
            # Check that the distribution is normalized, otherwise perform
            # the calculation using a smaller range of m molecules to evaluate
            # since this solves the numerical problem that high number of 
            # repressor copy numbers tend to have.
            if (np.sum(prob) <= 1 + tol) & (np.sum(prob) >= 1 - tol):
                QmC[:, i] = prob
            else:
                print('Dist not normalized, re-computing with smaller range.')
                # If not normalize compute up to 1/3 of the original range
                mnew = np.floor(m[-1] / 3)
                QmC[m[0]:mnew, i] = np.exp(logPmC_fun(c, np.arange(m[0], mnew), 
                                                      **param))
                
    return QmC

#=============================================================================== 
# Automatic gating of the flow cytometry data
#=============================================================================== 

def fit_2D_gaussian(df, x_val='FSC-A', y_val='SSC-A', log=False):
    '''
    This function hacks astroML fit_bivariate_normal to return the mean and
    covariance matrix when fitting a 2D gaussian fuction to the data contained
    in the x_vall and y_val columns of the DataFrame df.
    Parameters
    ----------
    df : DataFrame.
        dataframe containing the data from which to fit the distribution
    x_val, y_val : str.
        name of the dataframe columns to be used in the function
    log : bool.
        indicate if the log of the data should be use for the fit or not
        
    Returns
    -------
    mu : tuple.
        (x, y) location of the best-fit bivariate normal
    cov : 2 x 2 array
        covariance matrix.
        cov[0, 0] = variance of the x_val column
        cov[1, 1] = variance of the y_val column
        cov[0, 1] = cov[1, 0] = covariance of the data
    '''
    if log:
        x = np.log10(df[x_val])
        y = np.log10(df[y_val])
    else:
        x = df[x_val]
        y = df[y_val]
        
    # Fit the 2D Gaussian distribution using atroML function
    mu, sigma_1, sigma_2, alpha = fit_bivariate_normal(x, y, robust=True)

    # compute covariance matrix from the standar deviations and the angle
    # that the fit_bivariate_normal function returns
    sigma_xx = ((sigma_1 * np.cos(alpha)) ** 2
                + (sigma_2 * np.sin(alpha)) ** 2)
    sigma_yy = ((sigma_1 * np.sin(alpha)) ** 2
                + (sigma_2 * np.cos(alpha)) ** 2)
    sigma_xy = (sigma_1 ** 2 - sigma_2 ** 2) * np.sin(alpha) * np.cos(alpha)
    
    # put elements of the covariance matrix into an actual matrix
    cov = np.array([[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]])
    
    return mu, cov

#=============================================================================== 

def gauss_interval(df, mu, cov, x_val='FSC-A', y_val='SSC-A', log=False):
    '''
    Computes the of the statistic
    (x - µx)'∑(x - µx) 
    for each of the elements in df columns x_val and y_val.
    
    Parameters
    ----------
    df : DataFrame.
        dataframe containing the data from which to fit the distribution
    mu : array-like.
        (x, y) location of bivariate normal
    cov : 2 x 2 array
        covariance matrix
    x_val, y_val : str.
        name of the dataframe columns to be used in the function
    log : bool.
        indicate if the log of the data should be use for the fit or not 
    
    Returns
    -------
    statistic_gauss : array-like.
        array containing the result of the linear algebra operation:
        (x - µx)'∑(x - µx) 
    '''
    # Determine that the covariance matrix is not singular
    det = np.linalg.det(cov)
    if det == 0:
        raise NameError("The covariance matrix can't be singular")
            
    # Compute the vector x defined as [[x - mu_x], [y - mu_y]]
    if log: 
        x_vect = np.log10(np.array(df[[x_val, y_val]]))
    else:
        x_vect = np.array(df[[x_val, y_val]])
    x_vect[:, 0] = x_vect[:, 0] - mu[0]
    x_vect[:, 1] = x_vect[:, 1] - mu[1]
    
    # compute the inverse of the covariance matrix
    inv_sigma = np.linalg.inv(cov)
    
    # compute the operation
    interval_array = np.zeros(len(df))
    for i, x in enumerate(x_vect):
        interval_array[i] = np.dot(np.dot(x, inv_sigma), x.T)
        
    return interval_array

#=============================================================================== 

def auto_gauss_gate(df, alpha, x_val='FSC-A', y_val='SSC-A', log=False,
                    verbose=False):
    '''
    Function that applies an "unsupervised bivariate Gaussian gate" to the data
    over the channels x_val and y_val.
    
    Parameters
    ----------
    df : DataFrame.
        dataframe containing the data from which to fit the distribution
    alpha : float. [0, 1]
        fraction of data aimed to keep. Used to compute the chi^2 quantile function
    x_val, y_val : str.
        name of the dataframe columns to be used in the function
    log : bool.
        indicate if the log of the data should be use for the fit or not 
    verbose : bool.
        indicate if the percentage of data kept should be print
    Returns
    -------
    df_thresh : DataFrame
        Pandas data frame to which the automatic gate was applied.
    '''
    data = df[[x_val, y_val]]
    # Fit the bivariate Gaussian distribution
    mu, cov = fit_2D_gaussian(data, log=log)

    # Compute the statistic for each of the pair of log scattering data
    interval_array = gauss_interval(data, mu, cov, log=log)
        
    # Find which data points fall inside the interval
    idx = interval_array <= scipy.stats.chi2.ppf(alpha, 2)

    # print the percentage of data kept
    if verbose:
        print('''
        with parameter alpha={0:0.2f}, percentage of data kept = {1:0.2f}
        '''.format(alpha, np.sum(idx) / len(df)))

    return df[idx]

#============================================================================== 
# Plotting style
#============================================================================== 
def set_plotting_style():
    """
    Formats plotting enviroment to that used in Physical Biology of the Cell,
    2nd edition. To format all plots within a script, simply execute
    `mwc_induction_utils.set_plotting_style() in the preamble.
    """
    rc = {'lines.linewidth': 2,
          'axes.labelsize': 16,
          'axes.titlesize': 18,
          'axes.facecolor': '#E3DCD0',
#          'xtick.major' : 20,
          'xtick.labelsize': 13,
          'ytick.labelsize': 13,
          'font.family': 'Lucida Sans Unicode',
          'grid.linestyle': ':',
          'grid.linewidth': 1.5,
          'grid.color': '#ffffff',
          'mathtext.fontset': 'stixsans',
          'mathtext.sf': 'sans',
          'legend.frameon': True,
          'legend.fontsize': 11}
    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    plt.rc('mathtext', fontset='stixsans', sf='sans')
    sns.set_style('darkgrid', rc=rc)
    sns.set_palette("colorblind", color_codes=True)
    sns.set_context('notebook', rc=rc)

#============================================================================== 
# Useful generic functions
#============================================================================== 
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
