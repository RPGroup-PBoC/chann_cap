# -*- coding: utf-8 -*-
"""
evolution_bits_utils

This file is a compilation of the funtions developed for the information
theory/evolution project. Most of the functions found here can also be found
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

#==============================================================================
# 01_simple_fitness_landscape.ipynb
#==============================================================================

def cost_func(p_rel, eta_o=0.02, M=1.8):
    '''
    Returns the relative growth rate cost of producing LacZ protein according
    to Dekel and Alon's model:
        eta_2 = eta_o * p_rel / (1 - p_rel / M)

    Parameter
    ---------
    p_rel : array-like.
        Relative expression with respect to the wild type expression when
        fully induced with IPTG
    eta_o : float.
        Parameter of the cost function
    M : float.
        Parameter of the cost function

    Returns
    -------
    eta_2 : array-like
        relative reduction in growth rate with respect to wild type when not
        expressing the enzyme.
    '''
    p_rel = np.array(p_rel)
    return eta_o * p_rel / (1 - p_rel / M)

#==============================================================================

def benefit_func(p_rel, C_array, delta=0.17, Ks=0.4):
    '''
    Returns the relative growth rate benefit of producing LacZ protein
    according to Dekel and Alon's model:
        r = delta * p_rel * C / (Ks + C)

    Parameter
    ---------
    p_rel : array-like.
        Relative expression with respect to the wild type expression when
        fully induced with IPTG.
    C_array : array-like.
        Substrate concentration.
    delta : float.
        growth benefit per substrate cleaved per enzyme.
    Ks : float.
        Monod constant for half maximum growth rate.

    Returns
    -------
    r : array-like
        relative increase in growth rate with respect to wild type when not
        expressing the enzyme.
    '''
    p_rel = np.array(p_rel)
    return delta * p_rel * C_array / (Ks + C_array)

#==============================================================================

def fitness(p_rel, C_array, delta=0.17, Ks=0.4, eta_o=0.02, M=1.8,
           logC=False):
    '''
    Returns the relative fitness according to Dekel and Alon's model.
    
    Parameter
    ---------
    p_rel : array-like.
        Relative expression with respect to the wild type expression when
        fully induced with IPTG.
    C_array : array-like.
        Substrate concentration. If logC==True this is defined as log10(C)
    delta : float.
        growth benefit per substrate cleaved per enzyme.
    Ks : float.
        Monod constant for half maximum growth rate.
    eta_o : float.
        Parameter of the cost function
    M : float.
        Parameter of the cost function
    logC : Bool.
        boolean indicating if the concentration is given in log scale
    
    Returns
    -------
    fitness : array-like
        relative fitness with respect to wild type when not
        expressing the enzyme.
    '''
    p_rel = np.array(p_rel)
    C_array = np.array(C_array)
    if logC:
        C_array = 10**C_array
    # Compute benefit - cost
    return benefit_func(p_rel, C_array, delta, Ks) - cost_func(p_rel, eta_o, M)

#============================================================================== 

def p_opt(C_array, delta=0.17, Ks=0.4, eta_o=0.02, M=1.8, logC=False):
    '''
    Returns the optimal protein expression level p* as a function of
    substrate concentration.
    
    Parameters
    ----------
    C_array : array-like.
        Substrate concentration.
    delta : float.
        growth benefit per substrate cleaved per enzyme.
    Ks : float.
        Monod constant for half maximum growth rate.
    eta_o : float.
        Parameter of the cost function
    M : float.
        Parameter of the cost function
    logC : Bool.
        boolean indicating if the concentration is given in log scale
        
    Returns
    -------
    p* the optimal expression level for a given concentration.
    '''
    C_array = np.array(C_array)
    if logC:
        C_array = 10**C_array
    
    # Dekel and Alon specify that concentrations lower than a lower 
    # threshold should be zero. Then let's build that array
    thresh = Ks * (delta / eta_o - 1)**-1
    popt = np.zeros_like(C_array)
    popt[C_array > thresh] = M * (1 - np.sqrt(eta_o / delta * \
        (C_array[C_array > thresh] + Ks) / C_array[C_array > thresh]))
    
    return popt

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
# 02_chemical_master_eq
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
    koff = k0 * rep * evo_utils.p_act(C, ka, ki, epsilon)

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
    koff = k0 * rep * evo_utils.p_act(C, ka, ki, epsilon)

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
# 03_mutual_information
#==============================================================================

def mutual_info(C, mRNA, PC_fun, logPmC_fun, params, cutoff=1E-10):
    '''
    Computes the mutual information between the environment and the gene
    expression level on a grid of values of C and mRNA.

    Parameters
    ----------
    C : array-like.
        discretized values of the concentration at which the numerical
        integral will be evaluated.
    mRNA : array-like.
        discretized values of the mRNA copy number at which the numerical
        integral will be evaluated.
    PC_fun : function.
        function to determine the probability distribution of each of the
        elements of C.
        NOTE: When applied to C it must return an array of the same
        length.
    logPmC_fun : function.
        function to determine the conditional distribution logP(m|C). This in
        general will be one of the versions of the chemical master equation
        solutions, but it can be extended to any input-outpu function one
        wants to use.
        NOTE: When applied to C and mRNA it must return an array of the same
        length.
    params : dictionary.
        dictionary containing all the parameters to compute the mRNA
        distribution with the chemical-master equations approach.
        the parameters are:
        rep : float.
            repressor copy number per cell.
        ka, ki : float.
            dissociation constants for the active and inactive states
            respectively in the MWC model of the lac repressor.
        omega : float.
            energetic barrier between the inactive and the active state.
        kon : float.
            rate of activation of the promoter in the chemical master equation.
        k0 : float.
            diffusion limited rate of a repressor binding the promoter
        gamma : float.
            half-life time for the mRNA.
        r_gamma : float.
            average number of mRNA in the unregulated promoter.
    cutoff : float.
        necessary lower bound to determine when to ignore a term in the
        integral given the convention that 0 x log0 = 0.

    Returns
    -------
    '''
    # Since we'll need 2D integrals, make a mesh of variables
    CC, mm = np.meshgrid(C, mRNA)

    # Build P(m|C)
    PmC = np.exp(logPmC_fun(CC, mm, **params)).astype(float)

    # Build P(C)
    PC = PC_fun(C).astype(float)

    # Build P(m) by integrating P(C) * P(m|C) over C.
    Pm = scipy.integrate.simps(PC * PmC, x=C, axis=1).astype(float)
    Pm_tile = np.tile(Pm, (len(C), 1)).T

    # Make P(m|C) * log(P(m|C)) making sure no log(0).
    PmC_log_PmC = np.zeros_like(PmC)
    PmC_log_PmC[PmC > cutoff] = PmC[PmC > cutoff] * np.log2(PmC[PmC > cutoff])

    # Make P(m|C) * log(P(m)) making sure no log(0).
    PmC_log_Pm = np.zeros_like(PmC)
    PmC_log_Pm[Pm_tile > cutoff] = \
            PmC[Pm_tile > cutoff] * np.log2(Pm_tile[Pm_tile > cutoff])

    # Integrate over m
    int_m = \
    scipy.integrate.simps(PmC_log_PmC - PmC_log_Pm,
                        x=mRNA, axis=0).astype(float)

    # Return integral over C
    return scipy.integrate.simps(PC * int_m, x=C).astype(float)

#==============================================================================

def PC_unif(C):
    '''
    Returns a uniform PDF for an array C. Properly since it is a continuous
    variable the probability should be zero, but since we will be using Simpson's
    rule for numerical integration this funciton returns a numerical value > 0
    for P(C = c).

    Parameter
    ---------
    C : array-like.
        Concentrations at which evaluate the function

    Returns
    -------
    P(C) : array-like.
        evaluation of the PDF at each discrete point.
    '''
    return np.repeat(1 / (C.max() - C.min()), len(C))

#==============================================================================

def PC_expo(C, tau=2):
    '''
    Returns an exponential PDF for an array C. Properly since it is a
    continuous variable the probability should be zero, but since we
    will be using Simpson's rule for numerical integration this funciton
    returns a numerical value > 0 for P(C = c).

    Parameter
    ---------
    C : array-like.
        concentrations at which evaluate the function

    Returns
    -------
    P(C) : array-like.
        evaluation of the PDF at each discrete point.
    '''
    return tau * np.exp(- tau * C)

#==============================================================================

def PlogC_unif(logC):
    '''
    Returns a uniform PDF for an array logC. Properly since it is a continuous
    variable the probability should be zero, but since we will be using Simpson's
    rule for numerical integration this funciton returns a numerical value > 0
    for P(logC = logc).

    Parameter
    ---------
    logC : array-like.
        Concentrations at which evaluate the function

    Returns
    -------
    P(C) : array-like.
        evaluation of the PDF at each discrete point.
    '''
    return np.repeat(1 / (logC.max() - logC.min()), len(logC))

#==============================================================================

def PC_log_unif(C):
    '''
    Returns a PDF for an array C on which log C is uniformly distributed.
    Properly since it is a continuous variable the probability should be zero,
    but since we will be using Simpson's rule for numerical integration this
    funciton returns a numerical value > 0 for P(C = c).

    Parameter
    ---------
    logC : array-like.
        log_10 concentrations at which evaluate the function

    Returns
    -------
    P(C) : array-like.
        evaluation of the PDF at each discrete point.
    '''
    return 1 / C / (np.log(C).max() - np.log(C).min())
#==============================================================================
# SPECTRAL INTETRATION

def cheb_points_1d(n, xends=[-1.0, 1.0]):
    """
    Computes Chebyshev points for a 1-D Chebyshev grid, returned as ndarray.

    xends specifies the end points of the array.
    """

    x = np.cos(np.pi * np.array(np.arange(n)) / (n-1))
    x = (xends[1] - xends[0]) / 2.0 * x + (xends[1] + xends[0]) / 2.0

    return x
#==============================================================================

def clenshaw_curtis_weights(n):
    """
    Computes the weights to be applied in Curtis-Clenshaw integration,
    sampling at n assuming Chebyshev points.

    Adapted from Nick Trefethen's book: Spectral Methods in Matlab
    """

    n -= 1  # This is to stay consistent with our indexing

    theta = np.pi * np.arange(n+1) / n
    w = np.zeros_like(theta)
    v = np.ones(n-1)
    if n % 2 == 0:
        w[0] = 1.0 / (n**2 - 1)
        w[-1] = w[0]
        for k in range(1, n//2):
            v -= 2.0 * np.cos(2.0 * k * theta[1:-1]) / (4.0 * k**2 - 1)
        v -= np.cos(n * theta[1:-1]) / (n**2 - 1)
    else:
        w[0] = 1.0 / n**2
        w[-1] = w[0]
        for k in range(1, (n-1)//2 + 1):
            v -= 2.0 * np.cos(2.0 * k * theta[1:-1]) / (4.0 * k**2 - 1)

    w[1:-1] = 2.0 * v / n

    return w

#==============================================================================

def cheb_quad(y, w, domain_size):
    """
    Perform spectral integration given Clenshaw-Curtis weights. The
    weights are computed from clenshaw_curtis_weights().
    """
    return np.dot(w, y) * domain_size / 2

#==============================================================================

def mutual_info_spectral(C_range, mRNA, PC_fun, logPmC_fun, params,
                         n_points=64, cutoff=1E-10):
    '''
    Computes the mutual information between the environment and the gene
    expression level on a grid of values of C and mRNA.

    Parameters
    ----------
    C_range : array-like.
        range of concentrations that should be taken for the integration.
        For spectral integration we just give the range because the grid points
        at which the integral is evaluated are computed with Chebyshev's formula.
    mRNA : array-like.
        discretized values of the mRNA copy number at which the numerical
        integral will be evaluated.
    PC_fun : function.
        function to determine the probability distribution of each of the
        elements of C.
        NOTE: When applied to C it must return an array of the same
        length.
    logPmC_fun : function.
        function to determine the conditional distribution logP(m|C). This in
        general will be one of the versions of the chemical master equation
        solutions, but it can be extended to any input-outpu function one
        wants to use.
        NOTE: When applied to C and mRNA it must return an array of the same
        length.
    params : dictionary.
        dictionary containing all the parameters to compute the mRNA
        distribution with the chemical-master equations approach.
        the parameters are:
        rep : float.
            repressor copy number per cell.
        ka, ki : float.
            dissociation constants for the active and inactive states
            respectively in the MWC model of the lac repressor.
        omega : float.
            energetic barrier between the inactive and the active state.
        kon : float.
            rate of activation of the promoter in the chemical master equation.
        k0 : float.
            diffusion limited rate of a repressor binding the promoter
        gamma : float.
            half-life time for the mRNA.
        r_gamma : float.
            average number of mRNA in the unregulated promoter.
    n_points : int.
        number of grid points used for the spectral integration. This is the
        number of Chebyshev points that are used for the integration.
    cutoff : float.
        necessary lower bound to determine when to ignore a term in the
        integral given the convention that 0 x log0 = 0.

    Returns
    -------
    The mutual information between the gene expession and the environment
    distribution in bits.
    '''
    # Convert C_range into a numpy array
    C_range = np.array(C_range)
    # Set the Chebyshev points and the Clenshaw curtis weights
    C_cheb= cheb_points_1d(n_points, [C_range.min(), C_range.max()])
    w = clenshaw_curtis_weights(n_points)

    # Since we'll need 2D integrals, make a mesh of variables
    CC, mm = np.meshgrid(C_cheb, mRNA)

    # Build P(m|C)
    PmC = np.exp(logPmC_fun(CC, mm, **params)).astype(float)

    # Build P(C)
    PC = PC_fun(C_cheb).astype(float)
    # Build P(m) by integrating P(C) * P(m|C) over C.
    Pm = cheb_quad((PC * PmC).T, w, C_cheb.max()-C_cheb.min()).astype(float)
    Pm_tile = np.tile(Pm, (n_points, 1)).T

    # Make P(m|C) * log(P(m|C)) making sure no log(0).
    PmC_log_PmC = np.zeros_like(PmC)
    PmC_log_PmC[PmC > cutoff] = PmC[PmC > cutoff] * np.log2(PmC[PmC > cutoff])

    # Make P(m|C) * log(P(m)) making sure no log(0).
    PmC_log_Pm = np.zeros_like(PmC)
    PmC_log_Pm[Pm_tile > cutoff] = \
            PmC[Pm_tile > cutoff] * np.log2(Pm_tile[Pm_tile > cutoff])

    # Integrate over m.
    # NOTE: since the distribution is only normalized for discrete values, we
    # actually sum over the values rather than integrating
    int_m = \
    np.sum(PmC_log_PmC - PmC_log_Pm, axis=0).astype(float)

    # Return integral over C
    return cheb_quad(PC * int_m, w, C_cheb.max()-C_cheb.min()).astype(float)
    
#==============================================================================
# 04_average_growth_rate
#==============================================================================

def avg_growth(C, mRNA, PC_fun, logPmC_fun, fitness_fun,
               mastereq_param, growth_param):
    '''
    Computes the average growth rate over a distribution of environments given
    a grid of values of C and mRNA.

    Parameters
    ----------
    C : array-like.
        discretized values of the concentration at which the numerical
        integral will be evaluated.
    mRNA : array-like.
        discretized values of the mRNA copy number at which the numerical
        integral will be evaluated.
    PC_fun : function.
        function to determine the probability distribution of each of the
        elements of C.
        NOTE: When applied to C it must return an array of the same
        length.
    logPmC_fun : function.
        function to determine the conditional distribution logP(m|C). This in
        general will be one of the versions of the chemical master equation
        solutions, but it can be extended to any input-outpu function one
        wants to use.
        NOTE: When applied to C and mRNA it must return an array of the same
        length.
    fitness_fun : function.
        function to determine the growth rate given an expression level mRNA
        and a substrate concentration C.
    mastereq_param : dictionary.
        dictionary containing all the parameters to compute the mRNA
        distribution with the chemical-master equations approach.
        the parameters are:
        rep : float.
            repressor copy number per cell.
        ka, ki : float.
            dissociation constants for the active and inactive states
            respectively in the MWC model of the lac repressor.
        omega : float.
            energetic barrier between the inactive and the active state.
        kon : float.
            rate of activation of the promoter in the chemical master equation.
        k0 : float.
            diffusion limited rate of a repressor binding the promoter
        gamma : float.
            half-life time for the mRNA.
        r_gamma : float.
            average number of mRNA in the unregulated promoter.
    growth_param : dictionary:
        dictionary containing all the parameters for the fitness function.
        the parameters are.
        r_gamma : float.
            average number of mRNA in the unregulated promoter.
        delta : float.
            growth benefit per substrate cleaved per enzyme.
        Ks : float.
            Monod constant for half maximum growth rate.
        eta_o : float.
            Parameter of the cost function
        M : float.
            Parameter of the cost function

    Returns
    -------
    average growth rate.
    '''
    # Since we'll need 2D integrals, make a mesh of variables
    CC, mm = np.meshgrid(C, mRNA)

    # Build P(m|C)
    PmC = np.exp(logPmC_fun(CC, mm, **mastereq_param)).astype(float)

    # Build P(C)
    PC = PC_fun(C).astype(float)

    # Build r(C, m)
    rCm = fitness_fun(CC, mm, **growth_param)

    # Integrate over m
    int_m = \
    scipy.integrate.simps(np.multiply(PmC, rCm), x=mRNA, axis=0)

    # Return integral over C
    return scipy.integrate.simps(PC * int_m, x=C).astype(float)

#==============================================================================

def avg_growth_spectral(C_range, mRNA, PC_fun, logPmC_fun, fitness_fun, 
               mastereq_param, growth_param,
               n_points=64, cutoff=1E-10):
    '''
    Computes the average growth rate over a distribution of environments given
    a range of values of C and a grid of mRNA using SPECTRAL INTEGRATION.
    
    Parameters
    ----------
    C_range : array-like.
        range of concentrations that should be taken for the integration.
        For spectral integration we just give the range because the grid points
        at which the integral is evaluated are computed with Chebyshev's formula.
    mRNA : array-like.
        discretized values of the mRNA copy number at which the numerical
        integral will be evaluated.
    PC_fun : function.
        function to determine the probability distribution of each of the
        elements of C. 
        NOTE: Should return n_points between C_range.min() and C_range.max()
    logPmC_fun : function.
        function to determine the conditional distribution logP(m|C). This in 
        general will be one of the versions of the chemical master equation
        solutions, but it can be extended to any input-outpu function one
        wants to use.
        NOTE: When applied to C and mRNA it must return an array of the same
        length.
    fitness_fun : function.
        function to determine the growth rate given an expression level mRNA
        and a substrate concentration C.
    mastereq_param : dictionary.
        dictionary containing all the parameters to compute the mRNA 
        distribution with the chemical-master equations approach.
        the parameters are:
        rep : float.
            repressor copy number per cell.
        ka, ki : float.
            dissociation constants for the active and inactive states 
            respectively in the MWC model of the lac repressor.
        omega : float.
            energetic barrier between the inactive and the active state.
        kon : float.
            rate of activation of the promoter in the chemical master equation.
        k0 : float.
            diffusion limited rate of a repressor binding the promoter
        gamma : float.
            half-life time for the mRNA.
        r_gamma : float.
            average number of mRNA in the unregulated promoter.
    growth_param : dictionary:
        dictionary containing all the parameters for the fitness function.
        the parameters are.
        r_gamma : float.
            average number of mRNA in the unregulated promoter.
        delta : float.
            growth benefit per substrate cleaved per enzyme.
        Ks : float.
            Monod constant for half maximum growth rate.
        eta_o : float.
            Parameter of the cost function
        M : float.
            Parameter of the cost function
    
    Returns
    -------
    '''
    # convert the C_range into a numpy array
    C_range = np.array(C_range)
    # Set the Chebyshev points and the Clenshaw curtis weights
    C_cheb= cheb_points_1d(n_points, [C_range.min(), C_range.max()])
    w = clenshaw_curtis_weights(n_points)
    
    # Since we'll need 2D integrals, make a mesh of variables
    CC, mm = np.meshgrid(C_cheb, mRNA)
    
    # Build P(m|C)
    PmC = np.exp(logPmC_fun(CC, mm, **mastereq_param)).astype(float)

    # Build P(C)
    PC = PC_fun(C_cheb).astype(float)
    
    # Build r(C, m)
    rCm = fitness_fun(CC, mm, **growth_param)
    
    # Integrate over m
    # Since the distribution is only normalized for discrete values we sum
    # over the values of m rather than using Simpson's rule
    int_m = np.sum(np.multiply(PmC, rCm), axis=0)
    
    # Return integral over C
    return cheb_quad(PC * int_m, w, \
                               C_cheb.max()-C_cheb.min()).astype(float)

#=============================================================================== 

def growth_C(C, mRNA, r_gamma=15.7, delta=0.17, Ks=0.4, eta_o=0.02, M=1.8):
    '''
    Returns the relative fitness according to Dekel and Alon's model.

    Parameter
    ---------
    C : array-like.
        substrate concentration.
    mRNA : array-like.
        mRNA copy number.
    r_gamma : float.
        average number of mRNA in the unregulated promoter.
    delta : float.
        growth benefit per substrate cleaved per enzyme.
    Ks : float.
        Monod constant for half maximum growth rate.
    eta_o : float.
        Parameter of the cost function
    M : float.
        Parameter of the cost function

    Returns
    -------
    fitness : array-like
        relative fitness with respect to wild type when not
        expressing the enzyme.
    '''
    p_rel = mRNA / r_gamma
    return benefit_func(p_rel, C, delta, Ks) - \
           cost_func(p_rel, eta_o, M)

#==============================================================================

def growth_logC(logC, mRNA, r_gamma=15.7, delta=0.17, Ks=0.4, eta_o=0.02, M=1.8):
    '''
    Returns the relative fitness according to Dekel and Alon's model.

    Parameter
    ---------
    logC : array-like.
        log substrate concentration.
    mRNA : array-like.
        mRNA copy number.
    r_gamma : float.
        average number of mRNA in the unregulated promoter.
    delta : float.
        growth benefit per substrate cleaved per enzyme.
    Ks : float.
        Monod constant for half maximum growth rate.
    eta_o : float.
        Parameter of the cost function
    M : float.
        Parameter of the cost function

    Returns
    -------
    fitness : array-like
        relative fitness with respect to wild type when not
        expressing the enzyme.
    '''
    p_rel = mRNA / r_gamma
    return benefit_func(p_rel, np.power(10, logC), delta, Ks) - \
           cost_func(p_rel, eta_o, M)

#==============================================================================
# 05_Blahut_algorithm_rate_distortion
#==============================================================================

def rate_dist(C, mRNA, beta, PC_fun, fitness_fun, param,
              fitness_scale=1, epsilon=1E-3):
    '''
    Performs the Blahut-Arimoto algorithm to compute the rate-distortion
    function R(<r>) given an input C and an output m.
    The average growth rate should be given as the RELATIVE DIFFERENCE WITH
    RESPECT TO THE OPTIMAL growth rate.

    Parameters
    ----------
    C : array-like.
        concentration discretized values that are used to compute the
        distributions. logC is assumed to be uniform over the log of the
        values given in the array.
    mRNA : array-like.
        relative number of mRNA with respect to wild-type discretized to
        compute the probability of this otherwise continuous variable.
    beta : float. [-inf, 0]
        slope of the line with constant I(Q) - beta * sD. This parameter emerges
        during the unconstraint optimization as a Lagrange multiplier. It plays
        the analogous role of the inverse temperature in the Boltzmann
        distribution.
    PC_fun : function.
        function to determine the probability distribution of each of the
        elements of C.
    fitness_fun : function.
        function to determine the growth rate given an expression level mRNA
        and a substrate concentration C.
    param : dictionary:
        dictionary containing all the parameters for the fitness function.
        the parameters are.
        r_gamma : float.
            average number of mRNA in the unregulated promoter.
        delta : float.
            growth benefit per substrate cleaved per enzyme.
        Ks : float.
            Monod constant for half maximum growth rate.
        eta_o : float.
            Parameter of the cost function
        M : float.
            Parameter of the cost function
    fitness_scale : float.
        An optional scaling parameter for the cost function.
        If set, the cost used in the algorithm is
        fitness_scale * fitness_fun(C, mRNA).
        This can be useful for avoiding numerical overflow or underflow
        for badly scaled cost functions.
    epsilon : float.
        error tolerance for the algorithm to stop the iterations. The smaller
        epsilon is the more precise the rate-distortion function is, but also
        the larger the number of iterations the algorithm must perform

    Returns
    -------
    q_m : len(m) array.
        marginal mRNA probability distribution.
    Q_m|C : len(C) x len(m) matrix.
        The input-output transition matrix for each of the inputs and outputs
        given in C and m respectively.
    D : float.
        Average distortion.
    R(D) : float.
        minimum amount of mutal information I(C;m) consistent with distortion D.
    '''
    # Let's make a 2D grid with the input and output values
    CC, mm = np.meshgrid(C, mRNA)

    # Define the probability of the concentration according to PC_fun.
    pC = PC_fun(C) / PC_fun(C).sum()

    # Initialize the proposed distribution as a uniform distribution
    # over the values of the mRNA
    qm0 = np.repeat(1 / len(mRNA), len(mRNA))

    # This will be the probabilities that will be updated on each cycle
    qm = qm0

    # Compute the cost matrix
    rhoCm = fitness_fun(CC, mm, **param).T * fitness_scale
    ACm = np.exp(beta * rhoCm)

    # Initialize variable that will serve as termination criteria
    Tu_Tl = 1

    # Initialize loop counter
    loop_count = 0

    # Perform a while loop until the stopping criteria is reached
    while Tu_Tl > epsilon:
        # compute the relevant quantities. check the notes on the algorithm
        # for the interpretation of these quantities
        # ∑_m qm ACm
        sum_m_qm_ACm = np.sum(qm * ACm, axis=1)

        # cm = ∑_C pC ACm / ∑_m qm ACm
        cm = np.sum(
        (pC * ACm.T / sum_m_qm_ACm).T, axis=0)

        # qm = qm * cm
        qm = qm * cm

        # Tu = ∑_m qm log cm
        Tu = - np.sum(qm * np.log(cm))

        # Tl = max_m log cm
        Tl = - np.log(cm).max()

        # Tu - Tl
        Tu_Tl = Tu - Tl

        # increase the loop count
        loop_count += 1

    # Compute the outputs after the loop is finished.

    # ∑_m qm ACm
    sum_m_qm_ACm = np.sum(qm * ACm, axis=1)

    # Qm|C = ACm qm / ∑_m ACm qm
    QmC = ((qm * ACm).T / sum_m_qm_ACm).T

    # D = ∑_c pC ∑_m Qm|C rhoCm
    D = np.sum(pC * np.sum(QmC * rhoCm , axis=1).T)

    # R(D) = beta D - ∑_C pC log ∑_m ACm qm - ∑_m qm log cm
    RD = beta * D \
    - np.sum(pC * np.log(sum_m_qm_ACm)) \
    - np.sum(qm * np.log(cm))

    # convert from nats to bits
    RD = RD / np.log(2)


    return pC, qm, rhoCm, QmC, D, RD

#==============================================================================

def growth_optimal(C, delta=0.17, Ks=0.4, eta_o=0.02, M=1.8):
    '''
    Returns the optimal growth rate at concentration C by obtaining the
    optimal expression level as defined in Dekel and Alon's paper and
    evaluating it in the fitness landscape function
    Parameter
    ---------
    C : array-like.
        substrate concentration.
    delta : float.
        growth benefit per substrate cleaved per enzyme.
    Ks : float.
        Monod constant for half maximum growth rate.
    eta_o : float.
        Parameter of the cost function
    M : float.
        Parameter of the cost function

    Returns
    -------
    optimal_growth : array-like
        maximum relative growth rate at concentration(s) C.
    '''
    # obtain the optimal expression level at concnetration(s) C
    mRNA_opt = p_opt(C, delta, Ks, eta_o, M)

    # return the optimal growth rate
    return benefit_func(mRNA_opt, C, delta, Ks) - \
           cost_func(mRNA_opt, eta_o, M)

#==============================================================================

def growth_diff_C(C, mRNA, r_gamma, delta=0.17, Ks=0.4, eta_o=0.02, M=1.8):
    '''
    Returns the difference between the growth rate r(C, m) and the optimal
    growth rate r_max(C).

    Parameter
    ---------
    C : array-like.
        substrate concentration.
    mRNA : array-like.
        mRNA copy number.
    r_gamma : float.
        average number of mRNA in the unregulated promoter.
    delta : float.
        growth benefit per substrate cleaved per enzyme.
    Ks : float.
        Monod constant for half maximum growth rate.
    eta_o : float.
        Parameter of the cost function
    M : float.
        Parameter of the cost function

    Returns
    -------
    growth_diff : array-like.
        difference between relative growth rate and the maximum relative growth
        rate.
    '''
    # Normalize the expression level with respect to the unregulated reference.
    mRNA = mRNA / r_gamma

    # Compute the non-optimal growth rate
    rC_max = growth_optimal(C, delta, Ks, eta_o, M)
    rCm = benefit_func(mRNA, C, delta, Ks) - \
          cost_func(mRNA, eta_o, M)

    # Return the difference between these two growth rates plus
    # a small amount to avoid zeros in further calculations
    return (rC_max - rCm) +1E-5

#==============================================================================

def growth_diff_logC(logC, mRNA, r_gamma, delta=0.17, Ks=0.4, eta_o=0.02, M=1.8):
    '''
    Returns the difference between the growth rate r(logC, m) and the optimal
    growth rate r_max(logC).

    Parameter
    ---------
    logC : array-like.
        log substrate concentration.
    mRNA : array-like.
        mRNA copy number.
    r_gamma : float.
        average number of mRNA in the unregulated promoter.
    delta : float.
        growth benefit per substrate cleaved per enzyme.
    Ks : float.
        Monod constant for half maximum growth rate.
    eta_o : float.
        Parameter of the cost function
    M : float.
        Parameter of the cost function

    Returns
    -------
    growth_diff : array-like.
        difference between relative growth rate and the maximum relative growth
        rate.
    '''
    # Convert from log to linear scale.
    C = np.power(10, logC)

    # Normalize the expression level with respect to the unregulated reference.
    mRNA = mRNA / r_gamma

    # Compute the non-optimal growth rate
    rC_max = growth_optimal(C, delta, Ks, eta_o, M)
    rCm = benefit_func(mRNA, C, delta, Ks) - \
          cost_func(mRNA, eta_o, M)

    # Return the difference between these two growth rates plus
    # a small amount to avoid zeros in further calculations
    return (rC_max - rCm) +1E-5

#==============================================================================

def avg_growth_opt_C(C, PC_fun, delta=0.17, Ks=0.4, eta_o=0.02, M=1.8):
    '''
    Returns the optimal growth rate at concentration C by obtaining the
    optimal expression level as defined in Dekel and Alon's paper and
    evaluating it in the fitness landscape function.
    NOTE:
    Performs the calculation with a simple sum.
    Parameter
    ---------
    C : array-like.
        substrate concentration.
    PC_fun : function.
        function to determine the probability distribution of each of the
        elements of C.
    delta : float.
        growth benefit per substrate cleaved per enzyme.
    Ks : float.
        Monod constant for half maximum growth rate.
    eta_o : float.
        Parameter of the cost function
    M : float.
        Parameter of the cost function

    Returns
    -------
    optimal_growth : array-like
        maximum relative growth rate at concentration(s) C.
    '''
    # obtain the optimal expression level at concnetration(s) C
    mRNA_opt = p_opt(C, delta, Ks, eta_o, M)

    # compute the optimal growth rate
    growth_opt = benefit_func(mRNA_opt, C, delta, Ks) - \
                 cost_func(mRNA_opt, eta_o, M)

    # compute P(C)
    pC = PC_fun(C) / PC_fun(C).sum()

    # return the average optimal growth rate
    return np.sum(pC * growth_opt)

#==============================================================================

def avg_growth_opt_logC(logC, PC_fun, delta=0.17, Ks=0.4, eta_o=0.02, M=1.8):
    '''
    Returns the optimal growth rate at concentration C by obtaining the
    optimal expression level as defined in Dekel and Alon's paper and
    evaluating it in the fitness landscape function.
    NOTE:
    Performs the calculation with a simple sum.
    Parameter
    ---------
    logC : array-like.
        log substrate concentration.
    PC_fun : function.
        function to determine the probability distribution of each of the
        elements of C.
    delta : float.
        growth benefit per substrate cleaved per enzyme.
    Ks : float.
        Monod constant for half maximum growth rate.
    eta_o : float.
        Parameter of the cost function
    M : float.
        Parameter of the cost function

    Returns
    -------
    optimal_growth : array-like
        maximum relative growth rate at concentration(s) C.
    '''
    # Convert from log to linear scale.
    C = np.power(10, logC)

    # obtain the optimal expression level at concnetration(s) C
    mRNA_opt = p_opt(C, delta, Ks, eta_o, M)

    # compute the optimal growth rate
    growth_opt = benefit_func(mRNA_opt, C, delta, Ks) - \
                 cost_func(mRNA_opt, eta_o, M)

    # compute P(C)
    pC = PC_fun(C) / PC_fun(C).sum()

    # return the average optimal growth rate
    return np.sum(pC * growth_opt)

#==============================================================================

def avg_growth_opt_int_C(C, PC_fun, delta=0.17, Ks=0.4, eta_o=0.02, M=1.8):
    '''
    Returns the optimal growth rate at concentration C by obtaining the
    optimal expression level as defined in Dekel and Alon's paper and
    evaluating it in the fitness landscape function.
    NOTE:
    Performs the calculation using the Simpson's integration rule.
    Parameter
    ---------
    C : array-like.
        substrate concentration.
    PC_fun : function.
        function to determine the probability distribution of each of the
        elements of C.
    delta : float.
        growth benefit per substrate cleaved per enzyme.
    Ks : float.
        Monod constant for half maximum growth rate.
    eta_o : float.
        Parameter of the cost function
    M : float.
        Parameter of the cost function

    Returns
    -------
    optimal_growth : array-like
        maximum relative growth rate at concentration(s) C.
    '''
    # obtain the optimal expression level at concnetration(s) C
    mRNA_opt = p_opt(C, delta, Ks, eta_o, M)

    # compute the optimal growth rate
    growth_opt = benefit_func(mRNA_opt, C, delta, Ks) - \
                 cost_func(mRNA_opt, eta_o, M)

    # compute P(C)
    pC = PC_fun(C)

    # return the average optimal growth rate
    return scipy.integrate.simps(pC * growth_opt, x=C)

#==============================================================================

def avg_growth_opt_int_logC(logC, PC_fun, delta=0.17, Ks=0.4,
                            eta_o=0.02, M=1.8):
    '''
    Returns the optimal growth rate at concentration C by obtaining the
    optimal expression level as defined in Dekel and Alon's paper and
    evaluating it in the fitness landscape function.
    NOTE:
    Performs the calculation using the Simpson's integration rule.
    Parameter
    ---------
    logC : array-like.
        log substrate concentration.
    PC_fun : function.
        function to determine the probability distribution of each of the
        elements of C.
    delta : float.
        growth benefit per substrate cleaved per enzyme.
    Ks : float.
        Monod constant for half maximum growth rate.
    eta_o : float.
        Parameter of the cost function
    M : float.
        Parameter of the cost function

    Returns
    -------
    optimal_growth : array-like
        maximum relative growth rate at concentration(s) C.
    '''
    # Convert from log to linear scale.
    C = np.power(10, logC)

    # obtain the optimal expression level at concnetration(s) C
    mRNA_opt = p_opt(C, delta, Ks, eta_o, M)

    # compute the optimal growth rate
    growth_opt = benefit_func(mRNA_opt, C, delta, Ks) - \
                 cost_func(mRNA_opt, eta_o, M)

    # compute P(C)
    pC = PC_fun(logC)

    # return the average optimal growth rate
    return scipy.integrate.simps(pC * growth_opt, x=logC)

#==============================================================================
# 06_exploring_fitness_landscape
#==============================================================================

def param_unif_sampling(param, param_range, n=1, logscale=False):
    '''
    Takes a dictionary containing all the parameters in the model used to compute
    the theoretical P(m|C), chooses n at random and changes them into a random
    value withing the limits set in the param_ranges dictionary.

    Parameters
    ----------
    param : dictionary.
        dictionary containing the parameters for the function that computes
        the probability P(m|C). The parameters are
        rep : float.
            repressor copy number per cell.
        ka, ki : float.
            dissociation constants for the active and inactive states respectively
            in the MWC model of the lac repressor.
        omega : float.
            energetic barrier between the inactive and the active state.
        kon : float.
            rate of activation of the promoter in the chemical master equation
        gamma : float.
            half-life time for the mRNA.
        r_gamma : float.
            average number of mRNA in the unregulated promoter.
    param_range : dictionary
        dictionary with the same key names as param containing the ranges on
        which the parameters can change
    n : int.
        number of parameters to sample per round.
    logscale : bool.
        boolean indicating if the sampling in the param_range should be done
        uniformly over log scale rather than lineal scale.

    Returns
    -------
    param_modified : dictionary.
        dictionary with n of the parameters changed from the input param dict.
    '''
    # copy the parameters into a new dictionary to be modified
    new_param = param.copy()

    # list the keys of the param dictionary without listin k0
    # it is a convoluted way, but I didn't find a better way to do it
    keys = np.array(list(param.keys()))[[i for i,x in enumerate(param.keys())\
                                         if x != 'k0']]

    # sample at random n parameters that will be modified
    rnd_keys = np.random.choice(keys, size=n, replace=False)

    if logscale:
        for key in rnd_keys:
            pmin, pmax = np.log(param_range[key])
            new_param[key] = np.exp(random.uniform(pmin, pmax))
    else:
        for key in rnd_keys:
            pmin, pmax = param_range[key]
            new_param[key] = random.uniform(pmin, pmax)

    return new_param

#==============================================================================

def in_silico_evo_growth(C, mRNA_range, PC_fun, logPmC_fun, fitness_fun,
                         mastereq_param, mastereq_range, growth_param,
                         logsampling=False, nparam=1, nsteps=1000, info=500):
    '''
    Function that samples parameter space for nsteps saving the list of parameters
    that strictly increase the average growth rate of the strains.

    Parameters
    ----------
    C : array-like.
        discretized values of the concentration at which the numerical
        integral will be evaluated.
    mRNA_range : array-like.
        range (IN UNITS OF THE MEAN r_gamma) of the mRNA that should be
        considered on each set of parameters. For example:
            if mRNA_range=[0, 1.5, 100] on each cycle the mRNA used for the
            computation will be np.linspace(0 * r_gamma, 1.5 * r_gamma, 100)
    PC_fun : function.
        function to determine the probability distribution of each of the
        elements of C.
        NOTE: When applied to C it must return an array of the same
        length.
    logPmC_fun : function.
        function to determine the conditional distribution logP(m|C). This in
        general will be one of the versions of the chemical master equation
        solutions, but it can be extended to any input-outpu function one
        wants to use.
        NOTE: When applied to C and mRNA it must return an array of the same
        length.
    fitness_fun : function.
        function to determine the growth rate given an expression level mRNA
        and a substrate concentration C.
    mastereq_param : dictionary.
        dictionary containing all the parameters to compute the mRNA
        distribution with the chemical-master equations approach.
        the parameters are:
        rep : float.
            repressor copy number per cell.
        ka, ki : float.
            dissociation constants for the active and inactive states
            respectively in the MWC model of the lac repressor.
        omega : float.
            energetic barrier between the inactive and the active state.
        kon : float.
            rate of activation of the promoter in the chemical master equation.
        k0 : float.
            diffusion limited rate of a repressor binding the promoter
        gamma : float.
            half-life time for the mRNA.
        r_gamma : float.
            average number of mRNA in the unregulated promoter.
    mastereq_range : dictionary
        dictionary with the same key names as param containing the ranges on
        which the parameters can change.
    growth_param : dictionary:
        dictionary containing all the parameters for the fitness function.
        the parameters are.
        r_gamma : float.
            average number of mRNA in the unregulated promoter.
        delta : float.
            growth benefit per substrate cleaved per enzyme.
        Ks : float.
            Monod constant for half maximum growth rate.
        eta_o : float.
            Parameter of the cost function
        M : float.
            Parameter of the cost function
    logsampling : bool.
        boolean indicating if parameter space should be sampled in log scale
        or not
    nparam : int.
        number of parameters to randomly change on every step
    nsteps : int.
        number of cycles to loop through.
    info : int.
        number to indicate every how many steps to print the step that
        the algorithm is running.

    Returns
    -------
    param_list : list.
        list of dictionaries containing the parameters that sequentially
        increased the fitness.
    avg_growth_rate : array.
        list of increasing average growth rates given the parameters in
        param_list.
    '''
    # extract the components from mRNA_range
    m_min, m_max, m_num = mRNA_range
    mRNA = np.linspace(mastereq_param['r_gamma'] * m_min,
                       mastereq_param['r_gamma'] * m_max,
                       m_num)
    # initialize the outputs with the wild-type values
    param_list = list()
    param_list.append(mastereq_param)
    avg_growth_rate = \
    np.array(avg_growth(C, mRNA, PC_fun, logPmC_fun, fitness_fun,
                        mastereq_param, growth_param))

    # loop nsteps sampling parameter space
    for i in np.arange(nsteps):
        if i%info == 0:
            print(i)
        # take the last updated list of parameters to start with
        param = param_list[-1]
        # get a new set of parameters by randomly sampling parameter space
        new_param = param_unif_sampling(param, mastereq_range,
                                        n=nparam, logscale=logsampling)

        # compute the new growth rate given this new parameter set
        new_growth_rate = \
        avg_growth(C, mRNA, PC_fun, logPmC_fun, fitness_fun,
                   new_param, growth_param)
        # save parameter set iff the new growth rate is greater than the
        # growth rate with the original parameter set
        if new_growth_rate > avg_growth_rate.max():
            param_list.append(new_param)
            avg_growth_rate = np.append(avg_growth_rate, new_growth_rate)
    return param_list, avg_growth_rate

#==============================================================================

def in_silico_evo_info(C, mRNA_range, PC_fun, logPmC_fun,
                       mastereq_param, mastereq_range,
                       logsampling=False, nparam=1, nsteps=1000, info=500):
    '''
    Function that samples parameter space for nsteps saving the list of parameters
    that strictly increase the average growth rate of the strains.

    Parameters
    ----------
    C : array-like.
        discretized values of the concentration at which the numerical
        integral will be evaluated.
    mRNA_range : array-like.
        range (IN UNITS OF THE MEAN r_gamma) of the mRNA that should be
        considered on each set of parameters. For example:
            if mRNA_range=[0, 1.5, 100] on each cycle the mRNA used for the
            computation will be np.linspace(0 * r_gamma, 1.5 * r_gamma, 100)
    PC_fun : function.
        function to determine the probability distribution of each of the
        elements of C.
        NOTE: When applied to C it must return an array of the same
        length.
    logPmC_fun : function.
        function to determine the conditional distribution logP(m|C). This in
        general will be one of the versions of the chemical master equation
        solutions, but it can be extended to any input-outpu function one
        wants to use.
        NOTE: When applied to C and mRNA it must return an array of the same
        length.
    mastereq_param : dictionary.
        dictionary containing all the parameters to compute the mRNA
        distribution with the chemical-master equations approach.
        the parameters are:
        rep : float.
            repressor copy number per cell.
        ka, ki : float.
            dissociation constants for the active and inactive states
            respectively in the MWC model of the lac repressor.
        omega : float.
            energetic barrier between the inactive and the active state.
        kon : float.
            rate of activation of the promoter in the chemical master equation.
        k0 : float.
            diffusion limited rate of a repressor binding the promoter
        gamma : float.
            half-life time for the mRNA.
        r_gamma : float.
            average number of mRNA in the unregulated promoter.
    mastereq_range : dictionary
        dictionary with the same key names as param containing the ranges on
        which the parameters can change.
    logsampling : bool.
        boolean indicating if parameter space should be sampled in log scale
        or not
    nparam : int.
        number of parameters to randomly change on every step
    nsteps : int.
        number of cycles to loop through.
    info : int.
        number to indicate every how many steps to print the step that
        the algorithm is running.

    Returns
    -------
    param_list : list.
        list of dictionaries containing the parameters that sequentially
        increased the fitness.
    avg_growth_rate : array.
        list of increasing average growth rates given the parameters in
        param_list.
    '''
    # extract the components from mRNA_range
    m_min, m_max, m_num = mRNA_range
    mRNA = np.linspace(mastereq_param['r_gamma'] * m_min,
                       mastereq_param['r_gamma'] * m_max,
                       m_num)
    # initialize the outputs with the wild-type values
    param_list = list()
    param_list.append(mastereq_param)
    avg_mutual_info = \
    np.array(mutual_info(C, mRNA, PC_fun, logPmC_fun, mastereq_param))

    # loop nsteps sampling parameter space
    for i in np.arange(nsteps):
        if i%info == 0:
            print(i)
        # take the last updated list of parameters to start with
        param = param_list[-1]
        # get a new set of parameters by randomly sampling parameter space
        new_param = param_unif_sampling(param, mastereq_range,
                                        n=nparam, logscale=logsampling)

        # compute the new growth rate given this new parameter set
        new_mutual_info = \
        mutual_info(C, mRNA, PC_fun, logPmC_fun, new_param)
        # save parameter set iff the new growth rate is greater than the
        # growth rate with the original parameter set
        if new_mutual_info > avg_mutual_info.max():
            param_list.append(new_param)
            avg_mutual_info = np.append(avg_mutual_info, new_mutual_info)
    return param_list, avg_mutual_info

#==============================================================================
# 07_populating_inf_fitness_plane
#==============================================================================

def rand_param_sets(param_range, n=10, logscale=True):
    '''
    Generates n random sets of parameters constrained by the limits given in
    the dictionary param_range.

    Parameters
    ----------
    param_range : dictionary.
        dictionary containig all the parameters to sample from and the ranges
        from which draw the samples
    n : int.
        number of random sets to generate.
    logscale : bool.
        boolean indicating if the sampling should be done in log scale
    Returns
    -------
    df_param : pandas DataFrame.
        returns a pandas DataFrame containing all the random set of parameters.
    '''
    param_list = list()

    if logscale:
        for i in np.arange(n):
            new_param = dict()
            for key in param_range.keys():
                pmin, pmax = np.log(param_range[key])
                new_param[key] = np.exp(random.uniform(pmin, pmax))
            param_list.append(new_param)
    else:
        for i in np.arange(n):
            new_param = dict()
            for key in param_range.keys():
                pmin, pmax = param_range[key]
                new_param[key] = random.uniform(pmin, pmax)
            param_list.append(new_param)

    return pd.DataFrame.from_dict(param_list)

#=============================================================================== 
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
