# -*- coding: utf-8 -*-
"""
Title:
    chann_cap_utils
Last update:
    2018-04-26
Author(s):
    Manuel Razo-Mejia
Purpose:
    This file is a compilation of the funtions developed for the channel
    capacity project. Most of the functions found here can also be found
    in different iPython notebooks, but in order to break down those
    notebooks into shorter and more focused notebooks it is necessary to
    call some functions previously defined.
"""

# =============================================================================
# Libraries to work with objects saved in memory
import dill
# Our numerical workhorses
import numpy as np
import scipy as sp
import scipy.optimize
import scipy.special
import scipy.integrate
import mpmath
import pandas as pd

# Import library to perform maximum entropy fits
from maxentropy.skmaxent import FeatureTransformer, MinDivergenceModel

# Import libraries to parallelize processes
from joblib import Parallel, delayed

# Import random library to make random sampling of parameters
import random

# Import plotting utilities
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

# Seaborn, useful for graphics
import seaborn as sns

# =============================================================================
# Generic themrodynamic functions
# =============================================================================


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

# =============================================================================
# chemical_master_eq_analytic_mRNA
# =============================================================================


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
    return 1.66 / 1 * k0 * 4.6E6 * np.exp(epsilon)

# =============================================================================


# define a np.frompyfunc that allows us to evaluate the sympy.mp.math.hyp1f1
np_log_hyp = np.frompyfunc(lambda x, y, z:
                           mpmath.ln(mpmath.hyp1f1(x, y, z, zeroprec=80)), 3, 1)


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

# =============================================================================
# chemical_masater_eq_analytic_protein
# =============================================================================


# define a np.frompyfunc that allows us to evaluate the sympy.mp.math.hyp1f1
np_log_gauss_hyp = np.frompyfunc(lambda a, b, c, z:
                                 mpmath.ln(mpmath.hyp2f1(a, b, c, z,  maxprec=60)).real, 4, 1)


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
    a = r_gamma_m * gamma_m / gamma_p  # r_m / gamma_p
    b = r_gamma_p * gamma_p / gamma_m  # r_p / gamma_m
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

# ==============================================================================


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
        +- Tolerance allowed for the normalization. The distribution is
         considered
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

# ==============================================================================


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


# =============================================================================
# chemical_master_mRNA_FISH_mcmc
# =============================================================================
# define a np.frompyfunc that allows us to evaluate the sympy.mp.math.hyp1f1
np_log_hyp = np.frompyfunc(lambda x, y, z:
                           mpmath.ln(mpmath.hyp1f1(x, y, z, zeroprec=1000)), 3, 1)


def log_p_m_unreg(mRNA, kp_on, kp_off, gm, rm):
    '''
    Computes the log probability lnP(m) for an unregulated promoter,
    i.e. the probability of having m mRNA.

    Parameters
    ----------
    mRNA : float.
        mRNA copy number at which evaluate the probability.
    kp_on : float.
        rate of activation of the promoter in the chemical master equation
    kp_off : float.
        rate of deactivation of the promoter in the chemical master equation
    gm : float.
        1 / half-life time for the mRNA.
    rm : float.
        production rate of the mRNA

    Returns
    -------
    log probability lnP(m)
    '''
    # Convert the mRNA copy number to a  numpy array
    mRNA = np.array(mRNA)

    # Compute the probability
    lnp = scipy.special.gammaln(kp_on / gm + mRNA) \
        - scipy.special.gammaln(mRNA + 1) \
        - scipy.special.gammaln((kp_off + kp_on) / gm + mRNA) \
        + scipy.special.gammaln((kp_off + kp_on) / gm) \
        - scipy.special.gammaln(kp_on / gm) \
        + mRNA * np.log(rm / gm) \
        + np_log_hyp(kp_on / gm + mRNA,
                     (kp_off + kp_on) / gm + mRNA, -rm / gm)

    return lnp.astype(float)


# =============================================================================
# chemical_master_moments_mRNA
# =============================================================================
# Import two-state mRNA moments
# Parameters are feed in the following order:
# (kp_on, kp_off, rm, gm)
with open('../../tmp/two_state_mRNA_lambdify.dill', 'rb') as file:
    first_unreg_m = dill.load(file)
    second_unreg_m = dill.load(file)
    third_unreg_m = dill.load(file)


# Import two-state mRNA moments
# Parameters are feed in the following order:
# (kr_on, kr_off, kp_on, kp_off, rm, gm)
with open('../../tmp/three_state_mRNA_lambdify.dill', 'rb') as file:
    first_reg_m = dill.load(file)
    second_reg_m = dill.load(file)
    third_reg_m = dill.load(file)


# =============================================================================
# chemical_master_moments_protein
# =============================================================================
# Import two-state protein moments
# Parameters are feed in the following order:
# (kp_on, kp_off, rm, gm, rp, gp)
with open('../../tmp/two_state_protein_lambdify.dill', 'rb') as file:
    first_unreg_p = dill.load(file)
    second_unreg_p = dill.load(file)
    third_unreg_p = dill.load(file)
    mp_unreg_p = dill.load(file)
    m2p_unreg_p = dill.load(file)
    mp2_unreg_p = dill.load(file)

# Import two-state protein moments
# Parameters are feed in the following order:
# (kr_on, kr_off, kp_on, kp_off, rm, gm, rp, gp)
with open('../../tmp/three_state_protein_lambdify.dill', 'rb') as file:
    first_reg_p = dill.load(file)
    second_reg_p = dill.load(file)
    third_reg_p = dill.load(file)
    mp_reg_p = dill.load(file)
    m2p_reg_p = dill.load(file)
    mp2_reg_p = dill.load(file)

# =============================================================================
# MaxEnt_approx_mRNA
# =============================================================================


def kr_off_fun(eRA, k0, kp_on, kp_off, Nns=4.6E6):
    '''
    Returns the off rate of the repressor as a function of the stat. mech.
    binding energy and the RNAP on and off rates
    Parameters
    ----------
    eRA : float.
        Repressor binding energies [kbT]
    k0 : float.
        Diffusion limited constant [s**-1 nM**-1]
    kp_on : float.
        RNAP on rate. [time**-1]
    kp_off : float.
        RNAP off rate. [time**-1]
    Nns : float.
        Number of non-specific binding sites
    Returns
    -------
    Repressor off rate
    '''
    return 1.66 * k0 * Nns * np.exp(eRA) * kp_off / (kp_off + kp_on)

# =============================================================================


def moment_reg_m(moment, C, rep, eRA,
                 k0=2.7E-3, kp_on=5.5, kp_off=28.9, rm=87.6, gm=1,
                 Nns=4.6E6, ka=139, ki=0.53, epsilon=4.5):
    '''
    Computes the steady-state mRNA distribution moments as a function of the
    parameters in the master equation for the three-state regulated promoter.

    Parameters
    ----------
    moment : string.
        Moment to be computed. Options: 'first', 'second', 'third'.
    C : array-like.
        Concentration at which evaluate the probability.
    rep: float.
        repressor copy number per cell.
    eRA : float.
        Repressor binding energy [kBT]
    rm : float.
        transcription initiation rate. [time**-1]
    gm : float.
        mRNA degradation rate. [time**-1]
    k0 : float.
        diffusion limited rate of a repressor binding the promoter
    kp_on : float.
        RNAP on rate. [time**-1]
    kp_off : float.
        RNAP off rate. [time**-1]
    Nns : float.
        Number of non-specific binding sites
    ki, ka : float.
        dissociation constants for the inactive and active states respectively
        in the MWC model of the lac repressor.
    epsilon : float.
        energetic barrier between the inactive and the active state.

    Returns
    -------
    mRNA copy number moment
    '''
    # Convert C into np.array
    C = np.array(C)

    # Calculate the repressor on rate including the MWC model
    kr_on = k0 * rep * p_act(C, ka, ki, epsilon)
    # Compute the repressor off-rate based on the on-rate and the binding energy
    kr_off = kr_off_fun(eRA, k0, kp_on, kp_off, Nns)

    if moment == 'first':
        return first_reg_m(kr_on, kr_off, kp_on, kp_off, rm, gm)
    elif moment == 'second':
        return second_reg_m(kr_on, kr_off, kp_on, kp_off, rm, gm)
    elif moment == 'third':
        return third_reg_m(kr_on, kr_off, kp_on, kp_off, rm, gm)
    else:
        print('please specify first, second or third moment.')

# =============================================================================


def maxent_reg_m_ss(constraint_dict, samplespace, C, rep, eRA,
                    k0=2.7E-3, kp_on=5.5, kp_off=28.9, rm=87.6, gm=1,
                    Nns=4.6E6, ka=139, ki=0.53, epsilon=4.5,
                    algorithm='Powell', disp=False):
    '''
    Computes the steady-state mRNA MaxEnt distribution approximation as a
    function of all the parameters that go into the chemical master equation.

    Parameters
    ----------
    constraint_dict : dictionary.
        Dictionary containing the functions to compute the constraints.
        The name of the entries should be the same as the name of the moments,
        for example constraint_dict = {'first' : first}.
    samplespace : array-like.
        Bins to be evaluated in the maximum entropy approach.
    C : array-like.
        Concentrations at which evaluate the probability.
    rep: float.
        repressor copy number per cell.
    eRA : float.
        Repressor binding energy [kBT]
    k0 : float.
        diffusion limited rate of a repressor binding the promoter
    kp_on : float.
        RNAP on rate. [time**-1]
    kp_off : float.
        RNAP off rate. [time**-1]
    rm : float.
        transcription initiation rate. [time**-1]
    gm : float.
        mRNA degradation rate. [time**-1]
    Nns : float.
        Number of non-specific binding sites
    ki, ka : float.
        dissociation constants for the inactive and active states respectively
        in the MWC model of the lac repressor.
    epsilon : float.
        energetic barrier between the inactive and the active state.
    algorithm : str.
        Algorithm to be used for the parameter optimization. See
        maxentropy.BaseModel help for a list of the available algorithms.
    disp : bool.
        Boolean indicating if the function should display the concentration
        which is computing at the moment

    Returns
    -------
    max_ent_dist : array. shape = len(C) x len(samplespace)
        Maximum Entropy probability distribution of mRNA for each concentration
        in C
    '''
    # Initialize matrix to save distributions
    max_ent_dist = np.zeros([len(C), len(samplespace)])
    # Loop through concentrations
    for j, c in enumerate(C):
        if disp:
            print(c)
        # Initialize list to save constraints and moments
        const_fn = []
        const_name = []
        # Extract each constraint function and element into lists
        for key, val in constraint_dict.items():
            const_name.append(key)
            const_fn.append(val)

        # Initialize array to save moment values
        moments = np.zeros(len(const_name))
        # Compute the value of the moments given the constraints
        for i, moment in enumerate(const_name):
            moments[i] = moment_reg_m(moment, c, rep, eRA,
                                      k0, kp_on, kp_off, rm, gm,
                                      Nns, ka, ki, epsilon)

        # Define the minimum entropy moel
        model = MinDivergenceModel(const_fn, samplespace, algorithm=algorithm)
        # Change the dimensionality of the moment array
        X = np.reshape(moments, (1, -1))
        # Fit the model
        model.fit(X)
        max_ent_dist[j, :] = model.probdist()

    # Return probability distribution
    return max_ent_dist

# =============================================================================
# MaxEnt_approx_protein
# =============================================================================


def moment_reg_p(moment, C, rep, eRA,
                 k0=2.7E-3, kp_on=5.5, kp_off=28.9, rm=87.6, gm=1,
                 rp=0.0975, gp=97.53,
                 Nns=4.6E6, ka=139, ki=0.53, epsilon=4.5):
    '''
    Computes the protein distribution moments as a function  of all the
    parameters that go into the chemical master equation.

    Parameters
    ----------
    moment : string.
        Moment to be computed. Options: 'first', 'second' and 'third'.
    C : array-like.
        Concentration at which evaluate the probability.
    rep: float.
        repressor copy number per cell.
    eRA : float.
        Repressor binding energy [kBT]
    rm : float.
        transcription initiation rate. [time**-1]
    gm : float.
        mRNA degradation rate. [time**-1]
    rp : float.
        translation initiation rate. [time**-1]
    gp : float.
        protein degradation rate. [time**-1]
    k0 : float.
        diffusion limited rate of a repressor binding the promoter
    kp_on : float.
        RNAP on rate. [time**-1]
    kp_off : float.
        RNAP off rate. [time**-1]
    Nns : float.
        Number of non-specific binding sites
    ki, ka : float.
        dissociation constants for the inactive and active states respectively
        in the MWC model of the lac repressor.
    epsilon : float.
        energetic barrier between the inactive and the active state.

    Returns
    -------
    protein copy number moment
    '''
    # Convert C into np.array
    C = np.array(C)

    # Calculate the repressor on rate including the MWC model
    kr_on = k0 * rep * p_act(C, ka, ki, epsilon)
    # Compute the repressor off-rate based on the on-rate and the binding energy
    kr_off = kr_off_fun(eRA, k0, kp_on, kp_off, Nns)

    if moment == 'first':
        return first_reg_p(kr_on, kr_off, kp_on, kp_off, rm, gm, rp, gp)
    elif moment == 'second':
        return second_reg_p(kr_on, kr_off, kp_on, kp_off, rm, gm, rp, gp)
    elif moment == 'third':
        return third_reg_p(kr_on, kr_off, kp_on, kp_off, rm, gm, rp, gp)
    else:
        print('please specify first, second or third moment')

# =============================================================================


def maxent_reg_p_ss(constraint_dict, samplespace, C, rep, eRA,
                    k0=2.7E-3, kp_on=5.5, kp_off=28.9, rm=87.6, gm=1,
                    rp=0.0975, gp=97.53,
                    Nns=4.6E6, ka=139, ki=0.53, epsilon=4.5,
                    algorithm='Powell', disp=False):
    '''
    Computes the steady-state MaxEnt distribution approximation as a function
    of all the parameters that go into the chemical master equation.

    Parameters
    ----------
    constraint_dict : dictionary.
        Dictionary containing the functions to compute the constraints.
        The name of the entries should be the same as the name of the moments,
        for example constraint_dict = {'first' : first}.
    samplespace : array-like.
        Bins to be evaluated in the maximum entropy approach.
    C : array-like.
        Concentrations at which evaluate the probability.
    rep: float.
        repressor copy number per cell.
    eRA : float.
        Repressor binding energy [kBT]
    k0 : float.
        diffusion limited rate of a repressor binding the promoter
    kp_on : float.
        RNAP on rate. [time**-1]
    kp_off : float.
        RNAP off rate. [time**-1]
    rm : float.
        transcription initiation rate. [time**-1]
    gm : float.
        mRNA degradation rate. [time**-1]
    rp : float.
        translation initiation rate. [time**-1]
    gp : float.
        protein degradation rate. [time**-1]
    Nns : float.
        Number of non-specific binding sites
    ki, ka : float.
        dissociation constants for the inactive and active states respectively
        in the MWC model of the lac repressor.
    epsilon : float.
        energetic barrier between the inactive and the active state.
    algorithm : str.
        Algorithm to be used for the parameter optimization. See
        maxentropy.BaseModel help for a list of the available algorithms.
    disp : bool.
        Boolean indicating if the function should display the concentration
        which is computing at the moment

    Returns
    -------
    max_ent_dist : array. shape = len(C) x len(samplespace)
        Maximum Entropy probability distribution of protein for each
        concentration in C
    '''
    # Initialize matrix to save distributions
    max_ent_dist = np.zeros([len(C), len(samplespace)])
    # Loop through concentrations
    for j, c in enumerate(C):
        if disp:
            print(c)
        # Initialize list to save constraints and moments
        const_fn = []
        const_name = []
        # Extract each constraint function and element into lists
        for key, val in constraint_dict.items():
            const_name.append(key)
            const_fn.append(val)

        # Initialize array to save moment values
        moments = np.zeros(len(const_name))
        # Compute the value of the moments given the constraints
        for i, moment in enumerate(const_name):
            moments[i] = moment_reg_p(moment, c, rep, eRA,
                                      k0, kp_on, kp_off, rm, gm, rp, gp,
                                      Nns, ka, ki, epsilon)

        # Define the minimum entropy moel
        model = MinDivergenceModel(const_fn, samplespace, algorithm=algorithm)
        # Change the dimensionality of the moment array
        X = np.reshape(moments, (1, -1))
        # Fit the model
        model.fit(X)
        max_ent_dist[j, :] = model.probdist()

    # Return probability distribution
    return max_ent_dist

# ==============================================================================
# MaxEnt_approx_joint
# ==============================================================================


def moment_ss_reg(moment_fun, C, rep, eRA,
                  k0=2.7E-3, kp_on=5.5, kp_off=28.9, rm=87.6, gm=1,
                  rp=0.0975, gp=97.53,
                  Nns=4.6E6, ka=139, ki=0.53, epsilon=4.5):
    '''
    Computes the mRNA and/or protein steady state moments given a list
    of functions (moments) and all the chemical master equation
    parameters.

    Parameters
    ----------
    moment_fun : list.
        List containing the functions to be used to compute the steady
        state moments.
    C : array-like.
        Concentration at which evaluate the probability.
    rep: float.
        repressor copy number per cell.
    eRA : float.
        Repressor binding energy [kBT]
    rm : float.
        transcription initiation rate. [time**-1]
    gm : float.
        mRNA degradation rate. [time**-1]
    rp : float.
        translation initiation rate. [time**-1]
    gp : float.
        protein degradation rate. [time**-1]
    k0 : float.
        diffusion limited rate of a repressor binding the promoter
    kp_on : float.
        RNAP on rate. [time**-1]
    kp_off : float.
        RNAP off rate. [time**-1]
    Nns : float.
        Number of non-specific binding sites
    ki, ka : float.
        dissociation constants for the inactive and active states respectively
        in the MWC model of the lac repressor.
    epsilon : float.
        energetic barrier between the inactive and the active state.

    Returns
    -------
    moments_num : array-like. len(C) x len(moments)
        Array containing all the required moments for each of the indicated
        concentrations of inducer
    '''
    # Convert C into np.array
    C = np.array(C)

    # Calculate the repressor on rate including the MWC model
    kr_on = k0 * rep * p_act(C, ka, ki, epsilon)

    # Compute the repressor off-rate based on the on-rate and the
    # binding energy
    kr_off = kr_off_fun(eRA, k0, kp_on, kp_off, Nns)

    # Generate array with variables
    param = [kr_on, kr_off, kp_on, kp_off, rm, gm, rp, gp]

    # Initialie array to save the moments
    moment_num = np.zeros(len(moment_fun))

    # Loop through functions to compute moments
    for i, fun in enumerate(moment_fun):
        # Find the number of variables in function. mRNA functions have
        # 6 arguments while protein functions have 8.
        arg_num = fun.__code__.co_argcount

        # Compute moment
        moment_num[i] = fun(*param[:arg_num])

    # Return moments
    return moment_num

# =============================================================================


# Functions used with the maxentropy package to fit the Lagrange multipliers of
# the MaxEnt distribution
# mRNA
def m1_fn(x):
    return x[0]


def m2_fn(x):
    return x[0]**2


def m3_fn(x):
    return x[0]**3

# protein


def p1_fn(x):
    return x[1]


def p2_fn(x):
    return x[1]**2


def p3_fn(x):
    return x[1]**3

# Cross correlations


def mp_fn(x):
    return x[0] * x[1]


def m2p_fn(x):
    return x[0]**2 * x[1]


def mp2_fn(x):
    return x[0] * x[1]**2


def feature_fn(x, x_expo):
    return x[0]**x_expo[0] * x[1]**x_expo[1]

# =============================================================================
# moment_dynamics_numeric_protein
# =============================================================================


def dpdt(mp, t, Kmat, Rm, Gm, Rp, Gp):
    '''
    function to integrate all mRNA and protein moment dynamics
    using scipy.integrate.odeint
    Parameters
    ----------
    m : array-like.
        Array containing all moments (mRNA, protein and cross correlations)
        Unregulated
        mp[0] = m0_P (RNAP bound)
        mp[1] = m0_E (Empty promoter)
        mp[2] = m1_P (RNAP bound)
        mp[3] = m1_P (Empty promoter)
        mp[4] = m2_P (RNAP bound)
        mp[5] = m2_P (Empty promoter)
        mp[6] = m3_P (RNAP bound)
        mp[7] = m3_P (Empty promoter)
        mp[8] = p1_P (RNAP bound)
        mp[9] = p1_P (Empty promoter)
        mp[10] = mp_P (RNAP bound)
        mp[11] = mp_P (Empty promoter)
        mp[12] = p2_P (RNAP bound)
        mp[13] = p2_P (Empty promoter)
        mp[14] = m2p_P (RNAP bound)
        mp[15] = m2p_P (Empty promoter)
        mp[16] = mp2_P (RNAP bound)
        mp[17] = mp2_P (Empty promoter)
        mp[18] = p3_P (RNAP bound)
        mp[19] = p3_P (Empty promoter)
        ---------
        Regulated:
        mp[0] = m0_P (RNAP bound)
        mp[1] = m0_E (Empty promoter)
        mp[2] = m0_R (Repressor bound)
        mp[3] = m1_P (RNAP bound)
        mp[4] = m1_E (Empty promoter)
        mp[5] = m1_R (Repressor bound)
        mp[6] = m2_P (RNAP bound)
        mp[7] = m2_E (Empty promoter)
        mp[8] = m2_R (Repressor bound)
        mp[9] = m3_P (RNAP bound)
        mp[10] = m3_E (Empty promoter)
        mp[11] = m3_R (Repressor bound)
        mp[12] = p1_P (RNAP bound)
        mp[13] = p1_E (Empty promoter)
        mp[14] = p1_R (Repressor bound)
        mp[15] = mp_P (RNAP bound)
        mp[16] = mp_E (Empty promoter)
        mp[17] = mp_R (Repressor bound)
        mp[18] = p2_P (RNAP bound)
        mp[19] = p2_E (Empty promoter)
        mp[20] = p2_R (Repressor bound)
        mp[21] = m2p_P (RNAP bound)
        mp[22] = m2p_E (Empty promoter)
        mp[23] = m2p_R (Repressor bound)
        mp[24] = mp2_P (RNAP bound)
        mp[25] = mp2_E (Empty promoter)
        mp[26] = mp2_R (Repressor bound)
        mp[27] = p3_P (RNAP bound)
        mp[28] = p3_E (Empty promoter)
        mp[29] = p3_R (Repressor bound)
    t : array-like.
        Time array
    Kmat : array-like.
        Matrix containing the transition rates between the promoter states.
    Rm : array-like.
        Matrix containing the mRNA production rate at each of the states.
    Gm : array-like.
        Matrix containing the mRNA degradation rate at each of the states.
    Rp : array-like.
        Matrix containing the protein production rate at each of the states.
    Gp : array-like.
        Matrix containing the protein degradation rate at each of the states.

    Returns
    -------
    dynamics of all mRNA and protein moments
    '''
    # Obtain the zeroth and first moment based on the size
    # of the Kmat matrix
    if Kmat.shape[0] == 2:
        m0 = mp[0:2]
        m1 = mp[2:4]
        m2 = mp[4:6]
        m3 = mp[6:8]
        p1 = mp[8:10]
        mp1 = mp[10:12]
        p2 = mp[12:14]
        m2p = mp[14:16]
        mp2 = mp[16:18]
        p3 = mp[18::]
    elif Kmat.shape[0] == 3:
        m0 = mp[0:3]
        m1 = mp[3:6]
        m2 = mp[6:9]
        m3 = mp[9:12]
        p1 = mp[12:15]
        mp1 = mp[15:18]
        p2 = mp[18:21]
        m2p = mp[21:24]
        mp2 = mp[24:27]
        p3 = mp[27::]

    # Initialize array to save all dynamics
    dmpdt = np.array([])

    # Compute the moment equations for the:
    # === mRNA === #
    # Zeroth moment
    dm0dt_eq = np.dot(Kmat, m0)
    dmpdt = np.append(dmpdt, dm0dt_eq)
    # <m1>
    dm1dt_eq = np.dot((Kmat - Gm), m1) + np.dot(Rm, m0)
    dmpdt = np.append(dmpdt, dm1dt_eq)
    # <m2>
    dm2dt_eq = np.dot((Kmat - 2 * Gm), m2) + np.dot((2 * Rm + Gm), m1) +\
        np.dot(Rm, m0)
    dmpdt = np.append(dmpdt, dm2dt_eq)
    # <m3>
    dm3dt_eq = np.dot((Kmat - 3 * Gm), m3) +\
        np.dot((3 * Rm + 3 * Gm), m2) +\
        np.dot((3 * Rm - Gm), m1) +\
        np.dot(Rm, m0)
    dmpdt = np.append(dmpdt, dm3dt_eq)
    # === protein and correlations === #
    # <p1>
    dp1dt_eq = np.dot((Kmat - Gp), p1) + np.dot(Rp, m1)
    dmpdt = np.append(dmpdt, dp1dt_eq)
    # <mp>
    dmpdt_eq = np.dot((Kmat - Gm - Gp), mp1) +\
        np.dot(Rm, p1) +\
        np.dot(Rp, m2)
    dmpdt = np.append(dmpdt, dmpdt_eq)
    # <p2>
    dp2dt_eq = np.dot((Kmat - 2 * Gp), p2) +\
        np.dot(Gp, p1) +\
        np.dot(Rp, m1) +\
        np.dot((2 * Rp), mp1)
    dmpdt = np.append(dmpdt, dp2dt_eq)
    # <m2p>
    dm2pdt_eq = np.dot((Kmat - 2 * Gm - Gp), m2p) +\
        np.dot(Rm, p1) +\
        np.dot((2 * Rm + Gm), mp1) +\
        np.dot(Rp, m3)
    dmpdt = np.append(dmpdt, dm2pdt_eq)
    # <mp2>
    dmp2dt_eq = np.dot((Kmat - Gm - 2 * Gp), mp2) +\
        np.dot(Rm, p2) +\
        np.dot((2 * Rp), m2p) +\
        np.dot(Rp, m2) +\
        np.dot(Gp, mp1)
    dmpdt = np.append(dmpdt, dmp2dt_eq)
    # <p3>
    dp3dt_eq = np.dot((Kmat - 3 * Gp), p3) +\
        np.dot((3 * Gp), p2) -\
        np.dot(Gp, p1) +\
        np.dot((3 * Rp), mp2) +\
        np.dot((3 * Rp), mp1) +\
        np.dot(Rp, m1)
    dmpdt = np.append(dmpdt, dp3dt_eq)

    return dmpdt

# =============================================================================


def dynamics_to_df(sol, t):
    '''
    Takes the output of the dpdt function and the vector time and returns
    a tidy pandas DataFrame with the GLOBAL moments.
    Parameters
    ----------
    sol : array-like.
        Array with 20 or 30 columns containing the dynamics of the mRNA and
        protein distribution moments.
    t : array-like.
        Time array used for integrating the differential equations
    Returns
    -------
    tidy dataframe with the GLOBAL moments
    '''
    # Define names of dataframe columns
    names = ['time', 'm1', 'm2', 'm3', 'p1', 'mp', 'p2', 'm2p', 'mp2', 'p3']

    # Initialize matrix to save global moments
    mat = np.zeros([len(t), len(names)])
    # Save time array in matrix
    mat[:, 0] = t

    # List index for columns depending on number of elements in matrix
    idx = np.arange(int(sol.shape[1] / 10), sol.shape[1],
                    int(sol.shape[1] / 10))

    # Loop through index and compute global moments
    for i, index in enumerate(idx):
        # Compute and save global moment
        mat[:, i+1] = np.sum(sol[:, int(index):int(index + sol.shape[1] / 10)],
                             axis=1)

    return pd.DataFrame(mat, columns=names)

# =============================================================================


def maxEnt_from_lagrange(mRNA, protein, lagrange,
                         exponents=[(1, 0), (2, 0), (3, 0),
                                    (0, 1), (0, 2), (1, 1)], log=False):
    '''
    Computes the mRNA and protein joint distribution P(m, p) as approximated
    by the MaxEnt methodology given a set of Lagrange multipliers.
    Parameters
    ----------
    mRNA, protein : array-like.
        Sample space for both the mRNA and the protein.
    lagrange : array-like.
        Array containing the value of the Lagrange multipliers associated
        with each of the constraints.
    exponents : list. leng(exponents) == len(lagrange)
        List containing the exponents associated with each constraint.
        For example a constraint of the form <m**3> has an entry (3, 0)
        while a constraint of the form <m * p> has an entry (1, 1).
    log : bool. Default = False
        Boolean indicating if the log probability should be returned.
    Returns
    -------
    Pmp : 2D-array. len(mRNA) x len(protein)
        2D MaxEnt distribution.
    '''
    # Generate grid of points
    mm, pp = np.meshgrid(mRNA, protein)

    # Initialize 3D array to save operations associated with each lagrange
    # multiplier
    operations = np.zeros([len(lagrange), len(protein), len(mRNA)])

    # Compute operations associated with each Lagrange Multiplier
    for i, expo in enumerate(exponents):
        operations[i, :, :] = lagrange[i] * mm**expo[0] * pp**expo[1]

    # check if the log probability should be returned
    if log:
        return np.sum(operations, axis=0) -\
            sp.misc.logsumexp(np.sum(operations, axis=0))
    else:
        return np.exp(np.sum(operations, axis=0) -
                      sp.misc.logsumexp(np.sum(operations, axis=0)))


# =============================================================================
# blahut_arimoto_channel_capacity
# =============================================================================

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
        # Compute QmC * np.log(QmC / sum_C_pC_QmC) avoiding errors with 0 and
        # neg numbers
        with np.errstate(divide='ignore', invalid='ignore'):
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

# =============================================================================


def theory_trans_matrix(df_prob, c, Rtot, tol=1E-20, clean=True, **kwargs):
    '''
    Function that builds the transition matrix Qg|c for a series of
    concentrations c. It builds the matrix by using the tidy data-frames
    containing the pre-computed distributions.
    Parameters
    ----------
    df_prob : Pandas data frame.
        Data frame containing the pre-computed distributions. The data frame
        should contain 3 columns:
        1) repressor : number of repressors.
        2) protein   : number of proteins.
        3) prob      : probability of a protein copy number.
    c : array-like.
        Concentrations at which to evaluate the input-output function.
    Rtot : int.
        Total number of repressors per cell.
    tol : float.
        tolerance under which if a marginal probability for a protein is
        lower than that, that column is discarded.
    clean : bool.
        Boolean indicating if the entire matrix should be returned or if the
        columns with cumulative probability < tol should be removed.
    kwargs : arguments to be passed to the p_act function such as
        ka, ki :  dissociation constants
        epsilon : energy difference between active and inactive state
    Returns
    -------
    Qg|c : input output matrix in which each row represents a concentration
    and each column represents the probability of a protein copy number.
    '''
    # Convert the concentration to a numpy array
    c = np.array(c)

    # compute the p_active probabilities for each concentration
    pacts = p_act(c, **kwargs)
    pacts = np.unique(pacts)
    # Compute the number of repressors given this p_active. The
    # repressors will be round down for fractional number of repressors
    repressors = np.floor(Rtot * pacts)

    # Initialize matrix to save input-output function
    Qgc = np.zeros([len(c), len(df_prob.protein.unique())])

    # Loop through every repressor and add the probabilities to each
    # row of the Qg|c matrix
    for i, rep in enumerate(repressors):
        Qgc[i, :] =\
            df_prob[df_prob.repressor == rep].sort_values(by='protein').prob

    # Conditional on whether or not to clean the matrix
    if clean:
        # Remove columns whose marginal protein probability is < tol
        prot_marginal = Qgc.sum(axis=0)
        return Qgc[:, prot_marginal > tol]
    else:
        return Qgc

# =============================================================================
# Plotting style
# =============================================================================


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

# =============================================================================
# Useful generic functions
# =============================================================================


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

# =============================================================================


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

# =============================================================================
# Plotting functions
# =============================================================================


def pmf_cdf_plot(x, px, legend_var, color_palette='Blues',
                 mean_mark=True, marker_height=0.3,
                 color_bar=True, cbar_label='', binstep=1,
                 figsize=(6, 5), title='', xlabel='', xlim=None, ylim=None):
    '''
    Custom plot of the PMF and the CDF of multiple distributions
    with a side legend.
    Parameters
    ----------
    x : array-like. 1 x N.
        X values at which the probability P(X) is being plotted
    px : array-like. M x N
        Probability of each of the values of x for different conditions
        such as varying repressor copy number, inducer concentration or
        binding energy.
    legend_var : array-like. 1 X M.
        Value of the changing variable between different distributions
        being plotted
    colors : str.
        Color palete from the seaborn options to use for the different
        distributions.
    mean_mark : bool.
        Boolean indicating if a marker should be placed to point at
        the mean of each distribution. Default=True
    marker_height : float.
        Height that all of the markers that point at the mean should
        have.
    color_bar : bool.
        Boolean indicating if a color bar should be added on the side
        to indicate the different variable between distributions.
        Default=True
    cbar_label : str.
        Side label for color bar.
    binstep : int.
        If not all the bins need to be plot it can plot every binstep
        bins. Especially useful when plotting a lot of bins.
    figsize : array-like. 1 x 2.
        Size of the figure
    title : str.
        Title for the plot.
    xlabel : str.
        Label for the x plot
    xlim : array-like. 1 x 2.
        Limits on the x-axis.
    ylim : array-like. 1 x 2.
        Limits on the y-axis for the PMF. The CDF goes from 0 to 1 by
        definition.
    '''

    colors = sns.color_palette(color_palette, n_colors=len(legend_var))

    # Initialize figure
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)
    ax[0].yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(
                                    useMathText=True,
                                    useOffset=False))

    # Loop through inducer concentrations
    for i, c in enumerate(legend_var):
        # PMF plot
        ax[0].plot(x[0::binstep], px[i, 0::binstep],
                   label=str(c), drawstyle='steps',
                   color='k')
        # Fill between each histogram
        ax[0].fill_between(x[0::binstep], px[i, 0::binstep],
                           color=colors[i], alpha=0.8, step='pre')
        # CDF plot
        ax[1].plot(x[0::binstep], np.cumsum(px[i, :])[0::binstep],
                   drawstyle='steps',
                   color=colors[i], linewidth=2)

    # Label axis
    ax[0].set_title(title)
    ax[0].set_ylabel('probability')
    ax[0].margins(0.02)
    # Set scientific notation
    ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)

    ax[1].legend(loc=0)
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel('CDF')
    ax[1].margins(0.02)

    # Declare color map for legend
    cmap = plt.cm.get_cmap(color_palette, len(legend_var))
    bounds = np.linspace(0, len(legend_var), len(legend_var) + 1)

    # Compute mean mRAN copy number from distribution
    mean_dist = [np.sum(x * prob) for prob in px]
    # Plot a little triangle indicating the mean of each distribution
    mean_plot = ax[0].scatter(mean_dist, [np.max(px) * 1.1] * len(mean_dist),
                              marker='v', s=200,
                              c=np.arange(len(mean_dist)), cmap=cmap,
                              edgecolor='k', linewidth=1.5)

    # Generate a colorbar with the concentrations
    cbar_ax = fig.add_axes([0.95, 0.25, 0.03, 0.5])
    cbar = fig.colorbar(mean_plot, cax=cbar_ax)
    cbar.ax.get_yaxis().set_ticks([])
    for j, c in enumerate(legend_var):
        cbar.ax.text(1, j / len(legend_var) + 1 / (2 * len(legend_var)),
                     c, ha='left', va='center',
                     transform=cbar_ax.transAxes, fontsize=12)
    cbar.ax.get_yaxis().labelpad = 35
    cbar.set_label(r'{:s}'.format(cbar_label))

    plt.figtext(-0.02, .9, '(A)', fontsize=18)
    plt.figtext(-0.02, .46, '(B)', fontsize=18)

    plt.subplots_adjust(hspace=0.06)

#==============================================================================


def joint_marginal_plot(x, y, Pxy,
                        xlabel='', ylabel='', title='',
                        size=5.5, ratio=5, space=0.1,
                        marginal_color='black',
                        marginal_fill=sns.color_palette('colorblind',
                                                        n_colors=1),
                        marginal_alpha=0.8,
                        joint_cmap='Blues', include_cbar=True,
                        cbar_label='probability', vmin=None, vmax=None):
    '''
    Plots the joint and marginal distributions like the seaborn jointplot.

    Parameters
    ----------
    x, y : array-like.
        Arrays that contain the values of the x and y axis. Used to set the
        ticks on the axis.
    Pxy : 2d array. len(x) x len(y)
        2D array containing the value of the joint distributions to be plot
    xlabel : str.
        X-label for the joint plot.
    ylabel : str.
        Y-label for the joint plot.
    title : str.
        Title for the entire plot.
    size : float.
        Figure size.
    ratio : float.
        Plot size ratio between the joint 2D hist and the marginals.
    space : float.
        Space beteween marginal and joint plot.
    marginal_color: str or RGB number. Default 'black'
        Color used for the line of the marginal distribution
    marginal_fill: str or RGB number. Default seaborn colorblind default
        Color used for the filling of the marginal distribution
    marginal_alpha : float. [0, 1]. Default = 0.8
        Value of alpha for the fill_between used in the marginal plot.
    joint_cmap : string. Default = 'Blues'
        Name of the color map to be used in the joint distribution.
    include_cbar : bool. Default = True
        Boolean indicating if a color bar should be included for the joint
        distribution values.
    cbar_label : str. Default = 'probability'
        Label for the color bar
    vmin, vmax : scalar, optional, default: None
        From the plt.imshow documentation:
        `vmin` and `vmax` are used in conjunction with norm to normalize
        luminance data.  Note if you pass a `norm` instance, your
        settings for `vmin` and `vmax` will be ignored.
    '''
    # Define the extent of axis and aspect ratio of heatmap
    extent = [x.min(), x.max(), y.min(), y.max()]
    aspect = (x.max() - x.min()) / (y.max() - y.min())

    # Initialize figure
    f = plt.figure(figsize=(size, size))

    # Specify gridspec
    gs = plt.GridSpec(ratio + 1, ratio + 1)

    # Generate axis
    # Joint
    ax_joint = f.add_subplot(gs[1:, :-1])

    # Marginals
    ax_marg_x = f.add_subplot(gs[0, :-1], sharex=ax_joint)
    ax_marg_y = f.add_subplot(gs[1:, -1], sharey=ax_joint)

    # Turn off tick visibility for the measure axis on the marginal plots
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    # Turn off the ticks on the density axis for the marginal plots
    plt.setp(ax_marg_x.yaxis.get_majorticklines(), visible=False)
    plt.setp(ax_marg_x.yaxis.get_minorticklines(), visible=False)
    plt.setp(ax_marg_y.xaxis.get_majorticklines(), visible=False)
    plt.setp(ax_marg_y.xaxis.get_minorticklines(), visible=False)
    plt.setp(ax_marg_x.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)
    ax_marg_x.yaxis.grid(False)
    ax_marg_y.xaxis.grid(False)

    # Set spacing between plots
    f.subplots_adjust(hspace=space, wspace=space)

    # Plot marginals
    ax_marg_x.plot(x, Pxy.sum(axis=0), drawstyle='steps', color=marginal_color)
    ax_marg_x.fill_between(x, Pxy.sum(axis=0), alpha=marginal_alpha, step='pre',
                           color=marginal_fill)
    ax_marg_y.plot(Pxy.sum(axis=1), y, drawstyle='steps', color=marginal_color)
    ax_marg_y.fill_between(Pxy.sum(axis=1), y, alpha=marginal_alpha, step='pre',
                           color=marginal_fill)

    # Set title above the ax_arg_x plot
    ax_marg_x.set_title(title)

    # Plot joint distribution
    cax = ax_joint.matshow(Pxy, cmap=joint_cmap, origin='lower',
                           extent=extent, aspect=aspect, vmin=vmin, vmax=vmax)
    # Move ticks to the bottom of the plot
    ax_joint.xaxis.tick_bottom()
    ax_joint.grid(False)

    # Label axis
    ax_joint.set_xlabel(xlabel)
    ax_joint.set_ylabel(ylabel)

    if include_cbar:
        # Generate a colorbar with the concentrations
        cbar_ax = f.add_axes([1.0, 0.25, 0.03, 0.5])

        # Add colorbar, make sure to specify tick locations to match desired ticklabels
        cbar = f.colorbar(cax, cax=cbar_ax, format='%.0E')

        # Label colorbar
        cbar.set_label(cbar_label)
