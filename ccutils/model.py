# -*- coding: utf-8 -*-
"""
Title:
    model.py
Last update:
    2018-10-22
Author(s):
    Manuel Razo-Mejia
Purpose:
    This file compiles all of the functions related to the
    theoretical model for transcriptional regulation relevant
    for the channel capacity project
"""
# Our numerical workhorses
import numpy as np
import scipy as sp
import scipy.optimize
import scipy.special
import scipy.integrate
import mpmath
import pandas as pd


# THERMODYNAMIC FUNCTIONS
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


def fold_change_statmech(C, R, eRA, ka, ki, Nns=4.6E6, epsilon=4.5,
                         logC=False):
    '''
    Computes the gene expression fold-change as expressed in the simple
    repression thermodynamic model of gene expression as a function of
    repressor copy number, repressor-DNA binding energy, and MWC parameters.

    Parameters
    ----------
    C : array-like.
        concentration(s) of ligand at which evaluate the function.
    R : array-like.
        repressor copy number per cell
    eRA : array-like.
        repressor-DNA binding energy
    ka, ki : float.
        dissociation constants for the active and inactive states respectively
        in the MWC model of the lac repressor.
    Nns : float. Default = 4.6E6
        number of non-specific binding sites in the bacterial genome.
    epsilon : float.
        energetic barrier between the inactive and the active state.
    logC : Bool.
        boolean indicating if the concentration is given in log scale

    Returns
    -------
    p_act : float.
        The probability of the repressor being in the active state.
    '''
    return (1 + R / Nns * p_act(C, ka, ki, epsilon, logC) * np.exp(-eRA))**-1


# TWO-STATE PROMOTER
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


# THREE-STATE PROMOTER 
def kr_off_fun(eRA, k0, kp_on, kp_off, Nns=4.6E6, Vcell=2.15):
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
    Vcell : float.
        Cell volume in femtoliters
    Returns
    -------
    Repressor off rate
    '''
    return 1 / Vcell / .6022 * k0 * Nns * np.exp(eRA) * \
        kp_off / (kp_off + kp_on)


# DISTRIBUTION MOMENT DYNAMICS
def dmomdt(A_mat, expo, t, mom_init, states=['E', 'P', 'R']):
    '''
    Function to integrate 
    dµ/dt = Aµ
    for any matrix A using the scipy.integrate.odeint
    function
    
    Parameters
    ----------
    A_mat : 2D-array
        Square matrix defining the moment dynamics
    expo : array-like
        List containing the moments involved in the 
        dynamics defined by A
    t : array-like
        Time array in seconds
    mom_init : array-like. lenth = A_mat.shape[1]
    states : list with strings. Default = ['E', 'P', 'R']
        List containing the name of the promoter states
    Returns
    -------
    Tidy dataframe containing the moment dynamics
    '''
    # Define a lambda function to feed to odeint that returns
    # the right-hand side of the moment dynamics
    def dt(mom, time):
        return np.dot(A_mat, mom)
    
    # Integrate dynamics
    mom_dynamics = sp.integrate.odeint(dt, mom_init, t)

    ## Save results in tidy dataframe  ##
    # Define names of columns
    names = ['m{0:d}p{1:d}'.format(*x) + s for x in expo 
             for s in states]

    # Save as data frame
    df = pd.DataFrame(mom_dynamics, columns=names)
    # Add time column
    df = df.assign(t_sec = t, t_min = t / 60)
    
    return df

