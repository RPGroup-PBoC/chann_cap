# -*- coding: utf-8 -*-
"""
Title:
    model.py
Last update:
    2019-11-02
Author(s):
    Manuel Razo-Mejia
Purpose:
    This file compiles all of the functions related to the
    theoretical model for transcriptional regulation relevant
    for the channel capacity project
"""
import pickle
import numpy as np
import scipy as sp
import scipy.optimize
import scipy.special
import scipy.integrate
import mpmath
import pandas as pd
import git


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
def rhs_dmomdt(mom, t, A):
    '''
    Function that computes the right-hand side of the moment
    dynamics equation
    dµ/dt = Aµ
    This funciton is fed to the scipy.integrate.odeint function
    Parameters
    ----------
    mom : array-like
        Array containing all of the moments included in the matrix
        dynamics A.
    t : array-like
        time array
    A : 2D-array.
        Matrix containing the linear coefficients of the moment
        dynamics equation
    Returns
    -------
    Right hand-side of the moment dynamics
    '''
    return np.dot(A, mom)

def dmomdt(A_mat, expo, t, mom_init, states=['I', 'A', 'R']):
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

def dmomdt_cycles(mom_init, t_single, t_double,
                  A_mat_fun, par_single, par_double,
                  expo, n_cycles, Z_mat,
                  n_steps=1000, states=['A', 'I']):
    '''
    Function that integrates the moment dynamics over several cell 
    cycles. The dynamics are integrated assuming a non-poisson
    protein degradation. So the protein is only degraded due to 
    cell division.
    
    Parameters
    ----------
    mom_init : array-like.
        Array containing the  initial conditions for the moment 
        of the states of the promoter.
    t_single : float.
        Time [in 1/mRNA degradation rate units] that cells spend 
        with a single promoter copy
    t_double : float.
        Time [in 1/mRNA degradation rate units] that cells spend 
        with a two promoter copies.
    A_mat_fun: function.
        Function to build the matrix moment dynamics. 
        This function takes as input the necessary rates 
        to build the matrix that defines the dynamics
        dµ/dt = A_mat * µ.
    par_single, par_double: list.
        Lists containing the rate parameters to be fed into the
        A_mat_fun function. These parameters must come in the 
        correct order that will be fed into the funciton.
        par_single = parameters for single promoter
        par_double = parameter for two promoters
    expo : array-like
        List containing the moments involved in the 
        dynamics defined by A
    n_cycles : int.
        Number of cell cycles to integrate for. A cell cycle is defined
        as t_single + t_double.
    Z_mat : array-like.
        Array containing the linear coefficients to compute the moments
        after the cell division
    n_steps : int. Default = 1000.
        Number of steps to use for the numerical integration.
    states : array-like. Default = ['A', 'I']
        Array containing the strings that define the moments that the
        promoter can be found at. For an unregulated promoter the only
        two available states are 'A' (active state) and 'E' (inactive).
        For the regulated case a third state 'R' (repressor bound) is
        available to the system.

    Returns
    -------
    distribution moment dynamics over cell cycles
    '''
    # Initialize names for moments in data frame
    names = ['m{0:d}p{1:d}'.format(*x) + s for x in expo 
             for s in states]
    
    # Substitute value of parameters on matrix
    # Single promoter
    A_mat_s = A_mat_fun(*par_single)
    # Two promoters
    A_mat_d = A_mat_fun(*par_double)

    # Generate division matrix for all states
    # Initialize matrix
    Z_mat_div = np.zeros([len(names), len(names)])
    
    # Loop through exponents
    for i, e in enumerate(expo):
        # Loop through states
        for j, s in enumerate(states):
            Z_mat_div[(i * len(states)) + j,
                      j::len(states)] = Z_mat[i]
    
    # Initialize data frame
    df = pd.DataFrame(columns=['time', 'state', 'cycle'] + names)
    
    # Initilaize global time
    t_sim = 0
    
    ###  Loop through cycles  ###
    for cyc in range(n_cycles):
        # == Single promoter == #
        # Define time array
        t = np.linspace(0, t_single, n_steps)

        # Integrate moment equations
        mom = sp.integrate.odeint(rhs_dmomdt, mom_init, t, 
                             args=(A_mat_s,))

        # Generate data frame
        df_mom = pd.DataFrame(mom, columns=names)
        # Append time, state and cycle
        df_mom = df_mom.assign(time=t + t_sim)
        df_mom = df_mom.assign(state=['single'] * mom.shape[0])
        df_mom = df_mom.assign(cycle=[cyc] * mom.shape[0])
        
        # Append results to global data frame
        df = df.append(df_mom, ignore_index=True, sort=False)
        
        # Update global time
        # NOTE: Here we account for whether or not this is the first cycle
        # This is because of the extra time bit we have to add in order not
        # to have two overlapping time points
        if cyc == 0:
            t_sim = t_sim + t[-1]
        else:
            t_sim = t_sim + t[-1] + np.diff(t)[0]
        
        # == Two promoters == #
        
        # Define initial conditions as last 
        # point of single promoter state
        mom_init = mom[-1, :]
        
        # Define time array
        t = np.linspace(0, t_double, n_steps)

        # Integrate moment equations
        mom = sp.integrate.odeint(rhs_dmomdt, mom_init, t, 
                                  args=(A_mat_d,))

        # Generate data frame
        df_mom = pd.DataFrame(mom, columns=names)
        # Append time, state and cycle
        df_mom = df_mom.assign(time=t + t_sim)
        df_mom = df_mom.assign(state=['double'] * mom.shape[0])
        df_mom = df_mom.assign(cycle=[cyc] * mom.shape[0])
        
        # Append results to global data frame
        df = df.append(df_mom, ignore_index=True, sort=False)
        
        # Update global time
        t_sim = t_sim + t[-1] + np.diff(t)[0]
        
        # == Cell division == #
        
        # Extract moments during last time point
        mom_fix = mom[-1, :]
        
        # Compute moments after cell division
        mom_init = np.dot(Z_mat_div, mom_fix)
        
    return df


def load_constants():
    '''
    Returns a dictionary of various constants 
    '''
    # Find project parental directory
    repo = git.Repo('./', search_parent_directories=True)
    homedir = repo.working_dir
    # Define constants
    epR_O1=-15.3
    epR_O2=-13.9
    epR_O3=-9.7 
    HG104=22
    RBS1027=260
    RBS1L=1740
    Nns=4.6E6
    epAI=4.5
    Ka=139
    Ki=0.53
    gm=1 / (3 * 60)
    k0=2.7E-3
    Vcell=2.15
    rp=0.05768706295740175
    # Load MCMC parameters
    with open(homedir + '/data/mcmc/lacUV5_constitutive_mRNA_double_expo.pkl',
              'rb') as file:
        unpickler = pickle.Unpickler(file)
        gauss_flatchain = unpickler.load()
        gauss_flatlnprobability = unpickler.load()
    # Generate a Pandas Data Frame with the mcmc chain
    index = ['kp_on', 'kp_off', 'rm']
    # Generate a data frame out of the MCMC chains
    df_mcmc = pd.DataFrame(gauss_flatchain, columns=index)
    index = df_mcmc.columns
    # map value of the parameters
    max_idx = np.argmax(gauss_flatlnprobability, axis=0)
    kp_on, kp_off, rm = df_mcmc.iloc[max_idx, :] * gm

    # Compute repressor dissociation constants
    kr_off_O1 = kr_off_fun(epR_O1, k0, kp_on, kp_off, Nns, Vcell)
    kr_off_O2 = kr_off_fun(epR_O2, k0, kp_on, kp_off, Nns, Vcell)
    kr_off_O3 = kr_off_fun(epR_O3, k0, kp_on, kp_off, Nns, Vcell)

    return dict(epR_O1=epR_O1, epR_O2=epR_O2, epR_O3=epR_O3,
                HG104=HG104, RBS1027=RBS1027, RBS1L=RBS1L,
                Nns=Nns, epAI=epAI, Ka=Ka, Ki=Ki,
                gm=gm, rm=rm, kp_on=kp_on, kp_off=kp_off, k0=k0, Vcell=Vcell,
                rp=rp,
                kr_off_O1=kr_off_O1, kr_off_O2=kr_off_O2, kr_off_O3=kr_off_O3)
