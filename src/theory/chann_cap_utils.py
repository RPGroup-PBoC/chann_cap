# -*- coding: utf-8 -*-
"""
Title:
    chann_cap_utils
Last update:
    2018-01-17
Author(s):
    Manuel Razo-Mejia
Purpose:
    This file is a compilation of the funtions developed for the channel 
    capacity project. Most of the functions found here can also be found
    in different iPython notebooks, but in order to break down those
    notebooks into shorter and more focused notebooks it is necessary to 
    call some functions previously defined.
"""

#==============================================================================
# Our numerical workhorses
import numpy as np
import scipy.optimize
import scipy.special
import scipy.integrate
import mpmath
import pandas as pd

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
    return 1.66 / 1 * k0 * 4.6E6 * np.exp(epsilon)

#=============================================================================== 

# define a np.frompyfunc that allows us to evaluate the sympy.mp.math.hyp1f1
np_log_hyp= np.frompyfunc(lambda x, y, z: \
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

#==============================================================================
# chemical_masater_eq_analytic_protein
#============================================================================== 

# define a np.frompyfunc that allows us to evaluate the sympy.mp.math.hyp1f1
np_log_gauss_hyp = np.frompyfunc(lambda a, b, c, z: \
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
# chemical_master_mRNA_FISH_mcmc 
#============================================================================== 
# define a np.frompyfunc that allows us to evaluate the sympy.mp.math.hyp1f1
np_log_hyp= np.frompyfunc(lambda x, y, z: \
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

#==============================================================================
# blahut_arimoto_channel_capacity
#==============================================================================

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

#============================================================================== 

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

#============================================================================== 
# Computing experimental channel capacity
#============================================================================== 

def trans_matrix(df, bins, frac=None,
                 output_col='mean_intensity', group_col='IPTG_uM'):
    '''
    Builds the transition matrix P(m|C) from experimental data contained in a
    tidy dataframe. The matrix is build by grouping the data according to the
    entries from group_col.
    Parameters
    ----------
    df : pandas Dataframe
        Single cell output reads measured at different inducer concentrations. 
        The data frame must contain a column output_col that will be binned to
        build the matrix, and a matrix group_col that will be used to group
        the different inputs.
    bins : int.
        Number of bins to use when building the empirical PMF of the data set.
        If `bins` is a string from the list below, `histogram` will use
        the method chosen to calculate the optimal bin width and
        consequently the number of bins from the data that falls within 
        the requested range.
    frac : None or float [0, 1]
        Fraction of the data to sample for building the matrix. Default = None
        meaning that the entire data set will be used. The fraction of data is 
        taken per input value.
    output_col : str.
        Name of the column that contains the quantity (usually fluorescence 
        measurements) to be binned in order to build the matrix
    group_col : str.
        Name of the column that contains the inputs C of the matrix (usually
        inducer concentrations). This column will be used to separate the
        different rows ot the transition matrix.
    Returns
    -------
    QmC : array-like.
        Experimentally determined input-output function.
    len(df) : int
        Number of data points considered for building the matrix
    '''
    
    # Extract the data to bin
    bin_data = df[output_col]
    
    # indicate the range in which bin the data
    bin_range = [np.min(bin_data), np.max(bin_data)]
    
    # If inidicated select a fraction frac of the data at random
    if frac != None:
        # Group by group_col and take samples
        group = df.groupby(group_col)
        # Initialize data frame to save samples
        df_sample = pd.DataFrame()
        for g, d in group:
            df_sample = pd.concat([df_sample, d.sample(frac=frac)])
        # Use the subsample data frame
        df = df_sample
    
    # Extract the number of unique inputs in the data frame
    n_inputs = df.IPTG_uM.unique().size
    
    # Initialize transition matrix
    QmC = np.zeros([bins, n_inputs])
    
    # Loop through different groups
    # Unfortunately we need to initalize a counter because the groupby
    # function is not compatible with enumerate
    k = 0
    for c, f in df.groupby(group_col):
        # Obtain the empirical PMF from the experimental data
        p, bin_edges = np.histogram(f[output_col], bins=int(bins), 
                                    range=bin_range)
        # Normalized the empirical PMF. We don't use the option from numpy
        # because it DOES NOT build a PMF but assumes a PDF.
        p = p / np.sum(p)
        # Add column to matrix
        QmC[:, k] = p
        # Increase counter
        k+=1
   
    return QmC, len(df)

#============================================================================== 

def channcap_bootstrap(df, nrep, bins, frac, **kwargs):
    '''
    Given a fraction of the data frac computes the channel capacity nrep times
    taking different random samples on each time.
    Parameters
    ----------
    df : pandas Dataframe
        Single cell output reads measured at different inducer concentrations. 
        The data frame must contain a column output_col that will be binned to
        build the matrix, and a matrix group_col that will be used to group
        the different inputs.
    bins : int.
        Number of bins to use when building the empirical PMF of the data set.
        If `bins` is a string from the list below, `histogram` will use
        the method chosen to calculate the optimal bin width and
        consequently the number of bins from the data that falls within 
        the requested range.
    frac : float [0, 1]
        Fraction of the data to sample for building the matrix. 
        The fraction of data is taken per input value.
    kwargs : dictionary
        Optional arguments that can be passed to the trans_matrix function.
        Optional arguments that can be passed to the channel_capacity function.
    '''
    #---------------------------------------------
    # Extract arguments for trans_matrix function
    tm_arg_names =  trans_matrix.__code__.co_varnames\
                        [0:trans_matrix.__code__.co_argcount]
    tm_kwargs = dict((k, kwargs[k]) for k in tm_arg_names if k in kwargs)
    
    # Extract the arguments for the channel capacity function
    cc_arg_names =  channel_capacity.__code__.co_varnames\
                        [0:channel_capacity.__code__.co_argcount]
    cc_kwargs = dict((k, kwargs[k]) for k in cc_arg_names if k in kwargs)
    #---------------------------------------------
    
    # Initialize array to save channel capacities
    MI = np.zeros(nrep)
    for i in np.arange(nrep):
        QgC, samp_size = trans_matrix(df, bins=bins, frac=frac,  **tm_kwargs)
        MI[i] = channel_capacity(QgC.T, **cc_kwargs)[0]
    
    return MI, samp_size

#============================================================================== 

def tidy_df_channcap_bs(channcap_list, fracs, bins, **kwargs):
    '''
    Breaks up the output of channcap_bs_parallel into a tidy data frame.
    Parameters
    ----------
    channcap_list : list of length len(bins)
        List containing the channel capacity bootstrap repeats for each bin.
        Each entry in the list contains 2 elements:
        1) MI_bs : matrix of size len(fracs) x nreps
        This matrix contains on each row the nreps bootrstrap estimates for a
        fraction of the data frac.
        2) samp_sizes : array of length len(fracs)
        This array keeps the amount of data used for each of the fractions
        indicated.
    fracs : array-like
        Array containing the fractions at which the bootstrap estimates were 
        computed.
    bins : array-like.
        Number of bins used when generating the matrix Qg|c
    kwargs : dictionary
        Dictionary containing extra fields to be included in the tidy dataframe.
        Every entry in this dictionary will be added to all rows of the dataframe.
        Examples of relevant things to add:
        - date of the sample
        - username that generated the data
        - operator
        - binding_energy
        - rbs
        - repressors
    Returns
    -------
    Tidy dataframe of the channel capacity bootstrap samples
    '''
    # Initialize data frame where all the information will be saved
    df = pd.DataFrame()
    
    # Loop through the elements of the list containing the bs samples
    # for each number of bins
    for i, b in enumerate(bins):
        # Extract the sample element
        bin_samples = channcap_list[i] 
        # Loop through each of the rows of the MI_bs matrix containing the
        # nrep samples for each fraction
        for j, s in enumerate(bin_samples[0]):
            # Initialize df to save the outcomes from this specific fraction
            df_frac = pd.DataFrame(s, columns=['channcap_bs'])
            # Save sample size
            df_frac['samp_size'] = [bin_samples[1][j]] * len(s)
            # Save fraction of data used
            df_frac['frac'] = [fracs[j]] * len(s)
            # Save the number of bins used for this bs samples
            df_frac['bins'] = [b] * len(s)    
            # append to the general data frame
            df = pd.concat([df, df_frac], axis=0)
        
    
    # Add elements contained in the kwards dictioary
    for key, value in kwargs.items():
        df[key] = [value] * len(df)
    
    return df

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

#============================================================================== 

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

#============================================================================== 
# Plotting functions
#============================================================================== 
def pmf_cdf_plot(x, px, legend_var, color_palette='Blues',
                 mean_mark=True, marker_height=0.3,
                 color_bar=True, cbar_label='',
                 figsize=(6,5), title='', xlabel='', xlim=None, ylim=None):
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
    ax[0].yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(\
                                    useMathText=True, 
                                    useOffset=False))

    # Loop through inducer concentrations
    for i, c in enumerate(legend_var):
        # PMF plot
        ax[0].plot(x, px[i,:],
                 label=r'${0:d}$'.format(c), drawstyle='steps',
                  color='k')
        # Fill between each histogram
        ax[0].fill_between(x, px[i,:],
                           color=colors[i], alpha=0.8, step='pre')
        # CDF plot
        ax[1].plot(x, np.cumsum(px[i,:]), drawstyle='steps',
                  color=colors[i], linewidth=2)

    # Label axis
    ax[0].set_title(title)
    ax[0].set_ylabel('probability')
    ax[0].margins(0.02)
    # Set scientific notation
    ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
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
    mean_plot = ax[0].scatter(mean_dist, [marker_height] * len(mean_dist), 
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
                     transform = cbar_ax.transAxes, fontsize=12)
    cbar.ax.get_yaxis().labelpad = 35
    cbar.set_label(r'{:s}'.format(cbar_label))

    plt.figtext(-0.02, .9, '(A)', fontsize=18)
    plt.figtext(-0.02, .46, '(B)', fontsize=18)

    plt.subplots_adjust(hspace=0.06)
