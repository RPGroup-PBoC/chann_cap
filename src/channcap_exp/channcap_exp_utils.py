# -*- coding: utf-8 -*-
"""
Title:
    channcap_exp_utils
Last update:
    2018-03-20
Author(s):
    Manuel Razo-Mejia
Purpose:
    This file is a compilation of the funtions developed to process
    microscopy data taken for the channel capacity project.
    Most of the functions found here can also be found
    in different iPython notebooks, but in order to break down those
    notebooks into shorter and more focused notebooks it is necessary to
    call some functions previously defined.
"""

# Our numerical workhorses
import numpy as np
import scipy as sp
import scipy.optimize
import scipy.special
import scipy.integrate
import pandas as pd

# Import libraries to parallelize processes
from joblib import Parallel, delayed

# Import plotting utilities
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

# Seaborn, useful for graphics
import seaborn as sns

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
# Computing experimental channel capacity
# =============================================================================

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

# =============================================================================

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

# =============================================================================

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


