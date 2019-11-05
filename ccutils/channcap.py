# -*- coding: utf-8 -*-
"""
Title:
    channcap.py
Last update:
    2018-11-22
Author(s):
    Manuel Razo-Mejia
Purpose:
    This file compiles all of the functions necessary to compute
    the channel capacity either from theoretical distributions
    or from experimental data.
"""

import numpy as np
import pandas as pd


# BLAHUT-ARIMOTO ALGORITHM
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


# EXPERIMENTAL CHANNEL CAPACITY

def trans_matrix(df, bins, frac=None, output_col='intensity', 
                 group_col='IPTG_uM', extract_auto=None):
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
    extract_auto : float.
        Mean autofluorescence per unit area that must be extracted to the
        group_col column values
    Returns
    -------
    QmC : array-like.
        Experimentally determined input-output function.
    len(df) : int
        Number of data points considered for building the matrix
    '''
    # Extract the data to bin
    bin_data = df[output_col]
    # Subtract background if asked for
    if extract_auto != None:
        bin_data = bin_data - extract_auto * df['area']

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
    QmC = np.zeros([int(bins), int(n_inputs)])

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
    # Extract arguments for trans_matrix function
    tm_arg_names =  trans_matrix.__code__.co_varnames\
                        [0:trans_matrix.__code__.co_argcount]
    tm_kwargs = dict((k, kwargs[k]) for k in tm_arg_names if k in kwargs)

    # Extract the arguments for the channel capacity function
    cc_arg_names =  channel_capacity.__code__.co_varnames\
                        [0:channel_capacity.__code__.co_argcount]
    cc_kwargs = dict((k, kwargs[k]) for k in cc_arg_names if k in kwargs)

    # Initialize array to save channel capacities
    MI = np.zeros(nrep)
    for i in np.arange(nrep):
        QgC, samp_size = trans_matrix(df, bins=bins, frac=frac,  **tm_kwargs)
        MI[i] = channel_capacity(QgC.T, **cc_kwargs)[0]

    return MI, samp_size


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
