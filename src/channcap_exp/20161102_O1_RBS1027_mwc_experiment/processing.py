import sys
import pickle
import os
import glob
import re
import datetime
import itertools

# Our numerical workhorses
import numpy as np
from sympy import mpmath
import scipy.optimize
import scipy.special
import scipy.integrate
import pandas as pd

# Import libraries to parallelize processes
from joblib import Parallel, delayed

# Import matplotlib stuff for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

# Seaborn, useful for graphics
import seaborn as sns

# Import the utils for this project
sys.path.insert(0, '../../theory/')
import chann_cap_utils as chann_cap

chann_cap.set_plotting_style()

#==============================================================================
# METADATA
#==============================================================================

DATE = 201611102
USERNAME = 'gchure'
OPERATOR = 'O1'
STRAIN = 'RBS1027'
REPRESSOR = 130
BINDING_ENERGY = -15.3

# Determine the parameters for the bootstraping
bins = np.floor(np.logspace(0, 4, 100))
fracs = 1 / np.linspace(1 / 0.6, 1, 10)
nreps = 25 # number of bootstrap samples per fraction

#============================================================================== 

# Read data
df_micro = pd.read_csv('../../../data/csv_microscopy/mwc_data/' + \
        str(DATE) + '_' + OPERATOR + '_' + STRAIN + \
        '_IPTG_titration_microscopy.csv', header=0, comment='#') 


#============================================================================== 

# Define output directory
outputdir = '../../../data/csv_channcap_bootstrap/'
# removing the auto and delta
df = df_micro[(df_micro.rbs != 'auto') & (df_micro.rbs != 'delta')]

#============================================================================== 
# Compute channel capacity for experimental data
#============================================================================== 
compute_exp = True
if compute_exp:
    def channcap_bs_parallel(b):
        # Initialize matrix to save bootstrap repeats
        MI_bs = np.zeros([len(fracs), nreps])
        samp_sizes = np.zeros(len(fracs))
        for i, frac in enumerate(fracs):
            MI_bs[i, :], samp_sizes[i] = chann_cap.channcap_bootstrap(df, bins=b,
                                                        nrep=nreps, frac=frac)
        return (MI_bs, samp_sizes)

    # Perform the parallel computation
    print('Performing bootsrap estimates of channel capacity...')
    channcap_list = Parallel(n_jobs=7)(delayed(channcap_bs_parallel)(b) \
                                        for b in bins)
    print('Done performing calculations.')

    # Define the parameters to include in the data frame
    kwarg_list = ['date', 'username', 'operator', 'binding_energy',  'rbs', 
                    'repressors']
    # Extract the parameters from the data frame
    kwargs = dict((x, df[x].unique()[0]) for x in kwarg_list)

    # Convert the list into a tidy data frame
    df_cc_bs = chann_cap.tidy_df_channcap_bs(channcap_list, fracs, bins, **kwargs)

    # Save outcome
    filename = str(kwargs['date']) + '_' + kwargs['operator'] + '_' +\
                kwargs['rbs'] + '_' + 'channcap_bootstrap.csv'
    df_cc_bs.to_csv(outputdir + filename, index=False)
    print('Saved as dataframe.')

#============================================================================== 
# Extrapolate to N -> oo
#============================================================================== 
filename = str(DATE) + '_' + OPERATOR + '_' +\
               STRAIN + '_' + 'channcap_bootstrap.csv'

df_cc_bs = pd.read_csv(outputdir + filename, header=0)

# Group by the number of bins
df_group = df_cc_bs.groupby('bins')

# Initialize data frame to save the I_oo estimates
df_cc = pd.DataFrame(columns=['date', 'bins', 'channcap'])

for group, data in df_group:
    x = 1 / data.samp_size
    y = data.channcap_bs
    # Perform linear regression
    lin_reg = np.polyfit(x, y, deg=1)
    df_tmp = pd.Series([DATE, group, lin_reg[1]],
                          index=['date', 'bins', 'channcap'])
    df_cc = df_cc.append(df_tmp, ignore_index=True)

# Convert date and bins into integer
df_cc[['date', 'bins']] = df_cc[['date', 'bins']].astype(int)

#============================================================================== 
# Computing the channel capacity for randomized data
#============================================================================== 

compute_shuff = True

if compute_shuff:
    print('shuffling mean_intensity data')
    df = df.assign(shuffled=df.mean_intensity.sample(frac=1).values)

    # Define the parallel function to run
    def channcap_bs_parallel_shuff(b):
        # Initialize matrix to save bootstrap repeats
        MI_bs = np.zeros([len(fracs), nreps])
        samp_sizes = np.zeros(len(fracs))
        for i, frac in enumerate(fracs):
            MI_bs[i, :], samp_sizes[i] = chann_cap.channcap_bootstrap(df, bins=b,
                                            nrep=nreps, frac=frac,
                                            **{'output_col' : 'shuffled'})
        return (MI_bs, samp_sizes)

    # Perform the parallel computation
    print('Performing bootsrap estimates on random data')
    channcap_list_shuff = Parallel(n_jobs=7)\
                          (delayed(channcap_bs_parallel_shuff)(b) \
                                        for b in bins)
    print('Done performing calculations.')

    # Define the parameters to include in the data frame
    kwarg_list = ['date', 'username', 'operator', 'binding_energy',  'rbs', 
                    'repressors']
    # Extract the parameters from the data frame
    kwargs = dict((x, df[x].unique()[0]) for x in kwarg_list)
    # Convert the list into a tidy data frame
    df_cc_bs_shuff = chann_cap.tidy_df_channcap_bs(channcap_list_shuff, fracs,
                                                   bins, **kwargs)
    # Save outcome
    filename = str(kwargs['date']) + '_' + kwargs['operator'] + '_' +\
                kwargs['rbs'] + '_' + 'channcap_bootstrap_shuffled.csv'
    df_cc_bs_shuff.to_csv(outputdir + filename, index=False)
    print('Saved as dataframe.')

#============================================================================== 
# Extraploate randomized data to N -> oo
#============================================================================== 

filename = str(DATE) + '_' + OPERATOR + '_' +\
               STRAIN + '_' + 'channcap_bootstrap_shuffled.csv'

df_cc_shuff = pd.read_csv(outputdir + filename, header=0)

# Group by the number of bins
df_group = df_cc_shuff.groupby('bins')

# Initialize data frame to save the I_oo estimates
df_cc_shuff = pd.DataFrame(columns=['date', 'bins', 'channcap'])

for group, data in df_group:
    x = 1 / data.samp_size
    y = data.channcap_bs
    # Perform linear regression
    lin_reg = np.polyfit(x, y, deg=1)
    df_tmp = pd.Series([DATE, group, lin_reg[1]],
                          index=['date', 'bins', 'channcap'])
    df_cc_shuff = df_cc_shuff.append(df_tmp, ignore_index=True)

# Convert date and bins into integer
df_cc_shuff[['date', 'bins']] = df_cc_shuff[['date', 'bins']].astype(int)

#============================================================================== 
# Plot channel capacity as a function of number of bins
#============================================================================== 
# Initialize figure
fig, ax = plt.subplots(1, 1)

# Initialize figure
fig, ax = plt.subplots(1, 1)
ax.plot(df_cc.bins, df_cc.channcap, label='experimental data')
ax.plot(df_cc_shuff.bins, df_cc_shuff.channcap, label='shuffled data')

ax.set_xlabel('# bins')
ax.set_ylabel(r'channel capacity $I_\infty$ (bits)')
ax.set_xscale('log')
ax.legend(loc=0, title='date ' + str(DATE))
plt.savefig('./outdir/bins_vs_channcap.png')

