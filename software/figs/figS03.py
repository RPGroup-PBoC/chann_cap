#%%
import os
import pickle
import cloudpickle
import itertools
import glob
import numpy as np
import scipy as sp
import pandas as pd
import git

# Import matplotlib stuff for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

# Seaborn, useful for graphics
import seaborn as sns

# Import the project utils
import ccutils

# Set PBoC plotting format
ccutils.viz.set_plotting_style()
# Increase dpi

#%%

# Find home directory for repo
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir

# Define directories for data and figure 
figdir = f'{homedir}/fig/si/'
datadir = f'{homedir}/data/mRNA_FISH/'
mcmcdir = f'{homedir}/data/mcmc/'
# %%

# Read the data
df = pd.read_csv(f'{datadir}Jones_Brewster_2014.csv', index_col=0)

# Extract the lacUV5 data
dfUV5 = df[df.experiment == 'UV5']

# Load the flat-chain
with open(f'{mcmcdir}lacUV5_constitutive_mRNA_prior.pkl', 'rb') as file:
    unpickler = pickle.Unpickler(file)
    gauss_flatchain = unpickler.load()
    gauss_flatlnprobability = unpickler.load()
    
# Generate a Pandas Data Frame with the mcmc chain
index = ['kp_on', 'kp_off', 'rm']

# Generate a data frame out of the MCMC chains
df_mcmc = pd.DataFrame(gauss_flatchain, columns=index)

# rerbsine the index with the new entries
index = df_mcmc.columns

# map value of the parameters
max_idx = np.argmax(gauss_flatlnprobability, axis=0)
kp_on, kp_off, rm = df_mcmc.iloc[max_idx, :]

# Define bins
bins = np.arange(0, dfUV5.mRNA_cell.max())

logp_mRNA = ccutils.model.log_p_m_unreg(bins, kp_on, kp_off, 1, rm)

# Plot the histogram of the data with bins of width 1
_ = plt.hist(dfUV5.mRNA_cell, bins=bins, density=1, histtype='stepfilled',
             alpha=1, label='sm-FISH data', align='left', lw=0)

plt.step(bins, np.exp(logp_mRNA), color='r', ls='-', lw=1.5,
         label='two-state promoter fit')

# Label the plot
plt.xlabel('mRNA / cell')
plt.ylabel('probability')
plt.legend()
plt.tight_layout()
plt.savefig(f'{figdir}/figS03.pdf', bbox_inches='tight')
