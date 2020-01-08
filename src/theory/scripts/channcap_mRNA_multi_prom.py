#%%
# Our numerical workhorses
import numpy as np
import pandas as pd

import itertools
# Import libraries to parallelize processes
from joblib import Parallel, delayed

# Import matplotlib stuff for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

# Seaborn, useful for graphics
import seaborn as sns

# Pickle is useful for saving outputs that are computationally expensive
# to obtain every time
import pickle

import os
import glob
import git

# Import the project utils
import ccutils

#%%
# Find home directory for repo
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir

# Read MaxEnt distributions
print('Reading MaxEnt distributions')
df_maxEnt_mRNA = pd.read_csv(
    f"{homedir}/data/csv_maxEnt_dist/MaxEnt_Lagrange_mult_mRNA.csv"
)

# Define dictionaries to map operator to binding energy and rbs to rep copy
op_dict = dict(zip(["O1", "O2", "O3"], [-15.3, -13.9, -9.7]))
rbs_dict = dict(
    zip(
        ["HG104", "RBS1147", "RBS446", "RBS1027", "RBS1", "RBS1L"],
        [22, 60, 124, 260, 1220, 1740],
    )
)

# Define sample space
mRNA_space = np.arange(0, 100)
protein_space = np.array([0])

# Group df_maxEnt by operator and repressor copy number
df_group = df_maxEnt_mRNA.groupby(["operator", "repressor"])

# Define column names for data frame
names = ["operator", "binding_enery", "repressor", "channcap"]

# Initialize data frame to save channel capacity computations
df_channcap = pd.DataFrame(columns=names)

# Define function to compute in parallel the channel capacity
def cc_parallel_mRNA(df_lagrange):
    # Build mRNA transition matrix
    Qmc = ccutils.channcap.trans_matrix_maxent(
        df_lagrange, 
        mRNA_space, 
        protein_space, 
        True
    )

    # Compute the channel capacity with the Blahut-Arimoto algorithm
    cc_m, _, _ = ccutils.channcap.channel_capacity(Qmc.T, epsilon=1e-4)

    # Extract operator and repressor copy number
    op = df_lagrange.operator.unique()[0]
    eRA = df_lagrange.binding_energy.unique()[0]
    rep = df_lagrange.repressor.unique()[0]

    return [op, eRA, rep, cc_m]

# Run the function in parallel
print('Running Blahut-Arimoto algorithm in multiple cores')
ccaps = Parallel(n_jobs=6)(
    delayed(cc_parallel_mRNA)(df_lagrange)
    for group, df_lagrange in df_group
)

# Convert to tidy data frame
ccaps = pd.DataFrame(ccaps, columns=names)

# Concatenate to data frame
df_channcap = pd.concat([df_channcap, ccaps], axis=0)

# Save results
print('Saving results into memory')
df_channcap.to_csv(
    f"{homedir}/data/csv_maxEnt_dist/chann_cap_multi_prom_mRNA.csv",
    index=False,
)
print('Done!')