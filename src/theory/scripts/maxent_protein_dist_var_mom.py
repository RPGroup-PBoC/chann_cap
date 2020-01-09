#%%
import os
import itertools
import cloudpickle
import re
import glob
import git

# Our numerical workhorses
import numpy as np
import pandas as pd
import scipy as sp

# Import library to perform maximum entropy fits
from maxentropy.skmaxent import FeatureTransformer, MinDivergenceModel

# Import libraries to parallelize processes
from joblib import Parallel, delayed

# Import the project utils
import ccutils

# Find home directory for repo
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir

tmpdir = f'{homedir}/tmp/'
datadir = f'{homedir}/data/csv_maxEnt_dist/'

#%%
# Load moments for multi-promoter level
df_constraints = pd.read_csv(
    f'{datadir}MaxEnt_multi_prom_constraints.csv'
)

# Remove the zeroth moment column
df_constraints = df_constraints.drop(labels='m0p0', axis=1)

# Remove all non-protein moments with a clever use of
# regular expressions
df_constraints = df_constraints[df_constraints.columns.drop(
                 list(df_constraints.filter(regex='m[1-9]')))]

# Define repressors to keep
repressors = [0, 22, 260, 1740]

# Keep only desired repressors
df_constraints = df_constraints[df_constraints['repressor'].isin(repressors)]

#%%

# Extract protein moments in constraints
prot_mom =  [x for x in df_constraints.columns if 'm0' in x]
# Define index of moments to be used in the computation
moments = [tuple(map(int, re.findall(r'\d+', s))) for s in prot_mom]

#%%

# Define sample space
mRNA_space = np.array([0])  # Dummy space
protein_space = np.arange(0, 10E4)

# Generate sample space as a list of pairs using itertools.
samplespace = list(itertools.product(mRNA_space, protein_space))

# Specify column names for data frame to save results
names = ['operator', 'binding_energy', 'repressor', 'inducer_uM', 'num_mom']
# Add names of the constraints
names = names + ['lambda_m' + str(m[0]) + 'p' + str(m[1]) for m in moments]

# Initialize empty dataframe
df_maxEnt = pd.DataFrame([], columns=names)

#%%

# Define function for parallel computation
def maxEnt_parallel(n, constraints_names, features, idx, df):
    # Report on progress
    print('iteration: ',idx)

    # Extract constraints
    constraints = df.loc[constraints_names]

    # Perform MaxEnt computation
    # We use the Powell method because despite being slower it is more
    # robust than the other implementations.
    Lagrange = ccutils.maxent.MaxEnt_bretthorst(constraints, features, 
                                 algorithm='Powell', 
                                 tol=1E-5, paramtol=1E-5,
                                 maxiter=10000)
    
    # Fill moments not used with nothing
    Lagrange_fill = np.zeros(len(prot_mom))
    # Substitute values of moments
    Lagrange_fill[0:len(Lagrange)] = Lagrange
    # Save Lagrange multipliers into dataframe
    series = pd.Series(Lagrange_fill, index=names[5::])

    # Create series to save parameters
    par_series = df.drop(prot_mom)
    # Append number of moments
    par_series = par_series.append(pd.Series(n, index=['num_mom']))
    # Add other features to series before appending to dataframe
    series = pd.concat([par_series, series])

    return series

# Loop through increasing numebr of constraints
for n in range(2, len(prot_mom) + 1):
    # Define moments to be use in this cycle
    constraints_names = prot_mom[0:n]
    print(constraints_names)
    
    # Extract exponents of moments
    moms = [tuple(map(int, re.findall(r'\d+', s))) for s in 
            constraints_names]
    
    # Initialize matrix to save all the features that are fed to the
    # maxentropy function
    features = np.zeros([len(moms), len(samplespace)])
    
    # Loop through constraints and compute features
    for i, mom in enumerate(moms):
        features[i, :] = [ccutils.maxent.feature_fn(x, mom) for x in
                          samplespace]
        
    # Run the function in parallel
    maxEnt_series = Parallel(n_jobs=6)(
        delayed(maxEnt_parallel)(n, constraints_names, features, idx, df)
                           for idx, df in df_constraints.iterrows())

    for s in maxEnt_series:
        df_maxEnt = df_maxEnt.append(s, ignore_index=True)

    df_maxEnt.to_csv(f"{datadir}MaxEnt_Lagrange_mult_protein_var_mom.csv",
                     index=False)
