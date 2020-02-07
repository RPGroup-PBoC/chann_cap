#%%
import os
import itertools
import cloudpickle
import re
import glob
import statsmodels.api as sm
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

# Define directories for data and figure 
datadir = f'{homedir}/data/csv_maxEnt_dist/'

#%%
# Load moments for multi-promoter level
df_constraints = pd.read_csv(
    f'{datadir}MaxEnt_multi_prom_constraints.csv'
)

print('reading distribution moments')
# Remove the zeroth moment column
df_constraints = df_constraints.drop(labels="m0p0", axis=1)

# %%
print('Finding multiplicative factor for noise')

# Read moments for multi-promoter model
df_mom_rep = pd.read_csv(datadir + 'MaxEnt_multi_prom_constraints.csv')

# Read experimental determination of noise
df_noise = pd.read_csv(f'{homedir}/data/csv_microscopy/' + 
                       'microscopy_noise_bootstrap.csv')

# Find the mean unregulated levels to compute the fold-change
mean_m_delta = np.mean(df_mom_rep[df_mom_rep.repressor == 0].m1p0)
mean_p_delta = np.mean(df_mom_rep[df_mom_rep.repressor == 0].m0p1)

# Compute the noise for the multi-promoter data
df_mom_rep = df_mom_rep.assign(
    m_noise=(
        np.sqrt(df_mom_rep.m2p0 - df_mom_rep.m1p0 ** 2) / df_mom_rep.m1p0
    ),
    p_noise=(
        np.sqrt(df_mom_rep.m0p2 - df_mom_rep.m0p1 ** 2) / df_mom_rep.m0p1
    ),
    m_fold_change=df_mom_rep.m1p0 / mean_m_delta,
    p_fold_change=df_mom_rep.m0p1 / mean_p_delta,
)

# Initialize list to save theoretical noise
thry_noise = list()
# Iterate through rows
for idx, row in df_noise.iterrows():
    # Extract information
    rep = float(row.repressor)
    op = row.operator
    if np.isnan(row.IPTG_uM):
        iptg = 0
    else:
        iptg = row.IPTG_uM
    
    # Extract equivalent theoretical prediction
    thry = df_mom_rep[(df_mom_rep.repressor == rep) &
                       (df_mom_rep.operator == op) &
                       (df_mom_rep.inducer_uM == iptg)].p_noise
    # Append to list
    thry_noise.append(thry.iloc[0])
df_noise = df_noise.assign(noise_theory = thry_noise)

# Linear regression to find multiplicative factor

# Extract fold-change
fc = df_noise.fold_change.values
# Set values for ∆lacI to be fold-change 1
fc[np.isnan(fc)] = 1
# Normalize weights
weights = fc / fc.sum()

# Declare linear regression model
wls_model = sm.WLS(df_noise.noise.values,
                   df_noise.noise_theory.values,
                   weights=weights)
# Fit parameter
results = wls_model.fit()
noise_factor = results.params[0]
# %%
print('Increasing noise')
# Compute variance
p_var = df_constraints['m0p2'] - df_constraints['m0p1']**2
# Update second moment
df_constraints['m0p2'] = (noise_factor**2) * df_constraints['m0p2'] - \
                         (noise_factor**2 - 1) * df_constraints['m0p1']**2

# %%
print('Finding multiplicative factor for skewness')

# Read moments for multi-promoter model
df_mom_rep = pd.read_csv(datadir + 'MaxEnt_multi_prom_constraints.csv')

# Read experimental determination of noise
df_noise = pd.read_csv(f'{homedir}/data/csv_microscopy/' + 
                       'microscopy_noise_bootstrap.csv')

# Find the mean unregulated levels to compute the fold-change
mean_m_delta = np.mean(df_mom_rep[df_mom_rep.repressor == 0].m1p0)
mean_p_delta = np.mean(df_mom_rep[df_mom_rep.repressor == 0].m0p1)

# Compute the skewness for the multi-promoter data
m_mean = df_mom_rep.m1p0
p_mean = df_mom_rep.m0p1
m_var = df_mom_rep.m2p0 - df_mom_rep.m1p0 ** 2
p_var = df_mom_rep.m0p2 - df_mom_rep.m0p1 ** 2

df_mom_rep = df_mom_rep.assign(
    m_skew=(df_mom_rep.m3p0 - 3 * m_mean * m_var - m_mean**3)
    / m_var**(3 / 2),
    p_skew=(df_mom_rep.m0p3 - 3 * p_mean * p_var - p_mean**3)
    / p_var**(3 / 2),
)

# Initialize list to save theoretical noise
thry_skew = list()
# Iterate through rows
for idx, row in df_noise.iterrows():
    # Extract information
    rep = float(row.repressor)
    op = row.operator
    if np.isnan(row.IPTG_uM):
        iptg = 0
    else:
        iptg = row.IPTG_uM
    
    # Extract equivalent theoretical prediction
    thry = df_mom_rep[(df_mom_rep.repressor == rep) &
                       (df_mom_rep.operator == op) &
                       (df_mom_rep.inducer_uM == iptg)].p_skew
    # Append to list
    thry_skew.append(thry.iloc[0])
    
df_noise = df_noise.assign(skew_theory = thry_skew)

# Extract fold-change
fc = df_noise.fold_change.values
# Set values for ∆lacI to be fold-change 1
fc[np.isnan(fc)] = 1
# Normalize weights
weights = fc / fc.sum()

# Declare linear regression model
wls_model = sm.WLS(df_noise.skewness.values,
                   df_noise.skew_theory.values,
                   weights=weights)
# Fit parameter
results = wls_model.fit()
skew_factor = results.params[0]

# Update third moment
print('Increasing skewness')
df_constraints['m0p3'] = (8 * skew_factor * df_constraints['m0p3'] - \
                (24 * skew_factor - 12) * df_constraints['m0p1'] * p_var - \
                (8 * skew_factor - 1) * df_constraints['m0p1']**3)

#%%

# Extract protein moments in constraints
prot_mom = [x for x in df_constraints.columns if "m0" in x]
# Define index of moments to be used in the computation
moments = [tuple(map(int, re.findall(r"\d+", s))) for s in prot_mom][0:3]
print(f'moments to be used for inference: {moments}')

# Define sample space
mRNA_space = np.array([0])  # Dummy space
protein_space = np.arange(0, 10e4)

# Generate sample space as a list of pairs using itertools.
samplespace = list(itertools.product(mRNA_space, protein_space))

# Initialize matrix to save all the features that are fed to the
# maxentropy function
features = np.zeros([len(moments), len(samplespace)])

# Loop through constraints and compute features
for i, mom in enumerate(moments):
    features[i, :] = [ccutils.maxent.feature_fn(x, mom) for x in samplespace]

#%%

# Initialize data frame to save the lagrange multipliers.
names = ["operator", "binding_energy", "repressor", "inducer_uM"]
# Add names of the constraints
names = names + ["lambda_m" + str(m[0]) + "p" + str(m[1]) for m in moments]

# Initialize empty dataframe
df_maxEnt = pd.DataFrame([], columns=names)

# Define column names containing the constraints used to fit the distribution
constraints_names = ["m" + str(m[0]) + "p" + str(m[1]) for m in moments]

# Define function for parallel computation
def maxEnt_parallel(idx, df):
    # Report on progress
    print("iteration: ", idx)

    # Extract constraints
    constraints = df.loc[constraints_names]

    # Perform MaxEnt computation
    # We use the Powell method because despite being slower it is more
    # robust than the other implementations.
    Lagrange = ccutils.maxent.MaxEnt_bretthorst(
        constraints,
        features,
        algorithm="Powell",
        tol=1e-5,
        paramtol=1e-5,
        maxiter=10000,
    )
    # Save Lagrange multipliers into dataframe
    series = pd.Series(Lagrange, index=names[4::])

    # Add other features to series before appending to dataframe
    series = pd.concat([df.drop(constraints_names), series])

    return series

# Run the function in parallel
maxEnt_series = Parallel(n_jobs=6)(
    delayed(maxEnt_parallel)(idx, df)
    for idx, df in df_constraints.iterrows()
)

# Initialize data frame to save list of parameters
df_maxEnt = pd.DataFrame([], columns=names)

for s in maxEnt_series:
    df_maxEnt = df_maxEnt.append(s, ignore_index=True)

df_maxEnt.to_csv(f'{datadir}MaxEnt_Lagrange_mult_protein_correction.csv',
                 index=False)