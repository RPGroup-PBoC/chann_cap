#%%
import os
import itertools
import cloudpickle
import re
import glob

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

tmpdir = '../../../tmp/'
datadir = '../../../data/csv_maxEnt_dist/'

#%%
# Load moments for multi-promoter level
df_constraints = pd.read_csv(
    f'{datadir}MaxEnt_multi_prom_constraints.csv'
)

print('reading distribution moments')
# Remove the zeroth moment column
df_constraints = df_constraints.drop(labels="m0p0", axis=1)

# print('removing other repressor copy numbers')
# rep = [0, 22, 260, 1740]
# idx = [x in rep for x in df_constraints.repressor.values]
# df_constraints = df_constraints[idx]
# print(df_constraints.shape)
print('Increasing noise')
# Compute variance
p_var = df_constraints['m0p2'] - df_constraints['m0p1']**2
# Update second moment
df_constraints['m0p2'] = (4 * df_constraints['m0p2'] - 
                          3 * df_constraints['m0p1']**2)

# Update third moment
df_constraints['m0p3'] = (16 * df_constraints['m0p3'] -
36 * df_constraints['m0p1'] * p_var - 15 * df_constraints['m0p1']**3)

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

df_maxEnt.to_csv(datadir + 'MaxEnt_Lagrange_mult_protein_noise.csv',
                 index=False)
