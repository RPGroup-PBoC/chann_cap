#%%
import os
import glob
import numpy as np
import scipy as sp
import pandas as pd
import re
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

#%%

# Find home directory for repo
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir

# Define directories for data and figure 
figdir = f'{homedir}/fig/si/'
datadir = f'{homedir}/data/csv_gillespie/'

# %%
df_sim_prot = pd.read_csv(datadir + "two_state_protein_gillespie.csv")

# Extract protein data
protein_names = [x for x in df_sim_prot.columns if re.match(r"[p]\d", x)]
protein_data = df_sim_prot.loc[:, protein_names].values

#%%
# Extract information from last cell cycle
idx = np.where(df_sim_prot.cycle == df_sim_prot.cycle.max())
protein_data = protein_data[idx, :]

# Define unique time points
time = df_sim_prot.iloc[idx]["time"]

# Define bins
bins = np.arange(0, protein_data.max())

# Initialize matrix to save histograms for each time point
histograms = np.zeros([len(bins) - 1, len(time)])

# Loop through time points and generate distributions
for i, t in enumerate(time):
    # Generate and save histogram
    histograms[:, i] = np.histogram(protein_data[:, i], bins, density=1)[0]
#%%
# Initialize array to save protein distribution
Pp = np.zeros(len(bins))

# Compute the time differences
time_diff = np.diff(time)

# Compute the cumulative time difference
time_cumsum = np.cumsum(time_diff)
time_cumsum = time_cumsum / time_cumsum[-1]

# Define array for spacing of cell cycle
a_array = np.zeros(len(time))
a_array[1:] = time_cumsum

# Compute probability based on this array
p_a_array = np.log(2) * 2 ** (1 - a_array)

# Loop through each of the protein copy numbers
for p in bins[:-1]:
    # Perform numerical integration
    Pp[p] = sp.integrate.simps(histograms[p, :] * p_a_array, a_array)
#%%
# Read resulting values for the multipliers.
df_maxEnt = pd.read_csv(
    "../../data/csv_maxEnt_dist/MaxEnt_Lagrange_mult_protein.csv"
)
#%%
# Extract protein moments in constraints
prot_mom = [x for x in df_maxEnt.columns if "m0" in x]
# Define index of moments to be used in the computation
moments = [tuple(map(int, re.findall(r"\d+", s))) for s in prot_mom]

# Define sample space
mRNA_space = np.array([0])
protein_space = np.arange(len(Pp))

# Extract values to be used
df_sample = df_maxEnt[
    (df_maxEnt.operator == "O1")
    & (df_maxEnt.repressor == 0)
    & (df_maxEnt.inducer_uM == 0)
]


# Select the Lagrange multipliers
lagrange_sample = df_sample.loc[
    :, [col for col in df_sample.columns if "lambda" in col]
].values[0]

# Compute distribution from Lagrange multipliers values
Pp_maxEnt = ccutils.maxent.maxEnt_from_lagrange(
    mRNA_space, protein_space, lagrange_sample, exponents=moments
).T[0]
#%%
# Define binstep for plot, meaning how often to plot
# an entry
binstep = 10

# Initialize figure
fig, ax = plt.subplots(2, 1, figsize=(3.5, 4), sharex=True)

# Plot gillespie results
ax[0].plot(bins[0::binstep], Pp[0::binstep], drawstyle="steps", color="k")
ax[0].fill_between(
    bins[0::binstep], Pp[0::binstep], step="pre", alpha=0.5, label="gillespie"
)
ax[1].plot(
    bins[0::binstep],
    np.cumsum(Pp[0::binstep]),
    drawstyle="steps",
    label="gillespie",
)

# Plot MaxEnt results
ax[0].plot(
    protein_space[0::binstep],
    Pp_maxEnt[0::binstep],
    drawstyle="steps",
    color="k",
)
ax[0].fill_between(
    protein_space[0::binstep],
    Pp_maxEnt[0::binstep],
    step="pre",
    alpha=0.5,
    label="MaxEnt",
)
ax[1].plot(
    protein_space[0::binstep],
    np.cumsum(Pp_maxEnt[0::binstep]),
    drawstyle="steps",
    label="MaxEnt",
)

# Add legend
ax[0].legend()
ax[1].legend()
# Label axis
ax[0].set_ylabel("probability")
ax[1].set_ylabel("CDF")
ax[1].set_xlabel("protein / cell")

# Change spacing between plots
plt.subplots_adjust(hspace=0.05)

plt.savefig(figdir + "figS28.pdf", bbox_inches="tight")

