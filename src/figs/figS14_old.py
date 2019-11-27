#%%
import os
import glob
import numpy as np
import scipy as sp
import pandas as pd
import git

# Import library to perform maximum entropy fits
from maxentropy.skmaxent import FeatureTransformer, MinDivergenceModel

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

# %%
# Fit a model p(x) for dice probabilities (x=1,...,6) with the
# single constraint E(X) = 4.5
def first_moment_die(x):
    return np.array(x)


# Put the constraint functions into an array
features = [first_moment_die]
# Write down the constraints (in this case mean of 4.5)
k = np.array([4.5])

# Define the sample space of the die (from 1 to 6)
samplespace = list(range(1, 7))

# Define the minimum entropy
model = MinDivergenceModel(features, samplespace)

# Change the dimensionality of the array
X = np.atleast_2d(k)

# initialize figure
fig, ax = plt.subplots(1, 2, figsize=(6, 2.5), sharex=True, sharey=True)

# Define probability distribution of the "wrong inference"
prob = [0, 0, 0, 0.5, 0.5, 0]
# Plot the "wrong" distribution
ax[0].bar(samplespace, prob)

# Plot the max ent distribution
ax[1].bar(samplespace, model.probdist())

# Label axis
ax[0].set_xlabel("die face")
ax[1].set_xlabel("die face")

ax[0].set_ylabel("probability")

# Set title for plots
ax[0].set_title(r"$\left\langle x \right\rangle = 4.5$")
ax[1].set_title(r"MaxEnt $\left\langle x \right\rangle = 4.5$")

# Add letter label to subplots
plt.figtext(0.1, 0.93, "(A)", fontsize=8)
plt.figtext(0.50, 0.93, "(B)", fontsize=8)

plt.subplots_adjust(wspace=0.05)

plt.savefig(figdir + "figS14.pdf", bbox_inches="tight")