#%%
import os
import glob
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
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
datadir = f'{homedir}/data/csv_maxEnt_dist/'

# %%
# Read predictions for linear regression to find multiplicative factor

# Read moments for multi-promoter model
df_mom_rep = pd.read_csv(datadir + 'MaxEnt_multi_prom_constraints.csv')

# Read experimental determination of noise
df_noise = pd.read_csv(f'{homedir}/data/csv_microscopy/' + 
                       'microscopy_noise_bootstrap.csv')
df_noise = df_noise[df_noise.percentile == 0.95]
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
# %%

# Read moments for multi-promoter model
df_mom_iptg = pd.read_csv(datadir + 'MaxEnt_multi_prom_IPTG_range.csv')

# Find the mean unregulated levels to compute the fold-change
mean_m_delta = np.mean(
    df_mom_iptg[df_mom_iptg.repressor==0].m1p0
)
mean_p_delta = np.mean(
    df_mom_iptg[df_mom_iptg.repressor==0].m0p1
)

# Compute the noise for the multi-promoter data
df_mom_iptg = df_mom_iptg.assign(
    m_noise=np.sqrt(df_mom_iptg.m2p0 - df_mom_iptg.m1p0**2) / 
            df_mom_iptg.m1p0,
    p_noise=np.sqrt(df_mom_iptg.m0p2 - df_mom_iptg.m0p1**2) / 
            df_mom_iptg.m0p1,
    m_fold_change=df_mom_iptg.m1p0 / mean_m_delta,
    p_fold_change=df_mom_iptg.m0p1 / mean_p_delta
)

# Read experimental determination of noise
df_noise = pd.read_csv(f'{homedir}/data/csv_microscopy/' + 
                       'microscopy_noise_bootstrap.csv')

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

#%%
# Linear regression to find multiplicative factor

# Select data with experimetal noise < 10
data = df_noise[df_noise.noise < 10]
# Define the weights for each of the datum to be the width
# of their bootstrap confidence interval.
noise_range = (data.noise_upper.values - data.noise_lower.values)

weights = noise_range
# Assign the non-zero minimum value to all zero weights
weights[weights == 0] = min(weights[weights > 0])

# Normalize weights
weights = weights / weights.sum()

def add_factor(x, a):
    '''
    Function to find additive constant used with scipy curve_fit
    '''
    return a + x

popt, pcov = sp.optimize.curve_fit(
    add_factor,
    data.noise_theory.values,
    data.noise.values,
    sigma=weights,
)

factor = popt[0]
# Print result
print(
    f"Additive factor: {factor}"
)

#%%
# Extract regulated promoter information
df_noise_reg = df_noise[df_noise.repressor > 0]
# Define repressor copy numbers to include
rep = df_noise_reg["repressor"].unique()

# Group moments by operator and repressor
df_group_exp = (
    df_noise_reg[df_noise_reg.noise > 0]
    .sort_values("IPTG_uM")
    .groupby(["operator", "repressor"])
)

df_group = (
    df_mom_iptg[df_mom_iptg["repressor"].isin(rep)]
    .sort_values("inducer_uM")
    .groupby(["operator", "repressor"])
)

# Generate index for each opeartor
operators = ["O1", "O2", "O3"]
op_idx = dict(zip(operators, np.arange(3)))

# List energies
energies = [-15.3, -13.9, -9.7]

# Generate list of colors
col_list = ["Blues_r", "Oranges_r", "Greens_r"]
# Loop through operators generating dictionary of colors for each
col_dict = {}
for i, op in enumerate(operators):
    col_dict[op] = dict(
        zip(rep, sns.color_palette(col_list[i], n_colors=len(rep) + 1)[0:3])
    )

# Define threshold to separate log scale from linear scale
thresh = 1e-1

#%%
# Initialize figure
fig, ax = plt.subplots(
    2,
    3,
    figsize=(7, 2.5),
    sharex=True,
    sharey="row",
    gridspec_kw={"height_ratios": [1, 5], "wspace": 0.05, "hspace": 0},
)
ax = ax.ravel()
# Loop through groups on multi-promoter
for i, (group, data) in enumerate(df_group):
    # Log scale
    ax[op_idx[group[0]] + 3].plot(
        data[data.inducer_uM >= thresh].inducer_uM,
        data[data.inducer_uM >= thresh].p_noise + factor,
        color=col_dict[group[0]][group[1]],
        label=int(group[1]),
    )
    # Linear scale
    ax[op_idx[group[0]] + 3].plot(
        data[data.inducer_uM <= thresh].inducer_uM,
        data[data.inducer_uM <= thresh].p_noise + factor,
        color=col_dict[group[0]][group[1]],
        label="",
        linestyle=":",
    )

# Set threshold for data
dthresh = 10
# Loop through groups on experimental data
for i, (group, data) in enumerate(df_group_exp):
    # Plot data points on lower plot
    ax[op_idx[group[0]] + 3].errorbar(
        x=data.IPTG_uM,
        y=data.noise,
        yerr=[data.noise - data.noise_lower, data.noise_upper - data.noise],
        fmt="o",
        ms=3.5,
        color=col_dict[group[0]][group[1]],
        label="",
    )
    # Plot same data points with different plotting style on the upper row
    ax[op_idx[group[0]]].plot(
        data[data.noise > dthresh].IPTG_uM,
        data[data.noise > dthresh].noise,
        linestyle="--",
        color="w",
        label="",
        lw=0,
        marker="o",
        markersize=3,
        markeredgecolor=col_dict[group[0]][group[1]],
    )

# Set scales of reference plots and the other ones will follow
ax[0].set_xscale("symlog", linthreshx=thresh, linscalex=1)
ax[0].set_yscale("log")
ax[3].set_yscale("log")

# Set limits of reference plots and the rest will folow
ax[3].set_ylim(top=8)
ax[0].set_ylim([8, 5e2])

# Set ticks for the upper plot
ax[0].set_yticks([1e1, 1e2])

# Define location for secondary legend
leg2_loc = ["lower left"] * 2 + ["upper left"]

for i in range(3):
    # Set title
    label = r"$\Delta\epsilon_r$ = {:.1f} $k_BT$".format(energies[i])
    ax[i].set_title(label, bbox=dict(facecolor="#ffedce"))
    # Label axis
    ax[i + 3].set_xlabel(r"IPTG ($\mu$M)")
    # Set legend
    leg = ax[i + 3].legend(title="rep./cell", fontsize=7)
    # Set legend font size
    plt.setp(leg.get_title(), fontsize=8)
ax[3].set_ylabel(r"noise")

plt.savefig(figdir + "figS33.pdf", bbox_inches="tight")
