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
figdir = f'{homedir}/fig/main/'
datadir = f'{homedir}/data/csv_maxEnt_dist/'

# %%
df_cc_protein = pd.read_csv(
    f"{homedir}/data/csv_maxEnt_dist/chann_cap_multi_prom_protein.csv"
)

# Drop infinities
df_cc_protein = df_cc_protein[df_cc_protein.channcap != np.inf]

# Generate list of colors for each operator
col_list = sns.color_palette("colorblind", n_colors=3)
col_dict = dict(zip(["O1", "O2", "O3"], col_list))
op_dict = dict(zip(df_cc_protein.operator.unique(),
                   df_cc_protein.binding_energy.unique()))

# Define directory where data is stored
expdir = f"{homedir}/data/csv_channcap_bootstrap/"

# Define directory where the bootstrap data was stored
bootsdir = f"{homedir}/src/channcap_exp/"

# List files of data taken exclusively for this experiment
bootsfiles = [
    x
    for x in os.listdir(bootsdir)
    if ("channel_capacity_experiment" in x) & ("ipynb" not in x)
]

# Extract dates for these experiments
project_dates = [x.split("_")[0] for x in bootsfiles]


# List files with the bootstrap sampling of the
files = glob.glob(f"{expdir}*channcap_bootstrap.csv")

# Extract dates from these files
file_dates = [file.split("/")[-1] for file in files]
file_dates = [file.split("_")[0] for file in file_dates]

##  Remove data sets that are ignored because of problems with the data quality
##  NOTE: These data sets are kept in the repository for transparency, but they
##  failed at one of our quality criteria
## (see README.txt file in microscopy folder)
ignore_files = [
    x
    for x in os.listdir(f"{homedir}/src/image_analysis/ignore_datasets/")
    if "microscopy" in x
]
# Extract data from these files
ignore_dates = [x.split("_")[0] for x in ignore_files]

# Filter for files taken exclusively for this experiment.
files = [
    file
    for i, file in enumerate(files)
    if (file_dates[i] in project_dates) & (not file_dates[i] in ignore_dates)
]

#%%

# Define dictionaries to map operator to binding energy and rbs to rep copy
op_dict = dict(zip(["O1", "O2", "O3", "Oid"], [-15.3, -13.9, -9.7, -17]))
rbs_dict = dict(
    zip(
        ["HG104", "RBS1147", "RBS446", "RBS1027", "RBS1", "RBS1L"],
        [22, 60, 124, 260, 1220, 1740],
    )
)

# Define index of entries to save
index = [
    "date",
    "bins",
    "operator",
    "rbs",
    "binding energy",
    "repressors",
    "channcap",
]
# Initialize DataFrame to save information
df_cc_exp = pd.DataFrame(columns=index)

# Define bin number to extract
bin_target = 100

# Loop through files
for f in files:
    # Split file name to extract info
    str_split = f.replace(expdir, "").split("_")
    # Extract date, operator and rbs info
    date, op, rbs = str_split[0:3]
    # Map the binding energy and repressor copy number
    eRA, rep = op_dict[op], rbs_dict[rbs]

    # Read file
    df_cc_bs = pd.read_csv(f, header=0)

    # Select df_cc_bs closest to desired number of bins
    # Find the index of the min df_cc_bs
    bin_idx = (np.abs(df_cc_bs["bins"] - bin_target)).idxmin()
    # Choose the bind number
    bin_num = df_cc_bs.iloc[bin_idx]["bins"]

    # Keep only df_cc_bs with this bin number
    df_cc_bs = df_cc_bs[df_cc_bs["bins"] == bin_num]

    # Extrapolate to N -> oo
    x = 1 / df_cc_bs.samp_size
    y = df_cc_bs.channcap_bs
    # Perform linear regression
    lin_reg = np.polyfit(x, y, deg=1)
    # Extract intercept to find channel capacity estimate
    cc = lin_reg[1]

    # Compile info into a pandas series to append it to the DataFrame
    series = pd.Series([date, bin_num, op, rbs, eRA, rep, cc], index=index)
    # Append to DataFrame
    df_cc_exp = df_cc_exp.append(series, ignore_index=True)

#%%

# Group data by operator
df_group = df_cc_protein.groupby("operator")

# Define colors for each operator
operators = df_cc_protein["operator"].unique()
colors = sns.color_palette("colorblind", n_colors=len(operators))
op_col_dict = dict(zip(operators, colors))

# Define threshold for log vs linear section
thresh = 1e0

fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.8))
for group, data in df_group:
    # Select x and y data for smoothing
    x = np.log10(data[data.repressor >= thresh].repressor.values)
    y = data[data.repressor >= thresh].channcap.values
    # Define lambda parameter for smoothing
    lam = 0.21
    # Smooth the channel capacity
    channcap_gauss = ccutils.stats.nw_kernel_smooth(x, x, y, lam)
    # Plot Log scale
    ax.plot(
        data[data.repressor >= thresh].repressor,
        channcap_gauss,
        label=op_dict[group],
        color=col_dict[group],
    )
    # Plot data from operator
    ax.plot(
        df_cc_exp[df_cc_exp["operator"] == group]["repressors"],
        df_cc_exp[df_cc_exp["operator"] == group]["channcap"],
        lw=0,
        marker="o",
        color=op_col_dict[group],
        label="",
        alpha=0.8,
        markeredgecolor="black",
        markeredgewidth=1,
    )

# Label plot
ax.set_xlabel("repressor copy number")
ax.set_ylabel("channel capacity (bits)")
ax.set_xscale("log")
ax.legend(loc="upper left", title=r"$\Delta\epsilon_r \; (k_BT)$")

# Upate axis range
ax.set_ylim([-0.05, 2.2])

# Save figure
plt.savefig(figdir + "fig05A.pdf", bbox_inches="tight")
plt.savefig(figdir + "fig05A.png", bbox_inches="tight")
