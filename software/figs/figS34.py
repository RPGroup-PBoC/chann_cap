#%%
import os
import glob
import numpy as np
import scipy as sp
import pandas as pd
import re
import git

# Import libraries to parallelize processes
from joblib import Parallel, delayed

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
datadir = f'{homedir}/data/csv_maxEnt_dist/'

# Read resulting values for the multipliers.
df_maxEnt = pd.read_csv(f"{datadir}MaxEnt_Lagrange_mult_protein_var_mom.csv")

# Group by operator, repressor copy number 
# and inducer concentartion
df_group = df_maxEnt.groupby(['operator', 'binding_energy',
                              'repressor', 'inducer_uM'])

# Define names for columns in DataFrame to save KL divergences
names = ['operator', 'binding_energy', 'repressor', 
         'inducer_uM', 'num_mom', 'DKL', 'entropy']

# Initialize data frame to save KL divergences
df_kl = pd.DataFrame(columns=names)

# Define sample space
mRNA_space = np.array([0])  # Dummy space
protein_space = np.arange(0, 4E4)

# Extract protein moments in constraints
prot_mom =  [x for x in df_maxEnt.columns if 'm0' in x]
# Define index of moments to be used in the computation
moments = [tuple(map(int, re.findall(r'\d+', s))) for s in prot_mom]

# Loop through groups
for group, data in df_group:
    # Extract parameters
    op = group[0]
    eR = group[1]
    rep = group[2]
    inducer = group[3]
    
    # List different number of moments
    num_mom = data.num_mom.unique()
    
    # Initialize matrix to save probability distributions
    Pp = np.zeros([len(num_mom), len(protein_space)])
    
    # Loop through number of moments
    for i, n in enumerate(num_mom):
        # Extract the multipliers 
        df_sample = df_maxEnt[(df_maxEnt.operator == op) &
                              (df_maxEnt.repressor == rep) &
                              (df_maxEnt.inducer_uM == inducer) &
                              (df_maxEnt.num_mom == n)]
        
        # Select the Lagrange multipliers
        lagrange_sample =  df_sample.loc[:, [col for col in data.columns 
                                         if 'lambda' in col]].values[0][0:n]

        # Compute distribution from Lagrange multipliers values
        Pp[i, :] = ccutils.maxent.maxEnt_from_lagrange(mRNA_space, 
                                                       protein_space, 
                                                       lagrange_sample,
                                                    exponents=moments[0:n]).T
        
    # Define reference distriution
    Pp_ref = Pp[-1, :]
    # Loop through distributions computing the KL divergence at each step
    for i, n in enumerate(num_mom):
        DKL = sp.stats.entropy(Pp_ref, Pp[i, :], base=2)
        entropy = sp.stats.entropy(Pp[i, :], base=2)
        
        # Generate series to append to dataframe
        series = pd.Series([op, eR, rep, inducer, 
                            n, DKL, entropy], index=names)
        
        # Append value to dataframe
        df_kl = df_kl.append(series, ignore_index=True)

#%%

# Group data by operator
df_group = df_kl.groupby('operator')

# Initialize figure
fig, ax = plt.subplots(1, 3, figsize=(7, 2.5),
                       sharex=True, sharey=True)

# Define colors for operators
col_list = ['Blues_r', 'Oranges_r', 'Greens_r']
col_dict = dict(zip(('O1', 'O2', 'O3'), col_list))

# Loop through operators
for i, (group, data) in enumerate(df_group):
    # Group by repressor copy number
    data_group = data.groupby('repressor')
    # Generate list of colors
    colors = sns.color_palette(col_dict[group], n_colors=len(data_group) + 1)
    
    # Loop through repressor copy numbers
    for j, (g, d) in enumerate(data_group):
        # Plot DK divergence vs number of moments
        ax[i].plot(d.num_mom, d.DKL, color=colors[j],
                   lw=0, marker='.', label=str(int(g)))
    
    # Change scale of y axis
    ax[i].set_yscale('symlog', linthreshy=1E-6)

    # Set y axis label
    ax[i].set_xlabel('number of moments')
    # Set title
    label = r'$\Delta\epsilon_r$ = {:.1f} $k_BT$'.\
               format(data.binding_energy.unique()[0])
    ax[i].set_title(label, bbox=dict(facecolor='#ffedce'))
    # Add legend
    ax[i].legend(loc='upper right', title='rep./cell', ncol=2,
                 fontsize=6)
    
# Set x axis label
ax[0].set_ylabel('KL divergenge (bits)')

# Adjust spacing between plots
plt.subplots_adjust(wspace=0.05)
   
plt.savefig(figdir + "figS34.pdf", bbox_inches="tight")

