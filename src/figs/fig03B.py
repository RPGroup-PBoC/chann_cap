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
figdir = f'{homedir}/fig/main/'
datadir = f'{homedir}/data/csv_maxEnt_dist/'

# %%
# Read matrix for binomial partitioning into memory
with open(f'{homedir}/src/theory/pkl_files/binom_coeff_matrix.pkl', 
'rb') as file:
    unpickler = pickle.Unpickler(file)
    Z_mat = unpickler.load()
    expo_binom = unpickler.load()

# %%
# Load constants
param = ccutils.model.load_constants()

# Integrate dynamics for single promoter steady state

gp_init = 1 / (60 * 60)
rp_init = 500 * gp_init

# Read protein ununregulated matrix 
with open(f'{homedir}/src/theory/pkl_files/two_state_protein_dynamics_matrix.pkl', 'rb') as file:
    # Load sympy object containing the matrix A that define the
    # moment dynamics
    A_mat_unreg_lam = cloudpickle.load(file)
    # Load the list of moments included in the matrix
    expo = cloudpickle.load(file)
    
# Substitute value of parameters on matrix
##  Initial conditions
A_mat_unreg_s_init = A_mat_unreg_lam(param['kp_on'], param['kp_off'],
                                     param['rm'], param['gm'],
                                     rp_init, gp_init)

# Define time on which to perform integration
t = np.linspace(0, 4000 * 60, 2000)

# Define initial conditions
mom_init = np.zeros(len(expo) * 2)
# Set initial condition for zero moment
# Since this needs to add up to 1
mom_init[0] = 1

# Numerically integrate equations
mp_sol = sp.integrate.odeint(ccutils.model.rhs_dmomdt, mom_init, t, 
                             args=(A_mat_unreg_s_init,))

mp_init = mp_sol[-1, :]
#%%
# Define doubling time
doubling_time = 100
# Define fraction of cell cycle spent with one copy
t_single_frac = 0.6
# Define time for single-promoter state
t_single = 60 * t_single_frac * doubling_time # sec
t_double = 60 * (1 - t_single_frac) * doubling_time # sec

# Define number of cell cycles
n_cycles = 6

# Define list of parameters
par_single = [param['kp_on'], param['kp_off'], param['rm'], param['gm'],
              param['rp'], 0]
par_double = [param['kp_on'], param['kp_off'], 2 * param['rm'],
              param['gm'], param['rp'], 0]

# Integrate moment equations
df_p_unreg = ccutils.model.dmomdt_cycles(mp_init, t_single, t_double, 
                           A_mat_unreg_lam, 
                           par_single, par_double, expo,
                           n_cycles, Z_mat, n_steps=10000)

# Extract index for mRNA and protein first moment
first_mom_names_m = [x for x in df_p_unreg.columns
                     if 'm1p0' in x]
first_mom_names_p = [x for x in df_p_unreg.columns
                     if 'm0p1' in x]

# Extract the last cycle information
df_m_unreg_first = df_p_unreg.loc[df_p_unreg.cycle == df_p_unreg.cycle.max(),
                                  first_mom_names_m]
df_p_unreg_first = df_p_unreg.loc[df_p_unreg.cycle == df_p_unreg.cycle.max(),
                                  first_mom_names_p]
# Extract time of last cell cycle
time = np.sort(df_p_unreg.loc[df_p_unreg.cycle == 
                              df_p_unreg.cycle.max(),
                              'time'].unique())

# Compute the time differences
time_diff = np.diff(time)

# Compute the cumulative time difference
time_cumsum = np.cumsum(time_diff)
time_cumsum = time_cumsum / time_cumsum[-1]

# Define array for spacing of cell cycle
a_array = np.zeros(len(time))
a_array[1:] = time_cumsum 

# Compute probability based on this array
p_a_array = np.log(2) * 2**(1 - a_array)

# Perform numerical integration
m_mean_unreg = sp.integrate.simps(df_m_unreg_first.sum(axis=1) * p_a_array,
                                  a_array)
p_mean_unreg = sp.integrate.simps(df_p_unreg_first.sum(axis=1) * p_a_array,
                                  a_array)

# %%
# Extract index for first moment
first_mom_names_m = [x for x in df_p_unreg.columns if 'm1p0' in x]
first_mom_names_p = [x for x in df_p_unreg.columns if 'm0p1' in x]

# Compute the mean mRNA copy number
m_mean = df_p_unreg.loc[:, first_mom_names_m].sum(axis=1)
p_mean = df_p_unreg.loc[:, first_mom_names_p].sum(axis=1)

# Extrac second moment
second_mom_names_m = [x for x in df_p_unreg.columns if 'm2p0' in x]
second_mom_names_p = [x for x in df_p_unreg.columns if 'm0p2' in x]

# Compute the second moments
m_second = df_p_unreg.loc[:, second_mom_names_m].sum(axis=1)
p_second = df_p_unreg.loc[:, second_mom_names_p].sum(axis=1)

# Compute variance
m_var = m_second - m_mean**2
p_var = p_second - p_mean**2

# %%

# Define colors
colors = sns.color_palette('Paired', n_colors=2)

# Initialize figure
fig, ax = plt.subplots(2, 1, figsize=(2.5, 2), sharex=True)

# Plot mean as solid line
ax[0].plot(df_p_unreg.time / 60, m_mean, label='', lw=1.25,
           color=colors[1])
ax[1].plot(df_p_unreg.time / 60, p_mean, label='', lw=1.25,
           color=colors[1])

# Plot +- standard deviation 
ax[0].fill_between(df_p_unreg.time / 60, 
                   y1=m_mean + np.sqrt(m_var),
                   y2=m_mean - np.sqrt(m_var),
                   label='', color=colors[0], alpha=0.85,
                   zorder=2)
ax[1].fill_between(df_p_unreg.time / 60, 
                   y1=p_mean + np.sqrt(p_var),
                   y2=p_mean - np.sqrt(p_var),
                   label='', color=colors[0], alpha=0.85,
                   zorder=2)

# Group data frame by cell cycle
df_group = df_p_unreg.groupby('cycle')


# Loop through cycles
for i, (group, data) in enumerate(df_group):
    # Define the label only for the last cell cycle not to repeat in legend
    if group == df_p_unreg['cycle'].max():
        label_s = 'single promoter'
        label_d = 'two promoters'
    else:
        label_s = ''
        label_d = ''
    # Find index for one-promoter state
    idx = np.where(data.state == 'single')[0]
    # Indicate states with two promoters
    ax[0].axvspan(data.iloc[idx.min()]['time'] / 60, 
                  data.iloc[idx.max()]['time'] / 60,
               facecolor='#e3dcd1', label=label_s)
    ax[1].axvspan(data.iloc[idx.min()]['time'] / 60, 
                  data.iloc[idx.max()]['time'] / 60,
               facecolor='#e3dcd1', label='')
    
    # Find index for two-promoter state
    idx = np.where(data.state == 'double')[0]
    # Indicate states with two promoters
    ax[0].axvspan(data.iloc[idx.min()]['time'] / 60,
                  data.iloc[idx.max()]['time'] / 60,
               facecolor='#ffedce', label=label_d)
    ax[1].axvspan(data.iloc[idx.min()]['time'] / 60,
                  data.iloc[idx.max()]['time'] / 60,
               facecolor='#ffedce', label='')

##  Indicate where the cell divisions happen
# First find where the cell cycle transition happen
trans_idx = np.array(np.diff(df_p_unreg.cycle) == 1)
# Add extra point to have same length
trans_idx = np.insert(trans_idx, 0, False)  
# Get the time points at which this happens
time_div = df_p_unreg[trans_idx].time.values
# Plot with a triangle the cell division moment
ax[0].plot(time_div / 60, [np.max(m_mean) * 1.1] * len(time_div),
           lw=0, marker='v', color='k')

# Set limits
# mRNA
ax[0].set_xlim(df_p_unreg['time'].min() / 60, df_p_unreg['time'].max() / 60)
ax[0].set_ylim([0, 40])
#protein
ax[1].set_xlim(df_p_unreg['time'].min() / 60, df_p_unreg['time'].max() / 60)
ax[1].set_ylim([5000, 14000])

# Label plot
ax[1].set_xlabel('time (min)')
ax[0].set_ylabel(r'$\left\langle \right.$mRNA$\left. \right\rangle$/cell')
ax[1].set_ylabel(r'$\left\langle \right.$protein$\left. \right\rangle$/cell')

# Align y axis labels
fig.align_ylabels()

# Set legend for both plots
ax[0].legend(loc='upper left', ncol=2, frameon=False,
             bbox_to_anchor=(-0.12, 0, 0, 1.3), fontsize=6.5)

plt.subplots_adjust(hspace=0.05)
plt.savefig(figdir + 'fig03B.svg', bbox_inches='tight',
            transparent=True)
plt.savefig(figdir + 'fig03B.png', bbox_inches='tight',
            transparent=True)

# %%
