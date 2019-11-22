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
figdir = f'{homedir}/fig/si/'
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

# %%
# Read protein ununregulated matrix 
with open(f'{homedir}/src/theory/pkl_files/three_state_protein_dynamics_matrix.pkl', 'rb') as file:
    A_mat_reg_lam = cloudpickle.load(file)
    expo_reg = cloudpickle.load(file)

#%%
# Define repressor copy number list
rep_array = [22, 60, 124, 260, 1220, 1740] # repressors per cell

# Define energies
operators = ['O1', 'O2', 'O3']
energies = [-15.3, -13.9, -9.7]

# Initialize data frame to save fold-changes
names = [
    "operator",
    "energy",
    "repressors",
    "iptg_uM",
    "mean_m_reg",
    "mean_m_unreg",
    "fold_change_m",
    "mean_p_reg",
    "mean_p_unreg",
    "fold_change_p",
]
df_fc_p = pd.DataFrame(columns=names)

# Loop through operators
for j, en in enumerate(energies):
    # Define binding energy
    eRA = en
    op = operators[j]
    # Loop through repressor copy numbers
    for i, rep in enumerate(rep_array):
        # Define parameters
        eRA = param[f"epR_{op}"]
        kp_on = param["kp_on"]
        kp_off = param["kp_off"]
        kr_off = param[f"kr_off_{op}"]
        ko = param["k0"]
        rm = param["rm"]
        gm = param["gm"]
        rp = param["rp"]
        ka = param["Ka"]
        ki = param["Ki"]
        epAI = param["epAI"]
        Nns = param["Nns"]

        # Calculate the repressor on rate including the MWC model
        kr_on = ko * rep * ccutils.model.p_act(0, ka, ki, epAI)

        # Generate matrices for dynamics
        # Single promoter
        par_reg_s = [kr_on, kr_off, kp_on, kp_off, rm, gm, rp, 0]
        # Two promoters
        par_reg_d = [kr_on, kr_off, kp_on, kp_off, 2 * rm, gm, rp, 0]

        # Initial conditions
        A_reg_s_init = A_mat_reg_lam(
            kr_on, kr_off, kp_on, kp_off, rm, gm, rp_init, gp_init
        )

        # Define initial conditions
        mom_init = np.zeros(len(expo_reg) * 3)
        # Set initial condition for zero moment
        # Since this needs to add up to 1
        mom_init[0] = 1

        # Define time on which to perform integration
        t = np.linspace(0, 4000 * 60, 10000)
        # Numerically integrate equations
        m_init = sp.integrate.odeint(ccutils.model.rhs_dmomdt, 
        mom_init, t, args=(A_reg_s_init,))
        # Keep last time point as initial condition
        m_init = m_init[-1, :]

        # Integrate moment equations
        df = ccutils.model.dmomdt_cycles(
            m_init,
            t_single,
            t_double,
            A_mat_reg_lam,
            par_reg_s,
            par_reg_d,
            expo_reg,
            n_cycles,
            Z_mat,
            states=["A", "I", "R"],
            n_steps=3000,
        )

        # Keep only last cycle
        df = df[df["cycle"] == df["cycle"].max()]

        # Extract index for first moment
        first_mom_names_m = [x for x in df.columns if "m1p0" in x]
        first_mom_names_p = [x for x in df.columns if "m0p1" in x]

        # Extract the last cycle information of the first moments
        df_m_reg_first = df.loc[:, first_mom_names_m]
        df_p_reg_first = df.loc[:, first_mom_names_p]

        # Extract time of last cell cycle
        time = np.sort(df["time"].unique())

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

        # Perform numerical integration
        m_mean_reg = sp.integrate.simps(
            df_m_reg_first.sum(axis=1) * p_a_array, a_array
        )
        p_mean_reg = sp.integrate.simps(
            df_p_reg_first.sum(axis=1) * p_a_array, a_array
        )

        # Compute the fold-change
        fold_change_m = m_mean_reg / m_mean_unreg
        fold_change_p = p_mean_reg / p_mean_unreg

        # Save results into series in order to append it to data frame
        series = pd.Series(
            [
                op,
                eRA,
                rep,
                0,
                m_mean,
                m_mean_unreg,
                fold_change_m,
                p_mean,
                p_mean_unreg,
                fold_change_p,
            ],
            index=names,
        )

        df_fc_p = df_fc_p.append(series, ignore_index=True)

#%%

# Define range of repressors for which compute the thermo fold-change
r_array = np.logspace(1, np.log10(2000), 50)

# Group data frame by operator
df_group = df_fc_p.groupby('energy')

# Define colors
colors = sns.color_palette('Paired', n_colors=len(df_group) * 2)

# Loop through operator
for i, (group, data) in enumerate(df_group):
    # Compute thermo fold-change
    fold_change_thermo = (1 + r_array / Nns * np.exp(- group))**-1

    # Plot thermodynamic fold-change prediction
    plt.plot(r_array, fold_change_thermo,
    label=r'$\beta\Delta\epsilon_r = {:.1f}$'.format(data.energy.unique()[0]),
             color=colors[i * 2])
    # Plot the kinetic fold-change prediciton
    # Protein
    plt.plot(data.repressors.values, data.fold_change_p.values,
             lw=0, marker='o', 
             color=colors[(i * 2) + 1], label='')
    # mRNA
    plt.plot(data.repressors.values, data.fold_change_m.values,
             lw=0, marker='v', 
             markeredgecolor=colors[(i * 2) + 1], markeredgewidth=1,
             markerfacecolor='w', label='')

# Generate labels for mRNA and protein
plt.plot([], [], lw=0, marker='v', 
         markeredgecolor='k', markeredgewidth=1,
         markerfacecolor='w', label='mRNA')
plt.plot([], [], lw=0, marker='o', 
         color='k', label='protein')

# Change scale to log
plt.xscale('log')
plt.yscale('log')

# Label axis
plt.xlabel('repressors per cell')
plt.ylabel('fold-change')

# Set legend
plt.legend(fontsize=6)

# Save figure
plt.tight_layout()
plt.savefig(figdir + 'figS09.pdf', bbox_inches='tight')