#%%
import os
import pickle
import cloudpickle
import itertools
import glob
import git
import numpy as np
import scipy as sp
import pandas as pd
from joblib import Parallel, delayed
import ccutils

# Find home directory for repo
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir

datadir = f'{homedir}/data/csv_maxEnt_dist/'
#%%
# Read protein ununregulated matrix 
with open('../pkl_files/three_state_protein_dynamics_matrix.pkl',
          'rb') as file:
    A_mat_reg_lam = cloudpickle.load(file)
    expo_reg = cloudpickle.load(file)

# Read matrix into memory
with open('../pkl_files/binom_coeff_matrix.pkl', 'rb') as file:
    unpickler = pickle.Unpickler(file)
    Z_mat = unpickler.load()
    expo_binom = unpickler.load()
#%%

# Load constants
param = ccutils.model.load_constants()
# Define experimental concentrations in µM
inducer = [0, 0.1, 5, 10, 25, 50, 75, 100, 250, 500, 1000, 5000] # µM
# Define repressor copy numebers
repressors = np.logspace(0, np.log10(2000), 50)
repressors = np.unique(repressors.round(0))

# Include repressor copy numbers for experimentally available strains
rep = [0, 22, 60, 124, 260, 1220, 1740]
# Add them to list
repressors =  np.sort(np.concatenate([repressors, rep]))

# Define operators and energies
operators = ['O1', 'O2', 'O3']
# Generate list of all operator, repressor and inducer concentrations
var =  [t for t in itertools.product(*[operators, repressors, inducer])]

# Define doubling time
doubling_time = 100
# Define fraction of cell cycle spent with one copy
t_single_frac = 0.6
# Define time for single-promoter state
t_single = 60 * t_single_frac * doubling_time # sec
t_double = 60 * (1 - t_single_frac) * doubling_time # sec
n_cycles = 6
#%%
# Initialize data frame to save the distribution moments.
names = ['operator', 'binding_energy', 'repressor', 'inducer_uM']
names = names + ['m' + str(m[0]) + 'p' + str(m[1]) for m in expo_reg]         

# Initialize DataFrame to save constraints
df_constraints = pd.DataFrame([], columns=names)

# Define function for parallel computation
def constraints_parallel(par):
    # Extract variables
    op = par[0]  #operator
    eRA = param[f'epR_{op}']  # binding energy
    rep = par[1]  # repressors
    iptg = par[2]  # inducer
    print(op, eRA, rep, iptg)

    # Load parameters
    ka = param['Ka']
    ki = param['Ki']
    epAI = param['epAI']
    Nns = param['Nns']
    Vcell = param['Vcell']
    kp_on = param['kp_on']
    kp_off = param['kp_off']
    rm = param['rm']
    gm = param['gm']
    rp = param['rp']
    ko = param['k0']
    # Single promoter
    gp_init = 1 / (60 * 60)
    rp_init = 500 * gp_init

    # Calculate the repressor on rate including the MWC model
    kr_on = ko * rep * ccutils.model.p_act(iptg, ka, ki, epAI) 

    # Compute the repressor off-rate based on the on-rate and 
    # the binding energy
    kr_off = ccutils.model.kr_off_fun(eRA, ko, kp_on, kp_off,
                                      Nns, Vcell)

    # Generate matrices for dynamics
    # Single promoter
    par_reg_s = [kr_on, kr_off, kp_on, kp_off, rm, gm, rp, 0]
    # Two promoters
    par_reg_d = [kr_on, kr_off, kp_on, kp_off, 2 * rm, gm, rp, 0]
    
    # Initial conditions
    A_reg_s_init = A_mat_reg_lam(kr_on, kr_off, kp_on, kp_off,
                                 rm, gm, rp_init, gp_init)
    
    # Define initial conditions
    mom_init = np.zeros(len(expo_reg) * 3)
    # Set initial condition for zero moment
    # Since this needs to add up to 1
    mom_init[0] = 1

    # Define time on which to perform integration
    t = np.linspace(0, 4000 * 60, 10000)
    # Numerically integrate equations
    m_init = sp.integrate.odeint(ccutils.model.rhs_dmomdt, mom_init, t, 
                                    args=(A_reg_s_init,))
    # Keep last time point as initial condition
    m_init = m_init[-1, :]
    
    # Integrate moment equations
    df = ccutils.model.dmomdt_cycles(m_init, t_single, t_double,
                                     A_mat_reg_lam, 
                                     par_reg_s, par_reg_d,
                                     expo_reg, n_cycles, Z_mat,
                                     states=['A', 'I', 'R'], 
                                     n_steps=3000)
    
    # Keep only last cycle
    df = df[df['cycle'] == df['cycle'].max()]

    # Extract time of last cell cycle
    time = np.sort(df['time'].unique())

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

    # Initialize list to append moments
    moms = list()
    # Loop through moments computing the average moment
    for i, mom in enumerate(expo_reg):
        # Generate string that finds the moment
        mom_name = 'm' + str(mom[0]) + 'p' + str(mom[1])
        # List rows with moment
        mom_bool = [x for x in df.columns if mom_name in x]
        # Extract data for this particular moment
        df_mom = df.loc[:, mom_bool].sum(axis=1)

        # Average moment and append it to list
        moms.append(sp.integrate.simps(df_mom * p_a_array,
                                        a_array))
            
    # Save results into series in order to append it to data frame
    series = pd.Series([op, eRA, rep, iptg] + moms,
                index=names)

    return series

# Run function in parallel
constraint_series = Parallel(n_jobs=6)(delayed(constraints_parallel)(par)
                             for par in var)

# Initialize data frame to save list of pareters
df_constraints = pd.DataFrame([], columns=names)

for s in constraint_series:
    df_constraints = df_constraints.append(s, ignore_index=True)

# Save progress at each step
df_constraints.to_csv(f'{datadir}MaxEnt_multi_prom_constraints.csv',
            index=False)

print('done!')