import pickle
import os
import glob
import datetime

# Our numerical workhorses
import numpy as np
from sympy import mpmath
import scipy.optimize
import scipy.special
import scipy.integrate
import pandas as pd
import itertools

# Import libraries to parallelize processes
from joblib import Parallel, delayed

# Import the utils for this project
import chann_cap_utils as chann_cap

# Protein parameters
#k0 = 2.7E-3 # From Jones & Brewster
k0 = 0.0002409 # From fitting to the O2-RBS1027 microscopy data
prot_params = dict(ka=139.55, ki=0.53, epsilon=4.5,
                   kon=chann_cap.kon_fn(-9.7, k0),
                   k0=k0,
                   gamma_m=0.00284, r_gamma_m=15.7,
                   gamma_p=0.000277, r_gamma_p=100)

# Define the protein blocks to evaluate in parallel
# Break into blocks to compute the distributions in parallel
prot_grid = np.reshape(np.arange(0, 4000), [-1, 50])

# define the array of repressor copy numbers to evaluate the function in
R_array = np.arange(0, 1050)

# Setting the kon parameter based on k0 and the binding energies form stat. mech.
kon_array = [chann_cap.kon_fn(-13.9, prot_params['k0']),
             chann_cap.kon_fn(-15.3, prot_params['k0']),
             chann_cap.kon_fn(-9.7, prot_params['k0']),
             chann_cap.kon_fn(-17, prot_params['k0'])]
kon_operators = ['O2', 'O1', 'O3', 'Oid']
kon_dict = dict(zip(kon_operators, kon_array))

compute_matrix = True
if compute_matrix:
    for kon, op in enumerate(kon_operators):
        print('operator : ' + op)
        # Set the value for the kon
        prot_params['kon'] = kon_dict[op]
        # Define filename
        file = '../../data/csv_protein_dist/lnp_' + op + '_O2_RBS1027_fit.csv'
	# If the file exists read the file, find the maximum number of repressors
	# And compute from this starting point.
        if os.path.isfile(file): 
            df = pd.read_csv(file, index_col=0)
            max_rep = df.repressor.max()
            df = df[df.repressor != max_rep]
            df.to_csv(file)
            r_array = np.arange(max_rep, np.max(R_array) + 1)
        else:
            r_array = R_array

        # Loop through repressor copy numbers
        for i, r in enumerate(r_array):
            if r%50==0:
                print('repressors : {:d}'.format(r))
            prot_params['rep'] = r * 1.66
            # -- Parallel computation of distribution -- #
            # define a function to run in parallel the computation
            def lnp_parallel(p):
                lnp = chann_cap.log_p_p_mid_C(C=0, protein=p, **prot_params)
                df = pd.DataFrame([r] * len(p), index=p, columns=['repressor'])
                df.loc[:, 'protein'] = pd.Series(p, index=df.index)
                df.loc[:, 'lnp'] = lnp
                
                # if file does not exist write header 
                if not os.path.isfile(file): 
                    df.to_csv(file) 
                else: # else it exists so append without writing the header
                    df.to_csv(file, mode='a', header=False)
            Parallel(n_jobs=40)(delayed(lnp_parallel)(p) for p in prot_grid)
