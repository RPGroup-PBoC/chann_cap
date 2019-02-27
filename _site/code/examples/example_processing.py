"""
Title:
    example_processing.py
Author:
    Griffin Chure and Manuel Razo-Mejia
Creation Date:
    20170220
Last Modified:
    20170220
Purpose:
    This script serves as a representative example of our data processing
    pipeline. This script reads in a set of csv files containing the output
    from the MACSQuant Flow Cytomter (after being converted from the flow
    cytometry standard .fcs format), peforms unsupervised gating, and computes
    the measured fold-change in gene expression.
"""

# Import dependencies.
import os
import glob
import numpy as np
import pandas as pd
import scipy

# Import matplotlib stuff for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Seaborn, useful for graphics
import seaborn as sns

# Set the plotting style.
import mwc_induction_utils as mwc
mwc.set_plotting_style


# Define variables to use over the script
date = 20160825
username = 'mrazomej'
run = 'r2'

# List the target directory.
datadir = 'example_data/'
files = np.array(os.listdir(datadir))
csv_bool = np.array([str(date) in f and 'csv' in f for f in files])
files = files[np.array(csv_bool)]

# define the patterns in the file names to read them
operator = 'O1'
energy = -15.3
rbs = np.array(['auto', 'delta', 'RBS1L',
                'RBS1', 'RBS1027', 'RBS446',
                'RBS1147', 'HG104'])
repressors = np.array([0, 0, 870, 610, 130, 62, 30, 11])

# Define the IPTG concentrations in units of µM.
concentrations = [0, 0.1, 5, 10, 25, 50, 75, 100, 250, 500, 1000, 5000]


# Define the parameter alpha for the automatic gating
alpha = 0.40

# Initialize the DataFrame to save the mean expression levels
df = pd.DataFrame()
# Read the files and compute the mean YFP value
for i, c in enumerate(concentrations):
    for j, strain in enumerate(rbs):
        # Find the file
        try:
            r_file = glob.glob(datadir + str(date) + '_' + run + '*' +
                               operator + '_' + strain + '_' + str(c) + 'uM' +
                               '*csv')
            print(r_file)
            # Read the csv file
            dataframe = pd.read_csv(r_file[0])
            # Apply an automatic bivariate gaussian gate to the log front
            # and side scattering
            data = mwc.auto_gauss_gate(dataframe, alpha,
                                       x_val='FSC-A', y_val='SSC-A',
                                       log=True)
            # Compute the mean and append it to the data frame along the
            # operator and strain
            df = df.append([[date, username, operator, energy,
                            strain, repressors[j], c,
                            data['FITC-A'].mean()]],
                           ignore_index=True)
        except:
            pass

# Rename the columns of the data_frame
df.columns = ['date', 'username', 'operator', 'binding_energy',
              'rbs', 'repressors', 'IPTG_uM', 'mean_YFP_A']

# Initialize pandas series to save the corrected YFP value
mean_bgcorr_A = np.array([])

# Correct for the autofluorescence background
for i in np.arange(len(df)):
    data = df.loc[i]
    auto = df[(df.IPTG_uM == data.IPTG_uM) &
              (df.rbs == 'auto')].mean_YFP_A
    mean_bgcorr_A = np.append(mean_bgcorr_A, data.mean_YFP_A - auto)

mean_bgcorr_A = pd.Series(mean_bgcorr_A)
mean_bgcorr_A.name = 'mean_YFP_bgcorr_A'
df = pd.concat([df, mean_bgcorr_A], join_axes=[df.index],
               axis=1, join='inner')
mean_fc_A = np.array([])

# Compute the fold-change
for i in np.arange(len(df)):
    data = df.loc[i]
    delta = df[(df.IPTG_uM == data.IPTG_uM) &
               (df.rbs == 'delta')].mean_YFP_bgcorr_A
    mean_fc_A = np.append(mean_fc_A, data.mean_YFP_bgcorr_A / delta)

# Convert the fold-change to a pandas DataFrame.
mean_fc_A = pd.Series(mean_fc_A)
mean_fc_A.name = 'fold_change_A'
df = pd.concat([df, mean_fc_A], join_axes=[df.index], axis=1, join='inner')

# Save the dataframe to disk as a csv including the comment header.
df.to_csv('example_nocomments_' + str(date) + '_' + run + '_' +
          operator + '_IPTG_titration_MACSQuant.csv', index=False)
filenames = ['./example_comments.txt', 'example_nocomments_' + str(date) + '_' + run +
             '_' + operator + '_IPTG_titration_MACSQuant.csv']
with open('./example_' + str(date) + '_' + run + '_' + operator +
          '_IPTG_titration_MACSQuant.csv', 'w') as output:
    for fname in filenames:
        with open(fname) as infile:
            output.write(infile.read())
