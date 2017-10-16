# operating system packages
import os
import sys
import glob
import re
from itertools import compress

# numerical workhorses
import scipy.io
import numpy as np
import pandas as pd

# Import matplotlib stuff for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Seaborn, useful for graphics
import seaborn as sns

sys.path.insert(0, '../theory/')
import chann_cap_utils as channcap
channcap.set_plotting_style()


# %% Define the directory where the data lives
datadir = '../../data/mRNA_FISH/constitutive_expression/'

# Find all dates in which data was measured
dates = [x for x in os.listdir(datadir) if '20' in x]

# List all the file tree in the directory
files = [x for x in os.walk(datadir)]

# %% Find entries with .mat files
# Initialize list to append mat files
mat_files = []
# Loop through files
for i, f in enumerate(files):
    # Find if the list is empty
    if not f[2]:
        continue
    if 'mat' in f[2][0]:
        mat_files.append(f[0] + '/' + f[2][0])

# Let's check that all files contain the same entries
# mat_keys = [scipy.io.loadmat(x).keys() for x in mat_files]
# mat_content = [x == y for x in mat_keys for y in mat_keys]
# all(mat_content)

# Define fields to be extracted from dictionaries
fields = ['area_cells', 'spots_totals']

# Initialize pandas DataFrame to collect all the data
columns = ['date', 'experiment'] + fields
df = pd.DataFrame(columns=columns)

# %% Loop through mat files identifying the date and the "promoter condition"
for i, f in enumerate(mat_files):
    # Split string by '/'
    split_str = f.split('/')
    # Find the date of this file
    date = list(compress(split_str, [x in dates for x in split_str]))
    # Remove the dash symbol
    date = re.sub('-', '', date[0])
    # Find the position of the 'analysis_results' entry since the previous
    # entry defines the "promoter condition"
    prom_idx = np.where([x == 'analysis_results' for x in split_str])[0] - 1
    prom = [split_str[int(prom_idx)]]
    # Read .mat file and extract the relevant fields
    mat_data = scipy.io.loadmat(f)
    # Extract relevant fields
    data = dict(zip(fields, [mat_data[x].ravel() for x in fields]))
    # generate DataFrame to collect this file data
    df_f = pd.DataFrame.from_dict(data)
    # Append data for file (converting it to a number)
    df_f['date'] = [int(date)] * len(df_f)
    # Append experiment for file
    df_f['experiment'] = prom * len(df_f)
    # Append data to general DataFrame
    df = pd.concat([df, df_f], ignore_index=True)

df.to_csv('../../data/mRNA_FISH/Jones_Brewster_2014.csv')
