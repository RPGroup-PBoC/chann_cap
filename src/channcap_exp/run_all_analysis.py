import os
import glob
import numpy as np

# List directories
dirs = np.sort(glob.glob('./*_channel_capacity*'))

# Loop through directories
for d in dirs:
    os.chdir(f'{d}/')
    print(os.getcwd())
    os.system('python processing.py')
    os.chdir('../')
