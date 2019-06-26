import os
#============================================================================== 
# METADATA
#============================================================================== 
# Defualt username
USERNAME = 'mrazomej'

# IPTG RANGE
IPTG_RANGE = (0,)
# Extra feature because my mistake when naming files
IPTG_NAMES = ('0',)

IPTG_DICT = dict(zip(IPTG_NAMES, IPTG_RANGE))

# Get current directory to extract information
directory = os.getcwd().split('/')[-1]
# Split by underscore to obtain values
values = directory.split('_')

# Extract directory info
DATE = int(values[0])
OPERATOR = values[1]
STRAIN = values[2]

# Define binding energy and represor values
# according to strain
binding_dict = {'O1': -15.3, 'O2': -13.9, 'O3': -9.7}
rep_dict = {'HG104': 22, 'RBS1027': 260, 'RBS1L': 1740}

# Binding energy from OPERATOR
BINDING_ENERGY = binding_dict[OPERATOR]

# Repressor copy number form STRAIN
REPRESSOR = rep_dict[STRAIN]
REPRESSORS = [0, 0] + [REPRESSOR]
IPDIST = 0.069  # in units of µm per pixel
STRAINS = ['auto', 'delta'] + [STRAIN]

