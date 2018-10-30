# ---
# status: accepted
# ---
# 
# # Description
# IPTG titration of the O2 - RBS1L strain.
# 
# | | |
# |-|-|
# | __Date__ | 2018-10-29 |
# | __Equipment__ | Artemis Nikon Microscope |
# | __User__ | mrazomej |
# 
# ## Strain infromation
# | Genotype | plasmid | Host Strain | Shorthand |
# | :------- | :------ | :---------- | :-------- |
# | `galK<>25` | `pZS3-mCherry` | HG105 | `auto` |
# | `galK<>25-O2+11-YFP` | `pZS3-mCherry` | HG105 | `delta` |
# | `galK<>25-O2+11-YFP | `pZS3-mCherry` | HG105 | `RBS1L` |
# 
# ## Titration series
# | Inducer | Concentration |
# | :------ | ------------: |
# | IPTG | 0, 0.1, 5, 10, 25, 50, 75, 100, 250, 500, 1000, 5000 [µM] |
# 
# ## Microscope settings
# 
# * 100x Oil objective
# * Exposure time:
# 1. Brightfield : 10 ms
# 2. mCherry : 30 ms (power = 1 mV)
# 3. YFP : 24 ms
# 
# ## Experimental protocol
# 
# The strains were grown overnight in tubes in 3 mL of LB + spec + kan.
# Next morning they were diluted 1:1000 into 0.5 mL of M9 + 0.5% glucose
# in deep 96-well plates.
# The auto and delta strains were grown with 5000 uM and without IPTG.
# After 8 hours the cells were diluted 1:3 into 1X PBS buffer and imaged
# using 2% agar pads also of 1X PBS buffer.
# 
# ## Notes & Observations
# 
# This dataset was taken in reverese with higher IPTG concentrations imaged
# before lower concentrations.
# In addition for this dataset the dilution just before imaging was done 1:3 into
# 1X PBS buffer rather than the usual M9 media to limit the amount of carbon that
# the cells are expose to during the imaging session. The agar pads into which
# the cells were mounted were also made out of 1x PBS with the corresponding IPTG
# concentration.
# 
# ## Analysis files
# 
# **Example segmentation**
# 
# ![](outdir/example_segmentation.png)
# 
# **ECDF (auto)**
# 
# ![](outdir/auto_fluor_ecdf.png)
# 
# **ECDF (∆lacI)**
# 
# ![](outdir/delta_fluor_ecdf.png)
# 
# **ECDF (RBS1L)**
# 
# ![](outdir/exp_fluor_ecdf.png)
# 
# **fold-change**
# 
# ![](outdir/fold_change.png)
