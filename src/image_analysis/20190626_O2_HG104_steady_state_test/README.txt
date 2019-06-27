# ---
# status: accepted
# ---
# 
# # Description
# Testing the validity of the steady state assumption for the delta
# strain.
# 
# | | |
# |-|-|
# | __Date__ | 2019-06-26 |
# | __Equipment__ | Artemis Nikon Microscope |
# | __User__ | mrazomej |
# 
# ## Strain infromation
# | Genotype | plasmid | Host Strain | Shorthand |
# | :------- | :------ | :---------- | :-------- |
# | `galK<>25` | `pZS3-mCherry` | HG105 | `auto` |
# | `galK<>25-O2+11-YFP` | `pZS3-mCherry` | HG105 | `delta` |
# | `galK<>25-O2+11-YFP` | `pZS3-mCherry` | HG104 | `HG104` |
# 
# ## Titration series
# | Inducer | Concentration |
# | :------ | ------------: |
# | IPTG | 0 |
# 
# ## Microscope settings
# 
# * 100x Oil objective
# * Exposure time:
# 1. Brightfield : 15 ms
# 2. mCherry : 75 ms
# 3. YFP : 100 ms
# 
# ## Experimental protocol
# 
# The strains were grown overnight in tubes in 3 mL of LB + spec + kan.
# The tubes were then left on the bench until the next night when I
# diluted them 1:50,000 into 3 mL of M9 + 0.5% glucose in 14 mL culture
# tubes. Next morning I diluted again the strains 1:100 into 3 mL of 
# fresh media, so the cells grew in M9 for ≈ 20 hours.
# I then prepared a 2% agar pad with PBS buffer and image them with the
# usual imaging settings
# 
# ## Notes & Observations
# These cells grew extended time in M9 to guarantee that they reached the
# expected steady state
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
# **ECDF (HG104)**
# 
# ![](outdir/exp_fluor_ecdf.png)
# 
# **fold-change**
# 
# ![](outdir/fold_change.png)
