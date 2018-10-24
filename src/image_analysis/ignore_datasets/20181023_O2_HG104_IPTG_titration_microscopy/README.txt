# ---
# status: rejected
# reason: The fold-change didn't pass the smell test. Since the ∆lacI strain had huge differences between conditions, the fold-chage drastically change when calculated with one vs the other one. 
# ---
# 
# # Description
# IPTG titration of the O2 - HG104 strain.
# 
# | | |
# |-|-|
# | __Date__ | 2018-10-23 |
# | __Equipment__ | Artemis Nikon Microscope |
# | __User__ | mrazomej |
# 
# ## Strain infromation
# | Genotype | plasmid | Host Strain | Shorthand |
# | :------- | :------ | :---------- | :-------- |
# | `galK<>25` | `pZS3-mCherry` | HG105 | `auto` |
# | `galK<>25-O2+11-YFP` | `pZS3-mCherry` | HG105 | `delta` |
# | `galK<>25-O2+11-YFP | `pZS3-mCherry` | HG104 | `RBS1L` |
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
# 2. mCherry : 20 ms (power = 1 mV)
# 3. YFP : 11 ms
# 
# ## Experimental protocol
# 
# The strains were grown overnight in tubes in 3 mL of LB + spec + kan.
# Next morning they were diluted 1:1000 into 0.5 mL of M9 + 0.5% glucose
# in deep 96-well plates.
# The auto and delta strains were grown without IPTG.
# After 8 hours the cells were diluted 1:3 into M9 + glucose and imaged
# using 2% agar pads also of M9 media.
# 
# ## Notes & Observations
# 
# This data set was taken in reverse. Instead of starting from the
# lowest concentration I started at the highest one as a control to
# see if the time the cells remain on the pads before being image 
# affects the result of the experiment.
# HJ re-aligned the laser lines before this seesion, and there were large changes
# with the mCherry laser. So the frist pads were taken using a 50 ms exposure. I
# then downgraded to a 40, then 30 and finally to 20 ms. Since the algorithm uses
# edge detection this should not affect the final result.
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
