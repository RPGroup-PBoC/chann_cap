# ---
# status: accepted
# ---
# 
# # Description
# IPTG titration of the O2 - RBS1L strain.
# 
# | | |
# |-|-|
# | __Date__ | 2016-12-03 |
# | __Equipment__ | Artemis Nikon Microscope |
# | __User__ | mrazomej |
# 
# ## Strain infromation
# | Genotype | plasmid | Host Strain | Shorthand |
# | :------- | :------ | :---------- | :-------- |
# | `galK<>25` | `pZS3-mCherry` | HG105 | `auto` |
# | `galK<>25-O2+11-YFP` | `pZS3-mCherry` | HG105 | `delta` |
# | `galK<>25-O2+11-YFP; ybcN<>4*RBS1L-lacI` | `pZS3-mCherry` | HG105 | `RBS1L` |
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
# 1. Brightfield : 3 ms
# 2. mCherry : 22 ms
# 3. YFP : 9 ms
# 
# ## Experimental protocol
# 
# The strains were grown overnight in tubes in 3 mL of LB + spec + kan.
# Next morning they were diluted 1:1000 into 0.5 mL of M9 + 0.5% glucose
# in deep 96-well plates.
# The auto and delta strains were grown without IPTG.
# After 8 hours the cells were diluted 1:10 into M9 + glucose and imaged
# using 2% agar pads also of M9 media.
# 
# ## Notes & Observations
# It was harder than it should have been to find cells. Next time I should
# do a smaller dilution.
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
