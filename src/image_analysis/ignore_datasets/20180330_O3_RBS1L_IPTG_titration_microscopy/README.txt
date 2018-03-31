#Date : 2018-03-30
#Equipment : Artemis Nikon microscope
#User : mrazomej
#Description :
#IPTG titration of the O3 - RBS1L strain.
#Strains :
#> HG105 galK::25 / pZS3-mCherry (auto)
#> HG105 galK::25-O3+11-YFP / pZS4-mCherry (delta)
#> HG105 galK::25-O3+11-YFP; ybcN::3*1-RBS1L-lacI / pZS4-mCherry
#
#The strains were grown overnight in tubes in 3 mL of LB + spec + kan.
#The RBS1L strain was grown with chloramphenicol too to avoid possible cross
#contamination as the data from 20180328 suggested
#Next morning they were diluted 1:1000 into 0.5 mL of M9 + 0.5% glucose
#also in deep 96-well plates.
#The auto and delta strains were grown without IPTG and with 5000 uM IPTG
#The analysis strain was grown with the following IPTG concentrations
#IPTG concentrations:
#0uM, 0.1uM, 5uM, 10uM, 25uM, 50uM, 75uM, 100uM, 250uM, 500uM, 1000uM, 5000uM
#After 8 hours the cells were diluted 1:10 into M9 + glucose and imaged
#using 2% agar pads also of M9 media.
#
#Microscope settings:
#100x Oil objective
#Exposure time:
#> Brightfield : 3 ms
#> mCherry : 22 ms
#> YFP : 10 ms
#
#Comments :
# The RBS1L strain was grown with chloramphenicol.
# There was a problem with the Delta strain at 0uM IPTG. It was probably
# inoculated with auto because it was not fluorescent at all.
# Given this I moved to the Delta 5000uM IPTG and set the exposure based on
# this strain. I then realized that it was too high and many of the samples
# reached saturation.
# There were a very small number of contaminations in the auto 5000uM IPTG that
# were really bright in YFP.
# This is not a good dataset.
