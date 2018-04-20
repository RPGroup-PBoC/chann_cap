#Date : 2018-04-19
#Equipment : Artemis Nikon microscope
#User : mrazomej
#Description :
#IPTG titration of the O3 - HG104 strain.
#Strains :
#> HG105 galK::25 / pZS3-mCherry (auto)
#> HG105 galK::25-O3+11-YFP / pZS4-mCherry (delta)
#> HG104 galK::25-O3+11-YFP / pZS4-mCherry
#
#The strains were grown overnight in tubes in 3 mL of LB + spec + kan.
#Next morning they were diluted 1:1000 into 0.5 mL of M9 + 0.5% glucose
#also in deep 96-well plates.
#The auto and delta strains were grown without IPTG and with 5000uM
#The analysis strain was grown with the following IPTG concentrations
#IPTG concentrations:
#0uM, 0.1uM, 5uM, 10uM, 25uM, 50uM, 75uM, 100uM, 250uM, 500uM, 1000uM, 5000uM
#After 8 hours the cells were diluted 1:10 into M9 + glucose and imaged
#using 2% agar pads also of M9 media.
#
#Microscope settings:
#100x Oil objective
#Exposure time:
#> Brightfield : 10 ms
#> mCherry : 10 ms
#> YFP : 7 ms
#
#Comments :
# This is the first data set taken with the new configuration where
# the laser doesn't turn off, but it is through the rotation of the
# filter that the exposure time is set. This is to minimize the
# variability in the exposure time since micro-manager has a delay
# of order 10 ms between hardware communication according to Heun Jin.
# This new set up should reduce that noise source.
