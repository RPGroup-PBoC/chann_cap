#Date : 2018-04-25
#Equipment : Artemis Nikon microscope
#User : mrazomej
#Description :
#IPTG titration of the O1 - RBS1L strain.
#Strains :
#> HG105 galK::25 / pZS3-mCherry (auto)
#> HG105 galK::25-O1+11-YFP / pZS4-mCherry (delta)
#> HG104 galK::25-O1+11-YFP; ybcN::3*1-RBS1L-lacI / pZS4-mCherry
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
#> YFP : 17 ms
#
#Comments :
# This is the first dataset taken with the new plate holder. Given the stability
# I was able to mark the positions and then take all images automatically. That
# means that the structure of this data set is completely different. There is
# a single folder per pad and a stacked image of each of the positions marked
# for this particular pad.
# I also happened to discover a little bit late the autofocus feature. So most
# of the pads were not taken with automatic focus, meaning that they could
# be slightly defocus. By eye it didn't seem to be a big issue, but let's keep
# this in mind for this particular dataset.
# Also given the new setup I wasn't able to take the YFP profile images, so I
# will use the images taken in 20180424.
