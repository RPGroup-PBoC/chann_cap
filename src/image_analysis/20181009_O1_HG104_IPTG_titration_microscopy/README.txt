#Date : 2018-10-08
#Equipment : Artemis Nikon microscope
#User : mrazomej
#Description :
#IPTG titration of the O1 - HG104 strain.
#Strains :
#> HG105 galK::25 / pZS3-mCherry (auto)
#> HG105 galK::25-O2+11-YFP / pZS4-mCherry (delta)
#> HG104 galK::25-O2+11-YFP / pZS4-mCherry
#
#The strains were grown overnight in tubes in 3 mL of LB + spec + kan.
#Next morning they were diluted 1:1000 into 0.5 mL of M9 + 0.5% glucose
#The auto and delta strains were grown without and with 5000uM IPTG 
#The analysis strain was grown with the following IPTG concentrations
#IPTG concentrations:
#0uM, 0.1uM, 5uM, 10uM, 25uM, 50uM, 75uM, 100uM, 250uM, 500uM, 1000uM, 5000uM
#After 8 hours the cells were diluted 1:3 into M9 + glucose + the
#corresponding IPTG concentration and imaged
#using 2% agar pads also of M9 media.
#
#Microscope settings:
#100x Oil objective
#Exposure time:
#> Brightfield : 10 ms
#> mCherry : 50 ms (power = 1 mV)
#> YFP : 13 ms
#
#Comments :
# This data set was not saved as an *ome.tif image stack. Instead each
# individual channel was saved as an independent image inside the folder
# for their corresponding position. Therefore the processing code
# had to be adapted to read and process these images
