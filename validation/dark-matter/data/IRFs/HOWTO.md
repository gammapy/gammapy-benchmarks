# Follow these instructions to set up properly the IRFs. 
# You need to download separately the IRFs and just set a 
# variable to the right folder

# Authors:
# Gonzalo Rodriguez Fernandez: gonzalo.rodriguezfernandez@cta-consortium.org 
# Michele Doro: michele.doro@unipd.it

# All our analysis is based on Prod5.1 CTAO IRFs 
# available at https://doi.org/10.5281/zenodo.5499840

# Instructions

1. Download the IRFs at your desired place.
Link: https://doi.org/10.5281/zenodo.5499840

2. Unzip them
```
unzip cta-prod5-zenodo-fitsonly-v0.1.zip
cd fits
tar zxvf CTA-Performance-prod5-v0.1-North-20deg.FITS.tar.gz
tar zxvf CTA-Performance-prod5-v0.1-South-20deg.FITS.tar.gz
```

3. Create local folders and copy files and rename to the local folder

```
mkdir <KSP_DSPH_MATERIAL_PATH>/IRFs/cta/prod5-v0.1/bcf/North_z20_50h/
mkdir <KSP_DSPH_MATERIAL_PATH>/IRFs/cta/prod5-v0.1/bcf/South_z20_50h/
cp Prod5-North-20deg-AverageAz-4LSTs09MSTs.180000s-v0.1.fits.gz <KSP_DSPH_MATERIAL_PATH>/IRFs/cta/prod5-v0.1/bcf/North_z20_50h/irf_file.fits.gz
cp Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits.gz $KSP_DSPH_MATERIAL_PATH/IRFs/cta/prod5-v0.1/bcf/South_z20_50h/irf_file.fits.gz
```

4. Enjoy!
