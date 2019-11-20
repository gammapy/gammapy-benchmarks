# Gammapy validation: Fermi-LAT 3FHL

## Task

Reproduce some Fermi-LAT 3FHL results to check Gammapy against the Fermi ST

## References

- Dataset: https://github.com/gammapy/gammapy-fermi-lat-data/tree/master/3fhl
- Dataset: https://github.com/gammapy/gammapy-extra/tree/master/datasets/fermi_3fhl
- Tutorial: https://docs.gammapy.org/0.14/notebooks/fermi_lat.html
- Catalog info as Gammapy objects, could be useful for comparison: https://docs.gammapy.org/dev/api/gammapy.catalog.SourceCatalogObject3FHL.html

## Method

We fit a selection of ROIs defined in the 3FHL catalog (starting by the ones containing the most significant source).  We compare the resulting source parameters and flux points to the catalogued values. The spectrum and flux points comparison is limited to sources with an average significance larger than 8 sigma (Signif_Avg > 8). In addition we show the observed/predicted counts and residuals maps.

- Events selection : 3FHL dataset (7 years,  all event types)
- Energies : 10 GeV - 2 TeV
- Spatial bins : 1/8 deg
- Energy bins : 10 per decade

## Results 

### Global Diagnostics

(100 ROIs fitted)

All following values are given in percent 

Optimization terminated successfully. 100.0  
Optimization failed. 0.0  
Optimization failed. Estimated distance to minimum too large. 0.0  

 ![](./res/Cash_stat_corr.png)

flux_points  
Rel. err. <10%: 61.971830985915496  
Rel. err. <30%: 91.72535211267606  
Rel. err. mean: -5.8491302077856595  

 ![](./res/flux_points_errel.png)

PL_index  
dx/x <10% :  72.25806451612904  
dx/x <30% :  99.35483870967742  
Rel. err. <10%: 99.35483870967742  
Rel. err. <30%: 100.0  
Rel. err. mean: -0.1633147376900733  
compatibility: 100.0  

 ![](./res/PL_index_corr.png)

 ![](./res/PL_index_errel.png)

 ![](./res/PL_index_error_errel.png)

PL_amplitude  
dx/x <10% :  49.67741935483871  
dx/x <30% :  99.35483870967742  
Rel. err. <10%: 95.48387096774194  
Rel. err. <30%: 100.0  
Rel. err. mean: -0.22198785615526734  
compatibility: 98.06451612903226  

 ![](./res/PL_amplitude_corr.png)

 ![](./res/PL_amplitude_errel.png)

 ![](./res/PL_amplitude_error_errel.png)

LP_alpha  
dx/x <10% :  76.47058823529412  
dx/x <30% :  100.0  
Rel. err. <10%: 100.0  
Rel. err. <30%: 100.0  
Rel. err. mean: 0.15134907307517098  
compatibility: 100.0  

 ![](./res/LP_alpha_corr.png)

 ![](./res/LP_alpha_errel.png)

 ![](./res/LP_alpha_error_errel.png)

LP_beta  
dx/x <10% :  47.05882352941177  
dx/x <30% :  64.70588235294117  
Rel. err. <10%: 76.47058823529412  
Rel. err. <30%: 100.0  
Rel. err. mean: -4.091718544272052  
compatibility: 100.0  

 ![](./res/LP_beta_corr.png)

 ![](./res/LP_beta_errel.png)

 ![](./res/LP_beta_error_errel.png)

LP_amplitude  
dx/x <10% :  82.3529411764706  
dx/x <30% :  100.0  
Rel. err. <10%: 100.0  
Rel. err. <30%: 100.0  
Rel. err. mean: -1.3330366391977653  
compatibility: 100.0  

 ![](./res/LP_amplitude_corr.png)

 ![](./res/LP_amplitude_errel.png)

 ![](./res/LP_amplitude_error_errel.png)

### Regions plots

#### Crab region

![](./res/counts_3FHL_ROI_num430.png)
![](./res/npred_3FHL_ROI_num430.png)
![](./res/resi_3FHL_ROI_num430.png)

Crab

![](./res/spec_3FHL_J0534.5+2201_ROI_num430.png)

TXS 0518+211

![](./res/spec_3FHL_J0521.7+2112_ROI_num430.png)

#### Vela region

![](./res/counts_3FHL_ROI_num135.png)
![](./res/npred_3FHL_ROI_num135.png)
![](./res/resi_3FHL_ROI_num135.png)

Vela Jr

![](./res/spec_3FHL_J0851.9-4620e_ROI_num135.png)

Vela PSR

![](./res/spec_3FHL_J0835.3-4510_ROI_num135.png)


Vela X

![](./res/spec_3FHL_J0833.1-4511e_ROI_num135.png)







