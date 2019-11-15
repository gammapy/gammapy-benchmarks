# Gammapy validation: Fermi-LAT 3FHL

## Task

- Reproduce some Fermi-LAT 3FHL results to check Gammapy against the Fermi ST
- Crab as example of isolated source?
- Vela pulsar with a strong spetral cutoff?
- Galactic center as example of overlapping source?
- LC from an AGN?

## References

- Dataset: https://github.com/gammapy/gammapy-fermi-lat-data/tree/master/3fhl
- Dataset: https://github.com/gammapy/gammapy-extra/tree/master/datasets/fermi_3fhl
- Tutorial: https://docs.gammapy.org/0.14/notebooks/fermi_lat.html
- Catalog info as Gammapy objects, could be useful for comparison: https://docs.gammapy.org/dev/api/gammapy.catalog.SourceCatalogObject3FHL.html

## Method

We fit a selection of ROIs defined in the 3FHL catalog (starting by the ones containing the most significant source).  We compare the resulting source parameters and flux points to the catalogued values. The spectrum and flux points comparison is limited to sources with an average significance larger than 8 sigma (Signif_Avg > 8). We also test the correlation between background normalisation and sources parameters relative error. In addition we show the observed/predicted counts and residuals maps.

- Events selection : 3FHL dataset
- Energies : 10 GeV - 2 TeV
- Spatial bins : 0.05 deg
- Energy bins : 10 per decade

## Results 

### Global Diagnostics

(90 ROIs fitted)

#### Flux points

flux_points

- Rel. err. <10%: 0.6106870229007634
- Rel. err. <30%: 0.9083969465648855
- Rel. err. mean: -0.0882757033462185

![](flux_points_errel.png)


#### Power-law spectrum

PL_index

- dx/x <10% :  0.7258064516129032
- dx/x <30% :  0.9516129032258065
- Rel. err. <10%: 0.9516129032258065
- Rel. err. <30%: 0.9838709677419355
- Rel. err. mean: 0.01706831205875889
- compatibility: 0.967741935483871

![](PL_index_corr.png)
![](PL_index_errel.png)
![](PL_index_errel_BKGcorr.png)

PL_amplitude

- dx/x <10% :  0.5
- dx/x <30% :  0.9032258064516129
- Rel. err. <10%: 0.9838709677419355
- Rel. err. <30%: 0.9838709677419355
- Rel. err. mean: -0.005484213317139602
- compatibility: 0.9838709677419355

![](PL_amplitude_corr.png)
![](PL_amplitude_errel.png)
![](PL_amplitude_errel_BKGcorr.png)

#### Log-parabola spectrums

LP_alpha

- dx/x <10% :  0.9230769230769231
- dx/x <30% :  0.9230769230769231
- Rel. err. <10%: 0.9230769230769231
- Rel. err. <30%: 0.9230769230769231
- Rel. err. mean: -0.010879249848928544
- compatibility: 0.9230769230769231

![](LP_alpha_corr.png)
![](LP_alpha_errel.png)
![](LP_alpha_errel_BKGcorr.png)

LP_beta

- dx/x <10% :  0.46153846153846156
- dx/x <30% :  0.9230769230769231
- Rel. err. <10%: 0.8461538461538461
- Rel. err. <30%: 0.9230769230769231
- Rel. err. mean: 0.03714624372322859
- compatibility: 0.9230769230769231

![](LP_beta_corr.png)
![](LP_beta_errel.png)
![](LP_beta_errel_BKGcorr.png)

LP_amplitude

- dx/x <10% :  0.9230769230769231
- dx/x <30% :  0.9230769230769231
- Rel. err. <10%: 0.9230769230769231
- Rel. err. <30%: 0.9230769230769231
- Rel. err. mean: -0.0016654427209143792
- compatibility: 0.9230769230769231

![](LP_amplitude_corr.png)
![](LP_amplitude_errel.png)
![](LP_amplitude_errel_BKGcorr.png)

### Regions plots

#### Crab region

![](counts_3FHL_ROI_num430.png)
![](npred_3FHL_ROI_num430.png)
![](resi_3FHL_ROI_num430.png)

Crab

![](spec_3FHL_J0534_5+2201_ROI_num430.png)

TXS 0518+211

![](spec_3FHL_J0521_7+2112_ROI_num430.png)

#### Vela region

![](counts_3FHL_ROI_num135.png)
![](npred_3FHL_ROI_num135.png)
![](resi_3FHL_ROI_num135.png)

Vela X

![](spec_3FHL_J0851_9-4620e_ROI_num135.png)

Vela PSR

![](spec_3FHL_J0835_3-4510_ROI_num135.png)


Vela Jr

![](spec_3FHL_J0851_9-4620e_ROI_num135.png)


#### GC region

![](counts_3FHL_ROI_num80.png)
![](npred_3FHL_ROI_num80.png)
![](resi_3FHL_ROI_num80.png)

GC

![](spec_3FHL_J1745_6-2900_ROI_num80.png)

HESS J1745-303

![](spec_3FHL_J1745_8-3028e_ROI_num80.png)

HESS J1746-285

![](spec_3FHL_J1746_2-2852_ROI_num80.png)

#### High-latitude

![](counts_3FHL_ROI_num118.png)
![](npred_3FHL_ROI_num118.png)
![](resi_3FHL_ROI_num118.png)

Note that a border source is missing

MS 1221.8+2452

![](spec_3FHL_J1224_4+2436_ROI_num118.png)

4C +21.35

![](spec_3FHL_J1224_9+2122_ROI_num118.png)

S3 1227+25

![](spec_3FHL_J1230_2_2517_ROI_num118.png)


### Convergence and fit statistic

Optimization success. :  0.02702702702702703
Optimization failed. :  0.0
Optimization failed. Estimated distance to minimum too large. :  0.972972972972973

This is more a Minuit configuration issue.
Despite the failure message the fit results seem consistent. I guess the default tolerance in Minuit is  too low.
We could add the tolerance and stragety options in optimize_minuit  :
```python
strategy = kwargs.pop("strategy", 1)
tol = kwargs.pop("tol", 0.1)
minuit = Minuit(minuit_func.fcn, **kwargs)
minuit.migrad(**migrad_opts)
minuit.tol = tol
minuit.set_strategy(strategy)
```
The few cases that return an optimization success actually have NaN parameters, we should check that in the results wrapper and fix the message.

![](Cash_stat_corr.png)


