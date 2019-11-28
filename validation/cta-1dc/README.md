# Gammapy: CTA 1DC validation

## Overview

As described [here](https://forge.in2p3.fr/projects/data-challenge-1-dc-1/wiki/Current_capabilities_and_limitations_of_the_analysis_tools),
the following test cases were chosen for the validation:

* [cas_a](cas_a) - Cassiopeia A
* [gc](gc) - Galactic center
* [rx_j1713](rx_j1713) - RX J1713.7-3946
* [hess_j1702](hess_j1702) - HESS J1702-420
* [agn_j1224](agn_j1224) - TeV J1224+212

**CasA**
- Spectrum :
- Position :

**J1702**

Need to add extra sources on top of J1702

## Current status and thoughts (this section will be removed later) 

Current workflow is driven by 2 configs files. One to define targets (position, run selection, Emin/Emax, etc) and a general analysis template (ROI size, bins, etc). Run separately dataset generation, fitting, and plotting steps.

Two workflows: 3d analysis, 1d (TODO)

Current thoughts, issues on HLI (some known issues):
- non uniform unit handling in yaml
- `sky_circle` `radius` and `border` both needed for selection. Only `radius` should be needed in circle right ?
- `analysis.datasets.names` attribute missing
- in yaml file number like 9e-12 are not accepted (need 9.0e-12)
- For Gaussian model, no default value taken for e, phi (default to 0 not applied)

Final outputs: Ideally want to generate automatically for each source a summary in each source directory (in markdown for direct visualization in Github) including :

- Spectral plot with DC1 model, DC1 Flux points, latest gammapy results 
- Table with : True model, DC1 gammapy results, latest gammapy results (like below)



### Validation results

| Target        | SpatialModel              | SpectralModel  |
| ------------- |:-------------------------:| --------------:|
| CasA          | lon= 1.117e+02 1.156e-03  | Norm=1.454e-12 ± 6.096e-14                |
|               | lat= -2.129e+00 1.151e-03 | Index=2.790e+00 ± 1.935e-02               |
| hess_j1702    | lon=   | Norm=               |
|               | lat=  | Index=              |

## 1DC model
In the sky model for 1DC, the parameters for this source are given in model_galactic_bright.xml under the name Cassiopeia A.
It's an isolated point source with a power-law spectrum.

- (GLON, GLAT) = (111.734, -2.129) deg
- Differential flux at 1 TeV = 1.45e-12 cm-2 s-1 TeV-1
- Photon index = 2.75

ctools/gammapy results from DC1:
![DC1](cas_a/cas_a_dc1_closeout.png)

