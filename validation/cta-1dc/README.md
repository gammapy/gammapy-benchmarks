# Gammapy: CTA 1DC validation

## Overview

As described [here](https://forge.in2p3.fr/projects/data-challenge-1-dc-1/wiki/Current_capabilities_and_limitations_of_the_analysis_tools),
the following test cases were chosen for the validation:

* [cas_a](cas_a) - Cassiopeia A
* [gc](gc) - Galactic center
* [rx_j1713](rx_j1713) - RX J1713.7-3946
* [hess_j1702](hess_j1702) - HESS J1702-420
* [agn_j1224](agn_j1224) - TeV J1224+212

## Validation results

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

## Task

- Script some or all of the analyses from the CTA 1DC tools validation with latest Gammapy.
- Compare to results on the wiki page or in the close-out document
- Note that the IRFs used to simulate and analyse CTA 1DC had some issues. Especially the energy dispersion was noisy, leading to unstable spectral results in several cases at low energies (see e.g. links [here](https://github.com/gammapy/gammapy/issues/2484#issuecomment-545904310)). The way we handle this is to just document discrepancies for now, and then in 2020 or later we'll change CTA validation dataset for Gammapy when something newer becomes available.

## References

- [CTA 1DC wiki page](https://forge.in2p3.fr/projects/data-challenge-1-dc-1/wiki)
- [CTA 1DC tools checks page](https://forge.in2p3.fr/projects/data-challenge-1-dc-1/wiki/Current_capabilities_and_limitations_of_the_analysis_tools)
- https://github.com/gammasky/cta-analyses/tree/master/dc-1-checks (private repo, ask on Slack if you want access)
- [CTA 1DC close-out document](https://forge.in2p3.fr/attachments/download/63626/CTA_DC1_CloseOut.pdf)
- Tutorial: https://docs.gammapy.org/0.14/notebooks/cta_1dc_introduction.html
- Tutorial: https://docs.gammapy.org/0.14/notebooks/cta_data_analysis.html
