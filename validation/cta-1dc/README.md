# Gammapy: CTA 1DC validation

## Overview

As described [here](https://forge.in2p3.fr/projects/data-challenge-1-dc-1/wiki/Current_capabilities_and_limitations_of_the_analysis_tools),
the following test cases were chosen for the validation:

* [cas_a](cas_a) - Cassiopeia A
* [hess_j1702](hess_j1702) - HESS J1702-420
* [agn_j1224](agn_j1224) - TeV J1224+212
* [gc](gc) - Galactic center
* [rx_j1713](rx_j1713) - RX J1713.7-3946

**CasA**
- Spectrum : norm spot on, index within 1 sigma
- Position : ok

**J1702**

Param not yet relevant, needs to add extra sources on top of J1702 to have a good model

## Execution

To run a all analyses use

    python make.py run-analyses all 
    
To run the analysis for a specific source:

    python make.py run-analyses 'cas_a'


## 1DC model


ctools/gammapy results from DC1:
![DC1](cas_a/cas_a_dc1_closeout.png)

## References

- [CTA 1DC wiki page](https://forge.in2p3.fr/projects/data-challenge-1-dc-1/wiki)
- [CTA 1DC tools checks page](https://forge.in2p3.fr/projects/data-challenge-1-dc-1/wiki/Current_capabilities_and_limitations_of_the_analysis_tools)
- https://github.com/gammasky/cta-analyses/tree/master/dc-1-checks (private repo, ask on Slack if you want access)
- [CTA 1DC close-out document](https://forge.in2p3.fr/attachments/download/63626/CTA_DC1_CloseOut.pdf)
