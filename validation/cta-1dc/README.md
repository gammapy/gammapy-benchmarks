# Gammapy: CTA 1DC validation

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
