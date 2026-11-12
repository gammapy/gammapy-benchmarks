# Gammapy validation: dark matter in dSphs (CTAO prospects paper)

Reproduces with Gammapy the dark matter continuum spectra and the median expected
annihilation limits of
[CTAO Consortium (2025), MNRAS 544, 2946](https://doi.org/10.1093/mnras/staf1798)
([arXiv:2508.19120](https://arxiv.org/abs/2508.19120)), using the astrophysical
factors, 2D maps and reference limits of its
[supplementary material](https://zenodo.org/records/17079907).

The paper itself was produced with Gammapy v1.2: this checks that the current
Gammapy reproduces a published result, not a comparison with an independent code.
Requires Gammapy >= 2.2 (`DarkMatterSpectralModel`, `ContinuumPrimaryFlux`).

## Analysis

- [make.py](make.py): runs the two scripts below and prints a summary of the checks.
- [make_spectra.py](make_spectra.py): PPPC4 and CosmiXs tables vs
  `ContinuumPrimaryFlux` (channels b, tau, mu, W). For every mass and x = E / m_DM
  in the tables, the Gammapy dN/dE must equal dN/dlog10(x) / (E ln 10), with the
  tables read independently of Gammapy.
- [make_limits.py](make_limits.py): median expected upper limit on <sigma v>,
  compared with the median of the paper. Check: ratio within [1/1.15, 1.15].

Limits: simulated 100 h observations of the classical dSphs Draco I and Sculptor
(prod5-v0.1 IRFs), Einasto profile, PPPC4 spectra, channels b (soft) and tau
(hard), masses 1, 10 and 50 TeV, without the J factor as nuisance (`wosj` in the
paper), as in the paper. The median is computed on an Asimov dataset
(`--n-realizations N` gives the median over N Poisson realizations instead).

## Execution

```
python make.py                       # spectra + annihilation limits, with summary
python make_spectra.py
python make_limits.py --mode annihilation [--targets DraI] [--n-realizations 100]
```

## Results

- [results/spectra.ecsv](results/spectra.ecsv): Gammapy reproduces both tables to
  numerical precision (relative differences < 1e-13).
- [results/limits_annihilation.ecsv](results/limits_annihilation.ecsv): TODO

Decay is not validated: the paper used Gammapy v1.2, where the decay spectrum was
evaluated at m_DM instead of m_DM / 2. This was corrected in later versions, so
current Gammapy gives lifetime limits weaker than the paper by a factor ~0.3-0.6.