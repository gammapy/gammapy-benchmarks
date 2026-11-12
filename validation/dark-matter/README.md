# Gammapy validation: dark matter in dSphs (CTAO prospects paper)

Reproduces with Gammapy the dark matter spectra and the median expected limits of
[CTAO Consortium (2025), MNRAS 544, 2946](https://doi.org/10.1093/mnras/staf1798)
([arXiv:2508.19120](https://arxiv.org/abs/2508.19120)), using the astrophysical
factors, 2D maps and reference limits of its
[supplementary material](https://zenodo.org/records/17079907).

The paper itself was produced with Gammapy v1.2: this checks that the current
Gammapy reproduces a published result, not a comparison with an independent code.

## Analysis

Three scripts, each writing one table to `results/`:
- [make.py](make.py):  Main script that calls all the subsequent ones.
- [make_spectra.py](make_spectra.py): PPPC4 and CosmiXs tables vs `PrimaryFlux`
  (b, tau, mu, W). Check: at the mass nodes Gammapy reproduces the table, read
  independently of Gammapy, to better than 1e-6. Between nodes, the difference
  with a cubic spline in log10(mass) is reported as an estimate of the
  interpolation uncertainty (no check). Uses `$GAMMAPY_DATA` only.
- [make_annihilation.py](make_annihilation.py): median upper limit on <sigma v>.
- [make_decay.py](make_decay.py): median lower limit on the lifetime.

Limits: continuum, DraI and Scl (classical), CBe and RetII (ultra-faint), b (soft)
and tau (hard) channels, 1, 10 and 50 TeV, Einasto, 100 h, without the
astrophysical factor as nuisance (`wosj` in the paper). The median is computed on
an Asimov dataset, or over `--n-realizations` Poisson realizations. Check:
Gammapy median / paper median within [0.8, 1.25].

## Data
The input data are in [data](data). They are a subset of the supplementary material 
of the CTAO dSph paper (https://zenodo.org/records/17079907) and of the prod5-v0.1 IRFs,
 kept here so that the validation does not depend on downloads. 

## Execution

```
python make.py
python make_spectra.py
python make_annihilation.py
python make_decay.py
```

With the Asimov dataset each limits script takes a few minutes. With
`--n-realizations 100` it takes hours; it can be split with `--targets`
(e.g. one job per target), which writes one table per subset.

## Results

### Spectra

[results/spectra.ecsv](results/spectra.ecsv). At the nodes Gammapy reproduces
the tables to numerical precision (< 1e-13). Between nodes, for masses above
100 GeV, the median difference with a cubic interpolation is below 2% for b, tau
and W and a few % for mu, with maxima of 20-50% close to the kinematic endpoint.

### Limits

TODO: results/limits_annihilation.ecsv, results/limits_decay.ecsv

## Results discussion

TODO




