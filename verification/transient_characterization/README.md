# Transient characterization

Verifies Gammapy's full detect → localize → time-resolve → fit pipeline for
a bright, short-lived transient, using real event sampling rather than
Poisson-faked datasets.

## Scenario

A 30-minute observation contains a point source whose flux follows a
`GeneralizedGaussianTemporalModel` (eta=1, i.e. a two-sided exponential):
negligible for most of the observation, then peaking sharply at `t_ref`
(10 minutes in) and decaying with `t_decay = 200 s` afterward. `t_rise` is
fixed to a short value (10 s) purely to give the pulse a sharp, well-posed
onset -- it is not a quantity under test. Events (source + background) are
drawn with `MapDatasetEventSampler`, IRF-folded, exactly as they would be
for a real CTAO observation.

## Workflow

1. **Sample events** for the full 30-minute observation in a single pass
   (`tc_utils.sample_transient_observation`).
2. **Build a 3D dataset** from the sampled events and run source detection
   (`ExcessMapEstimator` + peak finding) to estimate the source position.
3. **Extract 1D on-region spectra** in 30 s time bins across the full
   observation, centered on the estimated position.
4. **Jointly fit** a single spectral (power law) + temporal (generalized
   Gaussian, eta and t_rise frozen) model across all time-binned datasets
   simultaneously.
5. Repeat 1-4 over many Monte Carlo realizations and check that the fitted
   parameters' **pull distributions** (`(fitted - true) / fitted_error` for
   position, spectral index/amplitude, peak time, and decay time) are
   compatible with a standard normal -- i.e. the pipeline is both unbiased
   and correctly estimates its own uncertainties.

## Usage

    python make.py tc --n_samples 30

Writes `results/transient_characterization_{...}.json` (pull statistics,
detection efficiency, fit convergence rate) and a pull-distribution PNG.
Run `pytest tests/` to check the results against the pull tolerance bands
(skips automatically if no results have been generated yet).
