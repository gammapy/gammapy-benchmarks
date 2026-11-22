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
   The position is taken directly from the peak and frozen for the rest of
   the pipeline -- it is never fitted, since position recovery is not a
   quantity under test here.
3. **Extract 1D on-region spectra** in 30 s time bins across the full
   observation, centered on the detected (frozen) position.
4. **Jointly fit** a spectral (power law, index free) + temporal (generalized
   Gaussian; `eta` and `t_rise` frozen, `t_ref` frozen at a light-curve-based
   estimate -- see below) model across all time-binned datasets
   simultaneously. Position stays frozen here too.
5. Repeat 1-4 over many Monte Carlo realizations and check that the fitted
   parameters' **pull distributions** (`(fitted - true) / fitted_error`) are
   compatible with a standard normal -- i.e. the pipeline is both unbiased
   and correctly estimates its own uncertainties.

## Known issue: amplitude/index/t_decay degeneracy (unresolved)

The joint fit's free parameters (`amplitude`, `index`, `t_decay` -- `t_ref`
is frozen, see step 4) show a structural degeneracy: across three rounds of
testing, freezing any one of a related pair to fix its pull broke another
parameter instead (index/amplitude -> froze index -> amplitude/t_ref -> froze
t_ref -> index/t_decay). This is not resolved. `tests/test_pull_distribution.py`
only asserts on `amplitude` for now; `t_decay` pulls are still computed and
stored in the results JSON but not tested. `index` stays free in the joint
fit (seeded from the detection fit) but its pull is not tracked at all --
same degeneracy. (`lon_0`/`lat_0` are not part of `PULL_PARAMETERS` either --
position is fixed at the detection peak throughout the pipeline, never
fitted.) See
`verification/CLAUDE.md` (transient_characterization, step 3) for the full
history before changing this further -- freezing another parameter without a
real investigation (e.g. the joint fit's full correlation matrix) is
expected to just move the problem again, not fix it.

## Usage

    python make.py tc --n_samples 30

Writes `results/transient_characterization_{...}.json` (pull statistics,
detection efficiency, fit convergence rate) and a pull-distribution PNG.
Run `pytest tests/` to check the results against the pull tolerance bands
(skips automatically if no results have been generated yet).
