# Gammapy validation

High-level science analysis validation for Gammapy.

Note that we have performance benchmarks separately, see [here](../README.md).

## Analyses

We would like to have the following analyses scripted as soon as possible:

- [cta-1dc](cta-1dc) - CTA first data challenge tools check
- [hess-dl3-dr1](hess-dl3-dr1) - H.E.S.S. DR1 & validation paper
- [joint-crab](joint-crab) - Joint Crab paper
- [fermi-3fhl](fermi-3fhl) - Fermi 3FHL catalog paper

Other ideas and contributions are welcome. If you want to contribute, please get in touch. Examples:

- Reproduce sensitivity curves or CTA simulation paper results with Gammapy using public IRFs
- Simulate and fit spectral line at low energy where energy dispersion matters
- Simulate and fit various models (e.g. each spectral and spatial model) and check that input=output.
  Possibly use ctools or Fermi ST to simulate, to get a validation against a different tool.
- Use 4FGL and test energy dispersion against latest Fermi ST
- Try to reproduce full 3FHL catalog (should be doable with a 1000 line script) or HGPS catalog (needs private data, more work)
- ...

## Howto

- We are looking for 1-2 people for each folder listed above that do it in Nov 2019. In each case there's ~ 1000 lines of code to write and it can be done in a few days. We want to go fast and not spread the task across too many people. The AGN lightcurve analyses are special cases and there it makes sense to do it separately.
- In each folder, write Python scripts that run the analysis and write the results (fit parameters, spectral points, lightcurve, ...) to machine-redable files (YAML, ECSV, FITS). Use the high-level Gammapy analysis interface, the goal is to write as little code as possible to test as much as possible.
- Write a result summary with limited float precision (e.g. spectral index with two digits, like `gammapy = 2.73 +- 0.13`) and store that in human-readable text files (YAML, ECSV, MD) in the repo.
- Manually compare the results against the paper, and write a `README.md` summarising whether results agree or not. For cases where there are differences, figure out why, and either document the reason for the discrepancy, or if it's suspected that it's a bug in Gammapy, file an issue there. After a bug in Gammapy is fixed, `git diff` should reveal the changes.
- Setting up a continuous integration server to run this "science validation test suite" for Gammapy nightly or weekly, and to change the `git diff` method against automated & scripted tests (e.g. using pytest, or writing a results HTML report) is something for 2020 or later.
