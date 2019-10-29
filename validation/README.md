# Gammapy validation

High-level science analysis validation for Gammapy.

## Analyses

- [fermi-3fhl](fermi-3fhl) - Fermi 3FHL catalog paper
- [hess-dl3-dr1](hess-dl3-dr1) - H.E.S.S. DR1 & validation paper
- [cta-1dc](cta-1dc) - CTA first data challenge tools check
- [joint-crab](joint-crab) - Joint Crab paper

## Howto

- We are looking for 1-2 people for each folder listed above that do it in Nov 2019. In each case there's ~ 1000 lines of code to write and it can be done in a few days. We want to go fast and not spread the task across too many people.
- In each folder, write Python scripts that run the analysis and write the results (fit parameters, spectral points, lightcurve, ...) to machine-redable files (YAML, ECSV, FITS).
- Write a result summary with limited float precision (e.g. spectral index with two digits, like `gammapy = 2.73 +- 0.13`) and store that in human-readable text files (YAML, ECSV, MD) in the repo.
- Manually compare the results against the paper, and write a `README.md` summarising whether results agree or not. For cases where there are differences, figure out why, and either document the reason for the discrepancy, or if it's suspected that it's a bug in Gammapy, file an issue there. After a bug in Gammapy is fixed, `git diff` should reveal the changes.
- Setting up a continuous integration server to run this "science validation test suite" for Gammapy nightly or weekly, and to change the `git diff` method against automated & scripted tests (e.g. using pytest, or writing a results HTML report) is something for 2020 or later.
