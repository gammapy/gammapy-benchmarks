# Gammapy benchmarks and validation

This is the set of benchmarks and validation for Gammapy that we
continuously maintain and improve as we develop Gammapy.

It should always work with the latest Gammapy master. Checks are scheduled nightly, 
if a check fails a notification is sent to Gammapy #dev Slack channel. 
The badge below displays the status of last nightly set of checks.

- [benchmarks](benchmarks) - Gammapy performance checks (time, CPU, RAM)
- [validation](validation) - Gammapy correctness checks (model fitting, spectra, lightcurves, ...)

[![benchmarks and validation](https://github.com/gammapy/gammapy-benchmarks/actions/workflows/scheduled.yml/badge.svg)](https://github.com/gammapy/gammapy-benchmarks/actions/workflows/scheduled.yml)

### Note if you cloned this repository before Oct 7th 2022

The  primary branch was renamed `main`. See the following instructions:
https://github.com/gammapy/gammapy/wiki/Changing-master-branch-to-main
