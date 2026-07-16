# Gammapy scientific verification

High-level science verification for Gammapy.

This module covers science use case coverage while checking for result robustness, in particular w.r.t. statistical correctness. The general approach relies on parametric bootstrap (MC) for a wide-range of science use cases and checks of the results against input. 

## Science Use Cases

The following science use cases are currently covered:

- [coverage-fp](coverage-fp) - 1D and 3D spectral flux points (SED) calculations
  - coverage of measurement errors and upper limits is checked
- [sensitivity](sensitivity) - 1D and 3D sensitivity calculation is validated agaisnt MC simulations

