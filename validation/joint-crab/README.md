# Gammapy validation: joint Crab

- Data reduction is performed with `Analysis`. See the general [config file](config.yaml)
- Only log parabola fit is performed for now

## Results

### MAGIC 

- See [MAGIC summary page](results/magic_summary.md)

### VERITAS

- See [VERITAS summary page](results/veritas_summary.md)

### HESS

- See [HESS summary page](results/hess_summary.md)

### FACT

- See [FACT summary page](results/fact_summary.md)

### Fermi

- See [Fermi summary page](results/fermi_summary.md)

### Joint

- See [Joint summary page](results/joint_summary.md)


## Task

- Write a script (or notebook) to reproduce the Joint Crab analysis using the latest Gammapy
- Start with the normal analysis. The systematic error special likelihood would be nice to have also, but that can come later.
- Compare against the paper results
- This will especially test the high-level modeling & fitting code in Gammapy, which is under heavy development (covariance matrix, error ellipses)

## References

- https://github.com/open-gamma-ray-astro/joint-crab
