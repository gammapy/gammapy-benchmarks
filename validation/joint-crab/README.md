# Gammapy validation: joint Crab

## Science use cases covered

- Perform a 1D data reduction 
- Perform a spectral fit and compute spectral parameters errors and confidence contours
- Perform a joint multi-instrument forward-folding likelihood fit

## Methodology

- Data from the joint-crab paper are used. Several runs from MAGIC, VERITAS, HESS and FACT are used, as well as some Fermi-LAT data.
- Data reduction is performed with the `Analysis` class. See the general [config file](config.yaml)
- All data are reduced into 1D spectral datasets using reflected region background estimation. A special handling is applied to Fermi-LAT data as techniques applied to IACTs are not relevant here.
- A log parabola fit is performed using the Minuit fitting backend. Errors and confidence contours are computed for the 3 parameters. The results are compared to the ones obtained for the joint-crab paper.

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




## References

- https://github.com/open-gamma-ray-astro/joint-crab
