# Gammapy validation: Open Crab DL3 dataset from LST-1

Performs the 1D spectral analysis (with energy-dependent directional cuts) of [DL3 open data sample from LST-1 Crab Nebula observations](https://zenodo.org/records/11445184).

## Analysis

Implemented analysis: 1D spectral joint likelihood analysis [script](make.py)

Plotting: [script](plot.py)


## Results:
- Crab 1D spectral analysis:
  - fit results: 
    [reference](reference/spectral_model.yml), 
    [validation](results/best_fit_model.yml) 
  - plots:
    ![plots](plots/flux-points.png)

## Execution

To run the analysis (including the download of the DL3 files from Zenodo) use:

```
python make.py run-analysis
```
    
To produce the plots with spectral models and flux points comparison:
    
```
python plot.py
```
    
## Results discussion
The differences seen in the validation results with respect to the results from the performance study may be originated from:

  - The DL3 files used here for validation use an updated version of the IRFs with respect to the ones used in the performance study. It was check that the effect of this was on the normalization flux (within systematics), but did not affected the spectral shape.
  - The DL3 subsample used here is about 2 hours from observations taken in a couple of nights, while the one used in the LST-1 paper is about 35-hour observations spanning over 1.5 years.
  - LST-1 single-telescope data suffers very much from background systematics at low energy.


## References

- LST-1 performance study [H. Abe et al 2023 ApJ 956 80]([10.3847/1538-4357/ace89d](https://doi.org/10.3847/1538-4357/ace89d))
- DL3 files used here, [openly available in Zenodo](https://zenodo.org/records/11445184), are a subset of the sample used in the LST-1 performance study.
- Results in machine-readable format: https://github.com/cta-observatory/lst-crab-performance-paper-2023
