# gammapy-benchmarks

Performance benchmarks for Gammapy

## Analyses

We have the following typical science cases tested:

-[maps_3d](maps_3d) - Data preparation for the 3D case
-[analysis_3d](analysis_3d) - Data fitting in 3D
-[spectrum_1d](spectrum_1d) - 1D spectral extraction and fitting
-[lightcurve_1d](lightcurve_1d) - Light-curve extraction in 1D
-[lightcurve_3d](lightcurve_3d) - Light-curve extraction in 3D


## Execution
To run a specific benchmark use:
```bash
./make.py run-benchmark benchmark-name
```

To run all benchmarks:
```bash
./make.py run-benchmark all
```


