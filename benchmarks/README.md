# gammapy-benchmarks

Performance benchmarks for Gammapy

## Analyses

We have the following typical science cases tested:

- [maps_3d.py](maps_3d.py) - Stacked data preparation for the 3D case
- [analysis_3d.py](analysis_3d.py) - Stacked data fitting in 3D
- [spectrum_1d.py](spectrum_1d.py) - 1D spectral extraction and fitting
- [lightcurve_1d.py](lightcurve_1d.py) - Light-curve extraction in 1D
- [lightcurve_3d.py](lightcurve_3d.py) - Light-curve extraction in 3D (tbd)
- [joint_maps_3d.py](joint_maps_3d.py) - Data preparation for the 3D case for joint analysis (tbd)
- [joint_analysis_3d.py](joint_analysis_3d.py) - Joint Data fit in 3D (tbd)

## Execution

To run a specific benchmark use:
```bash
./make.py run-benchmark benchmark-name
```

To run all benchmarks:
```bash
./make.py run-benchmark all
```


