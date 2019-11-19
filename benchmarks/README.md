# gammapy-benchmarks

Performance benchmarks for Gammapy.

Note that we have validation separately, see [here](../README.md).

## Analyses

We have the following typical science cases tested:

- analysis_3d / [script](analysis_3d.py) / [results](results/analysis_3d) / 3d stacked analysis
- analysis_3d_joint / [script](analysis_3d_joint.py) / [results](results/analysis_3d_joint) - 3d joint analysis
- spectrum_1d / [script](spectrum_1d.py) / [results](results/spectrum_1d) - 1d stacked analysis
- spectrum_1d_joint / [script](spectrum_1d_joint.py) / [results](results/spectrum_1d_joint) - 1d joint analysis
- lightcurve_1d / [script](lightcurve_1d.py) / [results](results/lightcurve_1d) - 1d lightcurve analysis
- lightcurve_3d / [script](lightcurve_3d.py) / [results](results/lightcurve_3d) - 3d lightcurve analysis
- io / [script](io.py) / [results](results/io) - Read DL3 events data

## Execution

To run a specific benchmark use [make.py](make.py).

    python make.py run-benchmark benchmark-name

To run all benchmarks:

    python make.py run-benchmark all

To debug scripts (run quickly using only 2 observations):

    GAMMAPY_BENCH_N_OBS=2 python make.py run-benchmark all

## Results

A summary of the results (for 100 runs) can be found [here](results/results.yaml).
