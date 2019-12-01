# Gammapy validation: HESS DL3 DR1 and validation paper

Validation of high-level analysis of HESS DL3 data.

## Analyses

- Implemented analyses: 1d (joint analysis) 3d (stacked analysis) / [script](make.py)
- Plotting: [script](plot.py)

## Results:
- Crab:
    - 1d / 
    [reference](crab/reference/gammapy_crab_1d_powerlaw.dat) / 
    [results](crab/results/results-summary-fit-1d.yaml) / 
    ![plots](crab/plots/flux-points-1d.png)
    - 3d / 
    [reference](crab/reference/gammapy_crab_3d_powerlaw.dat) / 
    [results](crab/results/results-summary-fit-3d.yaml) / 
    ![plots](crab/plots/flux-points-3d.png)
- PKS 2155-304:
    - 1d / 
    [reference](pks2155/reference/gammapy_pks2155_1d_powerlaw.dat) / 
    [results](pks2155/results/results-summary-fit-1d.yaml) / 
    ![plots](pks2155/plots/flux-points-1d.png)
    - 3d / 
    [reference](pks2155/reference/gammapy_pks2155_3d_powerlaw.dat) / 
    [results](pks2155/results/results-summary-fit-3d.yaml) / 
    ![plots](pks2155/plots/flux-points-3d.png)
- MSH 1552:
    - 1d / 
    [reference](msh1552/reference/gammapy_msh1552_1d_powerlaw.dat) / 
    [results](msh1552/results/results-summary-fit-1d.yaml) / 
    ![plots](msh1552/plots/flux-points-1d.png)
    - 3d / 
    [reference](msh1552/reference/gammapy_msh1552_3d_powerlaw.dat) / 
    [results](msh1552/results/results-summary-fit-3d.yaml) / 
    ![plots](msh1552/plots/flux-points-3d.png)
- RXJ1739-3946:
    - 1d / Not implemented
    - 3d / Not implemented
    


## Execution

To run a all analyses use

    python make.py run-analyses all 
    
To run the analysis (both 1d and 3d) for a specific source:

    python make.py run-analyses 'crab'

To run in debug mode (quickly):

    python make.py run-analyses all --debug
    
To produce the plots (for now, only spectral points are plotted):
    
    python plot.py

## References

- Dataset (including background models from HESS validation paper): https://github.com/gammapy/gammapy-extra/tree/master/datasets/hess-dl3-dr1
- Lars Mohrmann paper: https://ui.adsabs.harvard.edu/abs/2019arXiv191008088M
  - Results in machine-readable format: https://github.com/lmohrmann/hess_ost_paper_material
