# Gammapy validation: HESS DL3 DR1 and validation paper

Validation of high-level analysis of HESS DL3 data.

## Science use cases covered

- Perform a 1D data reduction 
- Perform a 3D data reduction
- Perform a flux point extraction and upper limit calculation for a set of pointlike and extended sources.

## Methodology

- Data from the HESS-DL3-DR1 are used. 
- Data reduction is performed with the HLI `Analysis` class.
- Analysis pipeline is duplicated between 1D spectral and 3D cube data reduction.
- 1D analysis relies on reflected region background estimation. 3D analysis use field-of-view background estimation. Both rely on stacking the resulting DL4 products (contrary to the reference analysis).
- Spectral model fitting and flux points estimation are applied to both 1D and 3D datasets.
- Results are plotted and compared to the [reference](https://ui.adsabs.harvard.edu/abs/2019A%26A...632A..72M/abstract).

## Analyses

- Implemented analyses: 1d (joint analysis) 3d (stacked analysis) / [script](make.py)
- Plotting: [script](plot.py)

## Results:
- Crab:
    - 1d :
        - fit results: 
    [reference](crab/reference/reference-1d.yaml), 
    [validation](crab/results/result-1d.yaml) 
        - plots:
    ![plots](crab/plots/flux-points-1d.png)
    - 3d: 
        - fit results: 
    [reference](crab/reference/reference-3d.yaml), 
    [validation](crab/results/result-3d.yaml) 
        - plots: 
    ![plots](crab/plots/flux-points-3d.png)
- MSH 1552:
    - 1d / 
        - fit results:     
    [reference](msh1552/reference/reference-1d.yaml), 
    [validation](msh1552/results/result-1d.yaml)
        - plots:
    ![plots](msh1552/plots/flux-points-1d.png)
    - 3d / 
        - fit results:     
    [reference](msh1552/reference/reference-2.52.53d.yaml), 
    [validation](msh1552/results/result-3d.yaml)
        - plots:
    ![plots](msh1552/plots/flux-points-3d.png)un(self.model.parameters['norm'], steps, null_
- PKS 2155-304:
    - 1d /
        - fit results:  
    [reference](pks2155/reference/reference-1d.yaml), 
    [validation](pks2155/results/result-1d.yaml) 
        - plots:
    ![plots](pks2155/plots/flux-points-1d.png)
    - 3d / 
        - fit results:     
    [reference](pks2155/reference/reference-3d.yaml), 
    [validation](pks2155/results/result-3d.yaml)
        - plots:
    ![plots](pks2155/plots/flux-points-3d.png)
- RXJ1739-3946:
    - 1d / Not implemented
    - 3d / Not implemented
    


## Execution

To run a all analyses (it takes ~10 min on this machines) use:

    python make.py run-analyses all-targets all-methods 
    
To run the analysis (both 1d and 3d) for a specific source use:

    python make.py run-analyses crab all-methods
    
To run only e.g. 1d analysis, for all targets use:

    python make.py run-analyses all-targets 1d

To run in debug mode (quickly, ~12 s on this machine):

    python make.py run-analyses all-targets all-methods --debug
    
To produce the plots (for now, only spectral points are plotted):
    
    python plot.py
    
## Results discussion
First of all notice that, for the sake of time (and code lines) saving, the validation that is implemented in `make.py`
 is a rather simplified version of the analysis that is performed in the reference paper. The main simplification 
 consists in the fact that we run a 3d "stacked" analysis, as opposed to a (much longer) "joint" one.

The fit results are generally in acceptable (sometimes even good) agreement with the reference values. 


 
## References

- Dataset (including background models from HESS validation paper): https://github.com/gammapy/gammapy-extra/tree/master/datasets/hess-dl3-dr1
- Validation of open source tools paper: [Mohrmann et al (2019)](https://ui.adsabs.harvard.edu/abs/2019A%26A...632A..72M/abstract)
  - Paper results in [machine-readable format](https://github.com/lmohrmann/hess_ost_paper_material)
