import logging

log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np
import yaml
from pathlib import Path
from regions import CircleSkyRegion
from astropy import units as u
from astropy.coordinates import SkyCoord
from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.spectrum import FluxPointsEstimator

PATH_CONFIG = Path("./config/")
PATH_RESULTS = Path("./results/")
OBS_IDS = ["23523", "23526", "23559", "23592"]
EMIN, EMAX = [0.7, 60] * u.TeV  # Used for flux points estimation
N_FLUX_POINTS = 15

filename = str(PATH_CONFIG / "config_3d.yaml")
config = yaml.load(open(filename))
config = AnalysisConfig(config)

#  Observation selection
analysis = Analysis(config)
analysis.get_observations()
assert analysis.observations.ids == OBS_IDS

# Data reduction
analysis.get_datasets()
assert len(analysis.datasets) == len(OBS_IDS)

# Set runwise energy threshold. See reference paper, section 5.1.1.
for dataset in analysis.datasets:
    # E_thr_bias (= 10% edisp)
    geom = dataset.counts.geom
    e_reco = geom.get_axis_by_name("energy").edges
    E_thr_bias = dataset.edisp.get_bias_energy(0.1)

    # E_thr_bkg (= background peak energy)
    background_model = dataset.background_model
    bkg_spectrum = background_model.map.get_spectrum()
    peak = bkg_spectrum.data.max()
    idx = list(bkg_spectrum.data).index(peak)
    E_thr_bkg = bkg_spectrum.energy.center[idx]

    esafe = max(E_thr_bias, E_thr_bkg)
    dataset.mask_fit = geom.energy_mask(emin=esafe)

# Model fitting
model_config = str(PATH_CONFIG / "config_model.yaml")
analysis.set_model(filename=model_config)

for dataset in analysis.datasets:
    dataset.background_model.norm.frozen = False
    dataset.background_model.tilt.frozen = False

result = analysis.run_fit()
assert analysis.fit_result.success == True

# Store fit results
fit_results_dict = {}
parameters = analysis.model.parameters
parameters.covariance = analysis.fit_result.parameters.covariance[0:5, 0:5]
for parameter in parameters:
    value = parameter.value
    error = parameters.error(parameter)
    unit = parameter.unit
    name = parameter.name
    string = "{0:.2e} +- {1:.2e} {2}".format(value, error, unit)
    fit_results_dict.update({name: string})
with open(str(PATH_RESULTS / "results-summary-fit-3d.yaml"), "w") as f:
    yaml.dump(fit_results_dict, f)

# Flux points
# TODO: This is a workaround to reoptimize the bkg in each
# energy bin. We need to add this option to the Analysis class
datasets = analysis.datasets.copy()
for dataset in datasets:
    for par in dataset.parameters:
        if par is not dataset.background_model.norm:
            par.frozen = True

e_edges = (
    np.logspace(np.log10(EMIN.value), np.log10(EMAX.value), N_FLUX_POINTS + 1) * u.TeV
)
fpe = FluxPointsEstimator(
    datasets=datasets, e_edges=e_edges, source="crab", reoptimize=True
)

flux_points = fpe.run()
flux_points.write(str(PATH_RESULTS / "flux-points-3d.html"))
