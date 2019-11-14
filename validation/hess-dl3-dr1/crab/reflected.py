import logging

log = logging.getLogger(__name__)

import numpy as np
import astropy.units as u
import yaml
from pathlib import Path
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from gammapy.maps import Map
from gammapy.modeling import Fit, Datasets
from gammapy.data import DataStore
from gammapy.modeling.models import PowerLawSpectralModel, create_crab_spectral_model
from gammapy.cube import SafeMaskMaker
from gammapy.spectrum import (
    SpectrumDatasetMaker,
    SpectrumDatasetOnOff,
    FluxPointsEstimator,
    FluxPointsDataset,
    ReflectedRegionsBackgroundMaker,
)

PATH_RESULTS = Path("./results/")
OBS_IDS = ["23523", "23526", "23559", "23592"]
TARGET_POS = SkyCoord(83.63, 22.01, unit="deg", frame="icrs")
ON_RADIUS = Angle("0.11 deg")
E_RECO = np.logspace(-1, np.log10(70), 20) * u.TeV
E_TRUE = np.logspace(np.log10(0.05), 2, 200) * u.TeV
CONTAINMENT_CORR = True
EMIN, EMAX = [0.7, 60] * u.TeV  # Used for flux points estimation
N_FLUX_POINTS = 15

# Observations selection
datastore = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
observations = datastore.get_observations(OBS_IDS)

# Exclusion mask
exclusion_region = CircleSkyRegion(
    center=SkyCoord(183.604, -8.708, unit="deg", frame="galactic"), radius=0.5 * u.deg
)
exclusion_mask = Map.create(
    npix=(150, 150), binsz=0.05, skydir=TARGET_POS, proj="TAN", coordsys="CEL"
)
mask = exclusion_mask.geom.region_mask([exclusion_region], inside=False)
exclusion_mask.data = mask

# Reflected regions background estimation
on_region = CircleSkyRegion(center=TARGET_POS, radius=ON_RADIUS)
dataset_maker = SpectrumDatasetMaker(
    region=on_region,
    e_reco=E_RECO,
    e_true=E_TRUE,
    containment_correction=CONTAINMENT_CORR,
)
bkg_maker = ReflectedRegionsBackgroundMaker(exclusion_mask=exclusion_mask)
safe_mask_masker = SafeMaskMaker(methods=["edisp-bias"], bias_percent=10)

datasets = []

for observation in observations:
    dataset = dataset_maker.run(observation, selection=["counts", "aeff", "edisp"])
    dataset_on_off = bkg_maker.run(dataset, observation)
    dataset_on_off = safe_mask_masker.run(dataset_on_off, observation)
    datasets.append(dataset_on_off)

# Fit spectrum
model = PowerLawSpectralModel(
    index=2, amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1.45 * u.TeV
)

for dataset in datasets:
    dataset.model = model

fit_joint = Fit(datasets)
result_joint = fit_joint.run()

# Store fit results
fit_results_dict = {}
parameters = model.parameters
parameters.covariance = result_joint.parameters.covariance[0:5, 0:5]
for parameter in parameters:
    value = parameter.value
    error = parameters.error(parameter)
    unit = parameter.unit
    name = parameter.name
    string = "{0:.2e} +- {1:.2e} {2}".format(value, error, unit)
    fit_results_dict.update({name: string})
with open(str(PATH_RESULTS / "results-summary-fit-1d.yaml"), "w") as f:
    yaml.dump(fit_results_dict, f)

# Flux points
e_edges = (
    np.logspace(np.log10(EMIN.value), np.log10(EMAX.value), N_FLUX_POINTS + 1) * u.TeV
)
fpe = FluxPointsEstimator(datasets=datasets, e_edges=e_edges)
flux_points = fpe.run()
flux_points.write(str(PATH_RESULTS / "flux-points-1d.html"))
