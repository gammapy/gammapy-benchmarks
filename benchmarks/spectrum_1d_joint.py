import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from gammapy.maps import Map
from gammapy.modeling import Fit, Datasets
from gammapy.data import DataStore
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.spectrum import (
    SpectrumDatasetMaker,
    SpectrumDatasetOnOff,
    SafeMaskMaker,
    FluxPointsEstimator,
    ReflectedRegionsBackgroundMaker,
)

N_OBS = 100
OBS_ID = 23523


def run_benchmark():
    # Set up data store and select N_OBS times the observation OBS_ID
    data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
    obs_ids = OBS_ID * np.ones(N_OBS)

    observations = data_store.get_observations(obs_ids)

    target_position = SkyCoord(ra=83.63, dec=22.01, unit="deg", frame="icrs")
    on_region_radius = Angle("0.11 deg")
    on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)

    exclusion_region = CircleSkyRegion(
        center=SkyCoord(183.604, -8.708, unit="deg", frame="galactic"),
        radius=0.5 * u.deg,
    )

    skydir = target_position.galactic
    exclusion_mask = Map.create(
        npix=(150, 150), binsz=0.05, skydir=skydir, proj="TAN", coordsys="GAL"
    )

    mask = exclusion_mask.geom.region_mask([exclusion_region], inside=False)
    exclusion_mask.data = mask

    e_reco = np.logspace(-1, np.log10(40), 40) * u.TeV
    e_true = np.logspace(np.log10(0.05), 2, 200) * u.TeV

    dataset_maker = SpectrumDatasetMaker(
        region=on_region, e_reco=e_reco, e_true=e_true, containment_correction=True
    )
    bkg_maker = ReflectedRegionsBackgroundMaker(exclusion_mask=exclusion_mask)
    safe_mask_masker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)

    # Data preparation
    datasets = []

    for observation in observations:
        dataset = dataset_maker.run(observation, selection=["counts", "aeff", "edisp"])
        dataset_on_off = bkg_maker.run(dataset, observation)
        dataset_on_off = safe_mask_masker.run(dataset_on_off, observation)
        datasets.append(dataset_on_off)

    # Modeling and fitting

    model = PowerLawSpectralModel(
        index=2, amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
    )

    for dataset in datasets:
        dataset.model = model

    fit_joint = Fit(datasets)
    result_joint = fit_joint.run()

    model_best_joint = model.copy()
    model_best_joint.parameters.covariance = result_joint.parameters.covariance

    # Flux points
    e_min, e_max = 0.7, 30
    e_edges = np.logspace(np.log10(e_min), np.log10(e_max), 11) * u.TeV
    fpe = FluxPointsEstimator(datasets=datasets, e_edges=e_edges)
    fpe.run()


if __name__ == "__main__":
    run_benchmark()
