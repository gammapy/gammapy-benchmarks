import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from gammapy.maps import Map
from gammapy.utils.fitting import Fit
from gammapy.data import ObservationStats, ObservationSummary, DataStore
from gammapy.background import ReflectedRegionsBackgroundEstimator
from gammapy.spectrum.models import PowerLaw
from gammapy.spectrum import (
    SpectrumExtraction,
    SpectrumDatasetOnOffStacker,
    FluxPointsEstimator,
    FluxPointsDataset,
)

N_OBS = 10
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

    background_estimator = ReflectedRegionsBackgroundEstimator(
        observations=observations, on_region=on_region, exclusion_mask=exclusion_mask
    )

    background_estimator.run()

    stats = []
    for obs, bkg in zip(observations, background_estimator.result):
        stats.append(ObservationStats.from_observation(obs, bkg))

    obs_summary = ObservationSummary(stats)

    e_reco = np.logspace(-1, np.log10(40), 40) * u.TeV
    e_true = np.logspace(np.log10(0.05), 2, 200) * u.TeV

    extraction = SpectrumExtraction(
        observations=observations,
        bkg_estimate=background_estimator.result,
        containment_correction=False,
        e_reco=e_reco,
        e_true=e_true,
    )

    extraction.run()

    extraction.compute_energy_threshold(method_lo="area_max", area_percent_lo=10.0)

    model = PowerLaw(
        index=2, amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
    )

    datasets_joint = extraction.spectrum_observations

    for dataset in datasets_joint:
        dataset.model = model

    fit_joint = Fit(datasets_joint)
    result_joint = fit_joint.run()

    # we make a copy here to compare it later
    model_best_joint = model.copy()
    model_best_joint.parameters.covariance = result_joint.parameters.covariance

    e_min, e_max = 1, 30
    e_edges = np.logspace(np.log10(e_min), np.log10(e_max), 11) * u.TeV

    fpe = FluxPointsEstimator(datasets=datasets_joint, e_edges=e_edges)
    flux_points = fpe.run()

    stacker = SpectrumDatasetOnOffStacker(datasets_joint)
    dataset_stacked = stacker.run()

    dataset_stacked.model = model
    stacked_fit = Fit([dataset_stacked])
    stacked_fit.run()


if __name__ == "__main__":
    run_benchmark()
