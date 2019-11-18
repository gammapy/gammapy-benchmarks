import numpy as np
import astropy.units as u
import time
import yaml
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from gammapy.maps import Map, MapAxis
from gammapy.modeling import Fit
from gammapy.data import DataStore
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.spectrum import (
    SpectrumDatasetMaker,
    SpectrumDatasetOnOff,
    FluxPointsEstimator,
    ReflectedRegionsBackgroundMaker,
)
from gammapy.cube import SafeMaskMaker

import os

N_OBS = 10
OBS_ID = 23523


def data_prep():
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

    e_reco = MapAxis.from_bounds(0.1, 40, nbin=40, interp="log", unit="TeV").edges
    e_true = MapAxis.from_bounds(0.05, 100, nbin=200, interp="log", unit="TeV").edges

    dataset_maker = SpectrumDatasetMaker(
        region=on_region, e_reco=e_reco, e_true=e_true, containment_correction=True
    )
    bkg_maker = ReflectedRegionsBackgroundMaker(exclusion_mask=exclusion_mask)
    safe_mask_masker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)

    # Data preparation
    datasets = []

    for ind, observation in enumerate(observations):
        dataset = dataset_maker.run(observation, selection=["counts", "aeff", "edisp"])
        dataset_on_off = bkg_maker.run(dataset, observation)
        dataset_on_off = safe_mask_masker.run(dataset_on_off, observation)
        dataset_on_off.name = str(ind)
        datasets.append(dataset_on_off)
    return datasets


def write(datasets):
    for ind, dataset in enumerate(datasets):
        dataset.to_ogip_files(overwrite=True)


def read():
    datasets = []
    model = PowerLawSpectralModel(
        index=2, amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
    )

    for ind in range(N_OBS):
        dataset = SpectrumDatasetOnOff.from_ogip_files(filename=f"pha_obs{ind}.fits")
        dataset.model = model
        datasets.append(dataset)

    return datasets


def data_fit(datasets):
    fit = Fit(datasets)
    result = fit.run()


def flux_point(datasets):
    e_edges = MapAxis.from_bounds(0.7, 30, nbin=11, interp="log", unit="TeV").edges
    fpe = FluxPointsEstimator(datasets=datasets, e_edges=e_edges, source="gc-source")
    fpe.run()


def run_benchmark():
    info = {}

    t = time.time()

    datasets = data_prep()
    info["data_preparation"] = time.time() - t
    t = time.time()

    write(datasets)
    info["writing"] = time.time() - t
    t = time.time()

    datasets = read()
    info["reading"] = time.time() - t
    t = time.time()

    data_fit(datasets)
    info["data_fitting"] = time.time() - t
    t = time.time()

    flux_point(datasets)
    info["flux_point"] = time.time() - t

    results_folder = "results/spectrum_1d_joint/"
    subtimes_filename = results_folder + "/subtimings.yaml"
    with open(subtimes_filename, "w") as fh:
        yaml.dump(info, fh, sort_keys=False, indent=4)

    os.system("rm *.fits")


if __name__ == "__main__":
    run_benchmark()
