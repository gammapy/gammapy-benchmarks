import numpy as np
import astropy.units as u
import time
import yaml
import tempfile
import pathlib
from astropy.coordinates import SkyCoord
from gammapy.modeling.models import (
    SkyModel,
    ExpCutoffPowerLawSpectralModel,
    PointSpatialModel,
)
from gammapy.spectrum import FluxPointsEstimator
from gammapy.modeling import Fit
from gammapy.data import DataStore
from gammapy.maps import MapAxis, WcsGeom
from gammapy.cube import MapDatasetMaker, MapDataset, SafeMaskMaker

import os

N_OBS = 10
OBS_ID = 110380


def data_prep():
    # Create maps
    data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")
    obs_ids = OBS_ID * np.ones(N_OBS)
    observations = data_store.get_observations(obs_ids)

    energy_axis = MapAxis.from_bounds(
        0.1, 10, nbin=10, unit="TeV", name="energy", interp="log"
    )
    geom = WcsGeom.create(
        skydir=(0, 0),
        binsz=0.02,
        width=(10, 8),
        coordsys="GAL",
        proj="CAR",
        axes=[energy_axis],
    )

    src_pos = SkyCoord(0, 0, unit="deg", frame="galactic")
    offset_max = 4 * u.deg
    maker = MapDatasetMaker(offset_max=offset_max)
    safe_mask_maker = SafeMaskMaker(methods=["offset-max"], offset_max="4 deg")
    stacked = MapDataset.create(geom=geom)

    datasets = []
    for obs in observations:
        dataset = maker.run(stacked, obs)
        dataset = safe_mask_maker.run(dataset, obs)
        dataset.edisp = dataset.edisp.get_energy_dispersion(
            position=src_pos, e_reco=energy_axis.edges
        )
        dataset.psf = dataset.psf.get_psf_kernel(
            position=src_pos, geom=geom, max_radius="0.3 deg"
        )

        datasets.append(dataset)
    return datasets


def write(datasets):
    for ind, dataset in enumerate(datasets):
        dataset.write(f"dataset-{ind}.fits", overwrite=True)


def read():

    datasets = []
    spatial_model = PointSpatialModel(
        lon_0="-0.05 deg", lat_0="-0.05 deg", frame="galactic"
    )
    spectral_model = ExpCutoffPowerLawSpectralModel(
        index=2,
        amplitude=3e-12 * u.Unit("cm-2 s-1 TeV-1"),
        reference=1.0 * u.TeV,
        lambda_=0.1 / u.TeV,
    )
    model = SkyModel(
        spatial_model=spatial_model, spectral_model=spectral_model, name="gc-source"
    )
    for ind in range(N_OBS):
        dataset = MapDataset.read(f"dataset-{ind}.fits")
        dataset.model = model
        datasets.append(dataset)

    return datasets


def data_fit(datasets):
    fit = Fit(datasets)
    result = fit.run()


def flux_point(datasets):
    e_edges = [0.3, 1, 3, 10] * u.TeV
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

    results_folder = "results/analysis_3d_joint/"
    subtimes_filename = results_folder + "/subtimings.yaml"
    with open(subtimes_filename, "w") as fh:
        yaml.dump(info, fh, sort_keys=False, indent=4)

    os.system("rm *.fits")


if __name__ == "__main__":
    run_benchmark()
