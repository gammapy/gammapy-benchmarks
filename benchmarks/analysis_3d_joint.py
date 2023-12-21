import os
import time
from pathlib import Path

import astropy.units as u
import numpy as np
import yaml

from gammapy.data import DataStore
from gammapy.datasets import Datasets, MapDataset
from gammapy.estimators import FluxPointsEstimator
from gammapy.makers import MapDatasetMaker, SafeMaskMaker
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (ExpCutoffPowerLawSpectralModel,
                                     PointSpatialModel, SkyModel, FoVBackgroundModel)

N_OBS = int(os.environ.get("GAMMAPY_BENCH_N_OBS", 10))


def data_prep():
    data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")
    OBS_ID = 110380
    obs_ids = OBS_ID * np.ones(N_OBS)
    observations = data_store.get_observations(obs_ids)

    energy_axis = MapAxis.from_bounds(
        0.1, 10, nbin=10, unit="TeV", name="energy", interp="log"
    )
    geom = WcsGeom.create(
        skydir=(0, 0),
        binsz=0.02,
        width=(10, 8),
        frame="galactic",
        proj="CAR",
        axes=[energy_axis],
    )

    energy_axis_true = MapAxis.from_bounds(
        0.05, 20, nbin=30, unit="TeV", name="energy_true", interp="log"
    )

    offset_max = 4 * u.deg
    maker = MapDatasetMaker()
    safe_mask_maker = SafeMaskMaker(methods=["offset-max"], offset_max=offset_max)
    stacked = MapDataset.create(geom=geom, energy_axis_true=energy_axis_true)

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

    datasets = Datasets([])
    for idx, obs in enumerate(observations):
        cutout = stacked.cutout(
            obs.get_pointing_icrs(obs.tmid), width=2 * offset_max, name=f"dataset{idx}"
        )
        dataset = maker.run(cutout, obs)
        dataset = safe_mask_maker.run(dataset, obs)
        dataset.models = [model, FoVBackgroundModel(dataset_name=dataset.name)]
        datasets.append(dataset)
    return datasets


def write(datasets, filename):
    path = Path.cwd()
    datasets.write( path / f"{filename}_datasets.yaml", filename_models= path / f"{filename}_models.yaml", overwrite=True)


def read(filename):
    path = Path.cwd()
    return Datasets.read(
        path / f"{filename}_datasets.yaml", filename_models= path / f"{filename}_models.yaml"
    )


def data_fit(datasets):
    fit = Fit()
    result = fit.run(datasets=datasets)


def flux_point(datasets):
    e_edges = [0.3, 1, 3, 10] * u.TeV
    fpe = FluxPointsEstimator(energy_edges=e_edges, source="gc-source")
    fpe.run(datasets=datasets)


def run_benchmark():
    info = {"n_obs": N_OBS}
    filename = "joint"

    t = time.time()

    datasets = data_prep()
    info["data_preparation"] = time.time() - t
    t = time.time()

    write(datasets, filename)
    info["writing"] = time.time() - t
    t = time.time()

    datasets = read(filename)
    info["reading"] = time.time() - t
    t = time.time()

    data_fit(datasets)
    info["data_fitting"] = time.time() - t
    t = time.time()

    flux_point(datasets)
    info["flux_point"] = time.time() - t

    Path("bench.yaml").write_text(yaml.dump(info, sort_keys=False, indent=4))


if __name__ == "__main__":
    run_benchmark()
