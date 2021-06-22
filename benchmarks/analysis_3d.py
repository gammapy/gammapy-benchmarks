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
        binsz=0.05,
        width=(10, 8),
        frame="galactic",
        proj="CAR",
        axes=[energy_axis],
    )

    stacked = MapDataset.create(geom, name="stacked_ds")
    maker = MapDatasetMaker()
    safe_mask_maker = SafeMaskMaker(methods=["offset-max"], offset_max="4 deg")
    for obs in observations:
        dataset = maker.run(stacked, obs)
        dataset = safe_mask_maker.run(dataset, obs)
        stacked.stack(dataset)

    spatial_model = PointSpatialModel(
        lon_0="0.01 deg", lat_0="0.01 deg", frame="galactic"
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

    stacked.models = [model, FoVBackgroundModel(dataset_name=stacked.name)]
    return Datasets([stacked])


def write(stacked, filename):
    path = Path.cwd()
    stacked.write( path / f"{filename}_datasets.yaml", filename_models= path / f"{filename}_models.yaml", overwrite=True)


def read(filename):
    path = Path.cwd()
    return Datasets.read(
        path / f"{filename}_datasets.yaml", filename_models= path / f"{filename}_models.yaml"
    )


def data_fit(stacked):
    # Data fitting
    fit = Fit(optimize_opts={"print_level": 1})
    result = fit.run(datasets=stacked)


def flux_point(stacked):
    e_edges = [0.3, 1, 3, 10] * u.TeV
    fpe = FluxPointsEstimator(energy_edges=e_edges, source="gc-source")
    fpe.run(datasets=stacked)


def run_benchmark():
    info = {"n_obs": N_OBS}
    filename = "stacked"

    t = time.time()

    stacked = data_prep()
    info["data_preparation"] = time.time() - t
    t = time.time()

    write(stacked, filename)
    info["writing"] = time.time() - t
    t = time.time()

    stacked = read(filename)
    info["reading"] = time.time() - t
    t = time.time()

    data_fit(stacked)
    info["data_fitting"] = time.time() - t
    t = time.time()

    flux_point(stacked)
    info["flux_point"] = time.time() - t

    Path("bench.yaml").write_text(yaml.dump(info, sort_keys=False, indent=4))


if __name__ == "__main__":
    run_benchmark()
