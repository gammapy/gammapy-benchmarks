import os
import time
from pathlib import Path

import astropy.units as u
import numpy as np
import yaml
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion

from gammapy.data import DataStore
from gammapy.datasets import Datasets, SpectrumDataset, SpectrumDatasetOnOff
from gammapy.estimators import FluxPointsEstimator
from gammapy.makers import (ReflectedRegionsBackgroundMaker, SafeMaskMaker,
                            SpectrumDatasetMaker)
from gammapy.maps import WcsGeom, MapAxis, RegionGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (PointSpatialModel, PowerLawSpectralModel,
                                     SkyModel)

N_OBS = int(os.environ.get("GAMMAPY_BENCH_N_OBS", 10))


def data_prep():
    data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
    OBS_ID = 23523
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
    mask_geom = WcsGeom.create(
        npix=(150, 150), binsz=0.05, skydir=skydir, proj="TAN", frame="galactic"
    )

    exclusion_mask = mask_geom.region_mask([exclusion_region], inside=False)

    e_reco = MapAxis.from_bounds(
        0.1, 40, nbin=40, interp="log", unit="TeV", name="energy"
    )
    e_true = MapAxis.from_bounds(
        0.05, 100, nbin=200, interp="log", unit="TeV", name="energy_true"
    )

    geom = RegionGeom(on_region, axes=[e_reco])

    stacked = SpectrumDatasetOnOff.create(geom=geom, energy_axis_true=e_true, name="stacked")

    dataset_maker = SpectrumDatasetMaker(
        containment_correction=False, selection=["counts", "exposure", "edisp"]
    )

    bkg_maker = ReflectedRegionsBackgroundMaker(exclusion_mask=exclusion_mask)
    safe_mask_masker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)

    spectral_model = PowerLawSpectralModel(
        index=2, amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
    )
    spatial_model = PointSpatialModel(
        lon_0=target_position.ra, lat_0=target_position.dec, frame="icrs"
    )
    spatial_model.lon_0.frozen = True
    spatial_model.lat_0.frozen = True

    sky_model = SkyModel(
        spatial_model=spatial_model, spectral_model=spectral_model, name=""
    )

    for observation in observations:
        dataset = stacked.copy(name=f"dataset-{observation.obs_id}")
        dataset = dataset_maker.run(dataset=dataset, observation=observation)
        dataset_on_off = bkg_maker.run(dataset, observation)
        dataset_on_off = safe_mask_masker.run(dataset_on_off, observation)
        stacked.stack(dataset_on_off)

    stacked.models = sky_model
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
    fit = Fit(stacked)
    result = fit.run(optimize_opts={"print_level": 1})
    print(result.success)


def flux_point(stacked):
    e_edges = MapAxis.from_bounds(0.7, 30, nbin=11, interp="log", unit="TeV").edges
    fpe = FluxPointsEstimator(energy_edges=e_edges)
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
