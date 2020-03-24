import os
from pathlib import Path
import numpy as np
import astropy.units as u
import time
import yaml
from astropy.coordinates import SkyCoord, Angle
from astropy.time import Time
from gammapy.data import Observation
from gammapy.irf import load_cta_irfs
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    ExpDecayTemporalModel,
    GaussianSpatialModel,
    SkyModel,
)
from gammapy.maps import MapAxis, WcsGeom
from gammapy.estimators import LightCurveEstimator
from gammapy.datasets import MapDataset, Datasets
from gammapy.makers import MapDatasetMaker, SafeMaskMaker
from gammapy.modeling import Fit

N_OBS = int(os.environ.get("GAMMAPY_BENCH_N_OBS", 10))
gti_t0 = Time("2020-03-01")


def simulate():

    irfs = load_cta_irfs(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )

    center = SkyCoord(0.0, 0.0, unit="deg", frame="galactic")
    energy_reco = MapAxis.from_edges(
        np.logspace(-1.0, 1.0, 10), unit="TeV", name="energy", interp="log"
    )
    pointing = SkyCoord(0.5, 0.5, unit="deg", frame="galactic")
    geom = WcsGeom.create(
        skydir=center, binsz=0.02, width=(6, 6), frame="galactic", axes=[energy_reco],
    )
    energy_true = MapAxis.from_edges(
        np.logspace(-1.5, 1.5, 30), unit="TeV", name="energy", interp="log"
    )

    spectral_model = PowerLawSpectralModel(
        index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    temporal_model = ExpDecayTemporalModel(t0="6 h", t_ref=gti_t0.mjd)
    spatial_model = GaussianSpatialModel(
        lon_0="0.2 deg", lat_0="0.1 deg", sigma="0.3 deg", frame="galactic"
    )
    model_simu = SkyModel(
        spectral_model=spectral_model,
        spatial_model=spatial_model,
        temporal_model=temporal_model,
        name="model-simu",
    )

    lvtm = np.ones(N_OBS) * 1.0 * u.hr
    tstart = 1.0 * u.hr

    datasets = []
    for i in range(N_OBS):
        obs = Observation.create(
            pointing=pointing,
            livetime=lvtm[i],
            tstart=tstart,
            irfs=irfs,
            reference_time=gti_t0,
        )
        empty = MapDataset.create(geom, name=f"dataset_{i}")
        maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
        maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=4.0 * u.deg)
        dataset = maker.run(empty, obs)
        dataset = maker_safe_mask.run(dataset, obs)
        dataset.models.append(model_simu)
        dataset.fake()
        datasets.append(dataset)
        tstart = tstart + 2.0 * u.hr

    return datasets


def get_lc(datasets):
    spatial_model1 = GaussianSpatialModel(
        lon_0="0.1 deg", lat_0="0.1 deg", sigma="0.5 deg", frame="galactic"
    )
    spectral_model1 = PowerLawSpectralModel(
        index=2, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    model_fit = SkyModel(
        spatial_model=spatial_model1, spectral_model=spectral_model1, name="model_fit",
    )
    for dataset in datasets:
        dataset.models[1] = model_fit
    lc_maker = LightCurveEstimator(datasets, source="model_fit", reoptimize=False)
    lc = lc_maker.run(e_ref=1 * u.TeV, e_min=1.0 * u.TeV, e_max=10.0 * u.TeV)
    print(lc.table)


def fit_lc(datasets):
    spatial_model1 = GaussianSpatialModel(
        lon_0="0.2 deg", lat_0="0.1 deg", sigma="0.3 deg", frame="galactic"
    )
    spatial_model1.parameters["lon_0"].frozen = True
    spatial_model1.parameters["lat_0"].frozen = True
    spatial_model1.parameters["sigma"].frozen = True
    spectral_model1 = PowerLawSpectralModel(
        index=2, amplitude="2e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    temporal_model1 = ExpDecayTemporalModel(t0="10 h", t_ref=gti_t0.mjd)
    model_fit = SkyModel(
        spatial_model=spatial_model1,
        spectral_model=spectral_model1,
        temporal_model=temporal_model1,
        name="model_fit",
    )

    for dataset in datasets:
        dataset.models[1] = model_fit
        dataset.background_model.parameters["norm"].frozen = True
    fit = Fit(datasets)
    result = fit.optimize()
    print(result.parameters.to_table())


def run_benchmark():
    info = {"n_obs": N_OBS}

    t = time.time()

    datasets = simulate()
    info["simulations"] = time.time() - t
    t = time.time()

    get_lc(datasets)
    info["lc estimation"] = time.time() - t
    t = time.time()

    fit_lc(datasets)
    info["temporal fitting"] = time.time() - t
    t = time.time()

    Path("bench.yaml").write_text(yaml.dump(info, sort_keys=False, indent=4))


if __name__ == "__main__":
    run_benchmark()
