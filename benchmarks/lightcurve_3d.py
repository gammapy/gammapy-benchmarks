import os
import time
from pathlib import Path

import astropy.units as u
import numpy as np
import yaml
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time

from gammapy.data import Observation
from gammapy.datasets import Datasets, MapDataset
from gammapy.estimators import LightCurveEstimator
from gammapy.irf import load_cta_irfs
from gammapy.makers import MapDatasetMaker, SafeMaskMaker
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (ExpDecayTemporalModel,
                                     GaussianSpatialModel,
                                     PowerLawSpectralModel, SkyModel,
                                     FoVBackgroundModel)

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
        skydir=center, binsz=0.02, width=(4, 4), frame="galactic", axes=[energy_reco],
    )
    energy_true = MapAxis.from_edges(
        np.logspace(-1.5, 1.5, 30), unit="TeV", name="energy_true", interp="log"
    )

    spectral_model = PowerLawSpectralModel(
        index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    temporal_model = ExpDecayTemporalModel(t0="6 h", t_ref=gti_t0.mjd * u.d)
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
        empty = MapDataset.create(
            geom, name=f"dataset_{i}", energy_axis_true=energy_true
        )
        maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
        maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=4.0 * u.deg)
        dataset = maker.run(empty, obs)
        dataset = maker_safe_mask.run(dataset, obs)
        dataset.models = [model_simu, FoVBackgroundModel(dataset_name=dataset.name)]
        dataset.fake()
        datasets.append(dataset)
        tstart = tstart + 2.0 * u.hr

    return datasets


def get_lc(datasets):
    spatial_model1 = GaussianSpatialModel(
        lon_0="0.2 deg", lat_0="0.1 deg", sigma="0.3 deg", frame="galactic"
    )
    spatial_model1.parameters["lon_0"].frozen = True
    spatial_model1.parameters["lat_0"].frozen = True
    spatial_model1.parameters["sigma"].frozen = True
    spectral_model1 = PowerLawSpectralModel(
        index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    model_fit = SkyModel(
        spatial_model=spatial_model1, spectral_model=spectral_model1, name="model_fit",
    )
    for dataset in datasets:
        dataset.models = [model_fit, FoVBackgroundModel(dataset_name=dataset.name)]
    lc_maker = LightCurveEstimator(
        energy_edges=[1.0, 10.0] * u.TeV, source="model_fit", reoptimize=False
    )
    lc = lc_maker.run(datasets)
    print(lc.to_table(format="lightcurve", sed_type="flux")["flux"])


def fit_lc(datasets):
    spatial_model1 = GaussianSpatialModel(
        lon_0="0.2 deg", lat_0="0.1 deg", sigma="0.3 deg", frame="galactic"
    )
    spatial_model1.parameters["lon_0"].frozen = False
    spatial_model1.parameters["lat_0"].frozen = False
    spatial_model1.parameters["sigma"].frozen = True
    spectral_model1 = PowerLawSpectralModel(
        index=3, amplitude="2e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    temporal_model1 = ExpDecayTemporalModel(t0="10 h", t_ref=gti_t0.mjd * u.d)
    model_fit = SkyModel(
        spatial_model=spatial_model1,
        spectral_model=spectral_model1,
        temporal_model=temporal_model1,
        name="fit",
    )

    for dataset in datasets:
        dataset.models = [model_fit, FoVBackgroundModel(dataset_name=dataset.name)]
        dataset.background_model.parameters["norm"].frozen = True

    fit = Fit()
    result = fit.run(datasets=datasets)
    print(result.success)
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
