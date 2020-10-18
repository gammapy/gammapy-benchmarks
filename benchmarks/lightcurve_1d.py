import os
import time
from pathlib import Path

import astropy.units as u
import numpy as np
import yaml
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from regions import CircleSkyRegion

from gammapy.data import Observation
from gammapy.datasets import SpectrumDataset
from gammapy.estimators import LightCurveEstimator
from gammapy.irf import load_cta_irfs
from gammapy.makers import SpectrumDatasetMaker
from gammapy.maps import MapAxis
from gammapy.modeling import Fit
from gammapy.modeling.models import (ExpDecayTemporalModel,
                                     PowerLawSpectralModel, SkyModel)

N_OBS = int(os.environ.get("GAMMAPY_BENCH_N_OBS", 10))

gti_t0 = Time("2020-03-01")


def simulate():

    irfs = load_cta_irfs(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )

    # Reconstructed and true energy axis
    center = SkyCoord(0.0, 0.0, unit="deg", frame="galactic")
    energy_axis = MapAxis.from_edges(
        np.logspace(-0.5, 1.0, 10), unit="TeV", name="energy", interp="log",
    )
    energy_axis_true = MapAxis.from_edges(
        np.logspace(-1.2, 2.0, 31), unit="TeV", name="energy_true", interp="log",
    )

    on_region_radius = Angle("0.11 deg")
    on_region = CircleSkyRegion(center=center, radius=on_region_radius)

    pointing = SkyCoord(0.5, 0.5, unit="deg", frame="galactic")

    spectral_model = PowerLawSpectralModel(
        index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    temporal_model = ExpDecayTemporalModel(t0="6 h", t_ref=gti_t0.mjd * u.d)
    model_simu = SkyModel(
        spectral_model=spectral_model, temporal_model=temporal_model, name="model-simu",
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
        empty = SpectrumDataset.create(
            e_reco=energy_axis,
            e_true=energy_axis_true,
            region=on_region,
            name=f"dataset_{i}",
        )
        maker = SpectrumDatasetMaker(selection=["exposure", "background", "edisp"])
        dataset = maker.run(empty, obs)
        dataset.models = model_simu
        dataset.fake()
        datasets.append(dataset)
        tstart = tstart + 2.0 * u.hr

    return datasets


def get_lc(datasets):
    spectral_model = PowerLawSpectralModel(
        index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    model_fit = SkyModel(spectral_model=spectral_model, name="model-fit",)
    for dataset in datasets:
        dataset.models = model_fit
    lc_maker_1d = LightCurveEstimator(
        e_edges=[1.0, 10.0] * u.TeV, source="model-fit", reoptimize=False
    )
    lc_1d = lc_maker_1d.run(datasets)


def fit_lc(datasets):
    spectral_model = PowerLawSpectralModel(
        index=2.0, amplitude="1e-12 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    temporal_model1 = ExpDecayTemporalModel(t0="10 h", t_ref=gti_t0.mjd * u.d)
    model = SkyModel(
        spectral_model=spectral_model,
        temporal_model=temporal_model1,
        name="model-test",
    )
    for dataset in datasets:
        dataset.models = model
    fit = Fit(datasets)
    result = fit.run()
    print(result)
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
