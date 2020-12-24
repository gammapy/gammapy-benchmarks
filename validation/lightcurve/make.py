#!/usr/bin/env python
"""Run Gammapy validation: CTA 1DC"""
import logging
import warnings
import click
from gammapy.data import DataStore
from gammapy.datasets import SpectrumDataset, MapDataset
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel, PointSpatialModel
from gammapy.maps import MapAxis, RegionGeom, WcsGeom, Map
from gammapy.estimators import LightCurveEstimator, LightCurve
from gammapy.makers import (
   SpectrumDatasetMaker,
   ReflectedRegionsBackgroundMaker,
   SafeMaskMaker,
   MapDatasetMaker,
   FoVBackgroundMaker,
)
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from regions import CircleSkyRegion
from astropy.coordinates import Angle
from gammapy.utils.scripts import make_path



log = logging.getLogger(__name__)

@click.group()
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING"]),)
@click.option("--show-warnings", is_flag=True, help="Show warnings?")

def cli(log_level, show_warnings):
   logging.basicConfig(level=log_level)
   log.setLevel(level=log_level)
   if not show_warnings:
       warnings.simplefilter("ignore")


@cli.command("run-analyses", help="Run Gammapy validation: Light curve")
@click.argument("type", type=click.Choice(["1d", "3d", "all"]))
def run_analyses(type):
    log.info("Run analysis.")

    target_position = SkyCoord(329.71693826 * u.deg, -30.2255890 * u.deg, frame="icrs")
    observations = select_data(target_position)

    t0 = Time("2006-07-29T20:30")
    duration = 10 * u.min
    n_time_bins = 35
    times = t0 + np.arange(n_time_bins) * duration
    time_intervals = [
        Time([tstart, tstop]) for tstart, tstop in zip(times[:-1], times[1:])
    ]

    log.info("Filter observations in time intervals")
    short_observations = observations.select_time(time_intervals)

    if type == "all":
        types = ["1d", "3d"]
    else:
        types = [type]

    for analysis_type in types:
        perform_analysis(analysis_type, short_observations, target_position, time_intervals)

    make_summary(types)


def select_data(target_position):
    data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")

    selection = dict(
        type="sky_circle",
        frame="icrs",
        lon=target_position.ra,
        lat=target_position.dec,
        radius=2 * u.deg,
    )
    obs_ids = data_store.obs_table.select_observations(selection)["OBS_ID"]
    observations = data_store.get_observations(obs_ids)
    return observations


def perform_analysis(type, observations, target_position, time_intervals):
    log.info(f"Dataset creation in {type}.")
    if type == "1d":
        datasets = create_datasets_1d(observations, target_position)
    else:
        datasets = create_datasets_3d(observations, target_position)


    log.info("Assign model on the datasets.")
    if type == "1d":
        sky_model = define_model_1d()
        for dataset in datasets:
            dataset.models = sky_model
    elif type == "3d":
        sky_model = define_model_3d(target_position)
        for dataset in datasets:
            dataset.models = [dataset.background_model, sky_model]

    log.info(f"Run LightCurveEstimator in {type}.")
    lc_maker = LightCurveEstimator(
        energy_edges=[0.7, 20] * u.TeV,
        source="pks2155",
        time_intervals=time_intervals
    )

    lc = lc_maker.run(datasets)

    log.info("Export results.")
    filename = make_path("results")
    filename.mkdir(exist_ok=True)
    path = filename / f"lightcurve_{type}.fits"
    log.info(f"Writing {path}")
    lc.table.write(path, overwrite=True)


def create_datasets_1d(observations, target_position):
    on_region_radius = Angle("0.11 deg")
    on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)

    # Target geometry definition
    e_reco = MapAxis.from_energy_bounds(0.4, 20, 10, "TeV")
    e_true = MapAxis.from_energy_bounds(0.1, 40, 40, "TeV", name="energy_true")

    #data reduction makers
    dataset_maker = SpectrumDatasetMaker(containment_correction=True, selection=["counts", "exposure", "edisp"])
    bkg_maker = ReflectedRegionsBackgroundMaker()
    safe_mask_maker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)

    datasets = []

    geom = RegionGeom(on_region, axes=[e_reco])
    dataset_empty = SpectrumDataset.create(geom=geom, energy_axis_true=e_true)

    for obs in observations:
        dataset = dataset_maker.run(dataset_empty.copy(), obs)
        dataset_on_off = bkg_maker.run(dataset, obs)
        if dataset_on_off.counts_off.data.sum()>0:
            dataset_on_off = safe_mask_maker.run(dataset_on_off, obs)

            datasets.append(dataset_on_off)
    return datasets

def create_datasets_3d(observations, target_position):

    # Target geometry definition
    e_reco = MapAxis.from_energy_bounds(0.4, 20, 10, "TeV")
    e_true = MapAxis.from_energy_bounds(0.1, 40, 40, "TeV", name="energy_true")

    geom = WcsGeom.create(
        skydir=target_position,
        width=(2, 2),
        binsz=0.02,
        axes=[e_reco]
    )

    exclusion_region = CircleSkyRegion(target_position, 0.3*u.deg)
    exclusion_mask = Map.from_geom(geom, data=geom.region_mask([exclusion_region], inside=False))

    offset_max = 2.0 * u.deg
    #data reduction makers
    maker = MapDatasetMaker()
    bkg_maker = FoVBackgroundMaker(method="scale", exclusion_mask=exclusion_mask)
    safe_mask_maker = SafeMaskMaker(methods=["aeff-max", "offset-max"], aeff_percent=10, offset_max=offset_max)

    datasets = []

    dataset_empty = MapDataset.create(geom=geom, energy_axis_true=e_true)

    for obs in observations:
        cutout = dataset_empty.cutout(obs.pointing_radec, width=2 * offset_max)
        # A MapDataset is filled in this cutout geometry
        dataset = maker.run(cutout, obs)
        # The data quality cut is applied
        dataset = safe_mask_maker.run(dataset, obs)
        # fit background model
        dataset = bkg_maker.run(dataset)
        print(
            f"Background norm obs {obs.obs_id}: {dataset.background_model.spectral_model.norm.value:.2f}"
        )
        datasets.append(dataset)
    return datasets


def define_model_1d():
    spectral_model = PowerLawSpectralModel(
        index=3.4,
        amplitude=2e-11 * u.Unit("1 / (cm2 s TeV)"),
        reference=1 * u.TeV,
    )
    spectral_model.parameters["index"].frozen = False
    sky_model = SkyModel(
           spatial_model=None, spectral_model=spectral_model, name="pks2155"
    )
    return sky_model

def define_model_3d(target_position):
    spatial_model = PointSpatialModel(
        lon_0=target_position.ra,
        lat_0=target_position.dec,
        frame="icrs"
    )
    spectral_model = PowerLawSpectralModel(
        index=3.4,
        amplitude=2e-11 * u.Unit("1 / (cm2 s TeV)"),
        reference=1 * u.TeV,
    )
    spectral_model.parameters["index"].frozen = False
    sky_model = SkyModel(
           spatial_model=spatial_model, spectral_model=spectral_model, name="pks2155"
    )
    return sky_model

def make_summary(types):
    log.info("Making summary plots.")
    ax=None
    for type in types:
        filename = make_path("results")
        path = filename / f"lightcurve_{type}.fits"
        lc = LightCurve.read(path)
        lc.plot(ax=ax, label=type)
        lc_ChandraNight = LightCurve.read("Flux_LC_ChandraNight_700GeV.fits")
        lc_ChandraNight.plot(ax=ax, label='ref', alpha=0.5)
    plt.legend()

    if len(types)>1:
        filename = make_path("results")
        path = filename / f"lightcurve_comparison.png"
        plt.savefig(path)
    else:
        filename = make_path("results")
        path = filename / f"lightcurve_{types[0]}.png"
        plt.savefig(path)

    plt.close()



if __name__ == "__main__":
   logging.basicConfig(level=logging.INFO)
   cli()
