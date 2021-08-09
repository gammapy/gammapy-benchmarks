#!/usr/bin/env python
"""Run Gammapy validation: CTA 1DC"""
import logging
import warnings
import click
from gammapy.data import DataStore
from gammapy.datasets import MapDataset
from gammapy.modeling import Fit
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel, PointSpatialModel, GaussianSpatialModel, FoVBackgroundModel
from gammapy.maps import MapAxis, WcsGeom, Map
from gammapy.makers import (
   SafeMaskMaker,
   MapDatasetMaker,
   FoVBackgroundMaker,
)
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
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


@cli.command("run-analyses", help="Run Gammapy validation: small extension validation.")
def run_analyses():
    log.info("Run small source extension check.")

    target_position = SkyCoord(329.71693826 * u.deg, -30.2255890 * u.deg, frame="icrs")

    log.info("Exteract observations from datastore.")
    observations = select_data()

    binsz=0.02
    log.info(f"Performing data reduction with bin size {binsz}.")
    dataset = create_dataset_3d(observations, target_position, binsz)
    bkg_model = FoVBackgroundModel(dataset_name="stacked")

    log.info("Fitting point source.")
    point_model = define_model_pointlike(target_position)
    dataset.models = [bkg_model, point_model]
    fit = Fit()
    result = fit.run([dataset])
    log.info(result)
    log.info(result["optimize_result"].parameters.to_table())

    log.info("Fitting extended gaussian source.")
    gauss_model = define_model_gaussian(target_position)
    dataset.models = [bkg_model, gauss_model]
    result = fit.run([dataset])
    log.info(result)
    log.info(result["optimize_result"].parameters.to_table())
    log.info("Fitting extended gaussian source.")
    log.info("Extract size UL and stat profile.")
    conf_result = fit.confidence([dataset], "sigma", 3)
    log.info(conf_result)

    gauss_model.spatial_model.sigma.scan_values = np.logspace(-3.5,-1.5,10)
    profile = fit.stat_profile([dataset], "sigma", reoptimize=True)
    plot_profile(profile)



def select_data():
    data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
    obs_ids = [33787, 33788, 33789, 33790, 33791, 33792, 33793, 33794, 33795, 33796, 33797, 33798, 33799, 33800, 33801]

    observations = data_store.get_observations(obs_ids)
    return observations

def create_dataset_3d(observations, center_position, binsz):

    # Target geometry definition
    e_reco = MapAxis.from_energy_bounds(0.4, 20, 10, "TeV")
    e_true = MapAxis.from_energy_bounds(0.1, 40, 40, "TeV", name="energy_true")

    geom = WcsGeom.create(
        skydir=center_position,
        width=(2, 2),
        binsz=binsz,
        axes=[e_reco]
    )

    exclusion_region = CircleSkyRegion(center_position, 0.3*u.deg)
    exclusion_mask = geom.region_mask([exclusion_region], inside=False)

    offset_max = 2.0 * u.deg
    #data reduction makers
    maker = MapDatasetMaker()
    bkg_maker = FoVBackgroundMaker(method="scale", exclusion_mask=exclusion_mask)
    safe_mask_maker = SafeMaskMaker(methods=["aeff-max", "offset-max"], aeff_percent=10, offset_max=offset_max)

    stacked = MapDataset.create(geom=geom, energy_axis_true=e_true, name="stacked")

    for obs in observations:
        cutout = stacked.cutout(obs.pointing_radec, width=2 * offset_max)
        # A MapDataset is filled in this cutout geometry
        dataset = maker.run(cutout, obs)
        # The data quality cut is applied
        dataset = safe_mask_maker.run(dataset, obs)
        # fit background model
        dataset = bkg_maker.run(dataset)
        print(
            f"Background norm obs {obs.obs_id}: {dataset.background_model.spectral_model.norm.value:.2f}"
        )
        stacked.stack(dataset)
    return stacked

def define_model_pointlike(test_position):
    spatial_model = PointSpatialModel(
        lon_0=test_position.ra,
        lat_0=test_position.dec,
        frame="icrs"
    )
    spectral_model = PowerLawSpectralModel(
        index=3.4,
        amplitude=2e-11 * u.Unit("1 / (cm2 s TeV)"),
        reference=1 * u.TeV,
    )
    spectral_model.parameters["index"].frozen = False
    sky_model = SkyModel(
           spatial_model=spatial_model, spectral_model=spectral_model, name="point"
    )
    return sky_model

def define_model_gaussian(test_position):
    spatial_model = GaussianSpatialModel(
        lon_0=test_position.ra,
        lat_0=test_position.dec,
        frame="icrs",
        sigma="0.02 deg"
    )
    spectral_model = PowerLawSpectralModel(
        index=3.4,
        amplitude=2e-11 * u.Unit("1 / (cm2 s TeV)"),
        reference=1 * u.TeV,
    )
    spectral_model.parameters["index"].frozen = False
    sky_model = SkyModel(
           spatial_model=spatial_model, spectral_model=spectral_model, name="gauss"
    )
    return sky_model

def plot_profile(profile):
    plt.semilogx(profile["sigma_scan"], profile["stat_scan"])
    plt.xlabel("Source gaussian sigma, deg")
    plt.ylabel("Total Stat")
    plt.savefig("stat_profile.png")

if __name__ == "__main__":
   logging.basicConfig(level=logging.INFO)
   cli()
