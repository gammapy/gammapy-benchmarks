#!/usr/bin/env python
"""Run Gammapy validation: CTA 1DC"""
import logging
import warnings
import click
import yaml
import time
from pathlib import Path

import matplotlib.pyplot as plt
import astropy.units as u

from gammapy.analysis import AnalysisConfig, Analysis
from gammapy.modeling.models import Models, PointSpatialModel, SkyModel
from extension import ExtensionEstimator

AVAILABLE_TARGETS = ["crab", "pks2155", "msh1552"]

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
@click.argument("targets", type=click.Choice(list(AVAILABLE_TARGETS) + ["all-targets"]))
def run_analyses(targets):
    log.info("Run small source extension check.")

    info = {}

    targets = list(AVAILABLE_TARGETS) if targets == "all-targets" else [targets]

    for target in targets:
        t = time.time()

        config = AnalysisConfig.read(f"configs/config_{target}.yaml")
        analysis = Analysis(config)
        analysis.get_observations()
        info["data_preparation"] = time.time() - t

        t = time.time()

        analysis.get_datasets()
        info["data_reduction"] = time.time() - t

        models = Models.read(f"models/model_{target}.yaml")

        point_models = Models(define_model_pointlike(models[0]))
        analysis.set_models(point_models)

        t = time.time()
        analysis.run_fit()

        info["point_model_fitting"] = time.time() - t
        log.info(f"\n{point_models.to_parameters_table()}")

        log.info("Fitting extended gaussian source.")

        analysis.datasets.models = []
        analysis.set_models(models)
        t = time.time()

        analysis.run_fit()

        info["gauss_model_fitting"] = time.time() - t

        log.info(analysis.fit_result)

        log.info(f"\n{models.to_parameters_table()}")

        log.info("Extract size error, UL and stat profile.")

        t = time.time()
        analysis.models[0].spatial_model.lon_0.frozen = True
        analysis.models[0].spatial_model.lat_0.frozen = True
        analysis.models[0].spectral_model.index.frozen = True

        size_est = ExtensionEstimator(source=models[0].name,
                                      energy_edges=[0.2, 10.0]*u.TeV,
                                      selection_optional=["errn-errp", "ul", "scan"],
                                      size_min="0.08 deg", size_max="0.12 deg",
                                      size_n_values=20,
                                      reoptimize=True)
        res = size_est.run(analysis.datasets)

        info["estimator"] = time.time() - t
        t = time.time()

        log.info(res)
        plot_profile(res[0], target)

        Path(f"bench_{target}.yaml").write_text(yaml.dump(info, sort_keys=False, indent=4))
        analysis.models.to_parameters_table().write(f"results/{target}_results.ecsv", overwrite=True)


def define_model_pointlike(model):
    spatial_model = PointSpatialModel(
        lon_0 = model.spatial_model.lon_0,
        lat_0 = model.spatial_model.lat_0,
        frame=model.spatial_model.frame
    )
    spectral_model = model.spectral_model.copy()

    sky_model = SkyModel(
           spatial_model=spatial_model, spectral_model=spectral_model, name=f"{model.name}_point"
    )
    return sky_model


def plot_profile(profile, target):
    plt.semilogx(profile["sigma_scan"], profile["stat_scan"])
    plt.axvline(profile["sigma"], color='k', alpha=0.5)
    plt.axvline(profile["sigma"]-profile["sigma_errn"],linestyle='dotted', color='b', alpha=0.5)
    plt.axvline(profile["sigma"]+profile["sigma_errp"],linestyle='dotted', color='b', alpha=0.5)
    plt.axvline(profile["sigma_ul"], linestyle='dashed', color='r', alpha=0.5)
    plt.xlabel("Source gaussian sigma, deg")
    plt.ylabel("Total Stat")
    plt.title(f"{target}")
    plt.savefig(f"results/stat_profile_{target}.png")

if __name__ == "__main__":
   logging.basicConfig(level=logging.INFO)
   cli()
