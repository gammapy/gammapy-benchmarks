#!/usr/bin/env python
"""Run Gammapy validation: CTA 1DC"""
import logging
from pathlib import Path

import yaml
import astropy.units as u
from astropy.coordinates import Angle
from gammapy.analysis import Analysis, AnalysisConfig

log = logging.getLogger(__name__)


def get_config(target):
    config = yaml.safe_load(open("targets.yaml"))
    return config[target]


def cli():
    targets = "all"
    if targets == "all":
        targets = ["cas_a", "hess_j1702"]
    else:
        targets = targets.split(",")

    for target in targets:
        analysis_3d(target)


def analysis_3d(target):
    log.info(f"analysis_3d: {target}")
    analysis_3d_data_reduction(target)
    analysis_3d_modeling(target)
    analysis_3d_summary(target)


def analysis_3d_data_reduction(target):
    log.info(f"analysis_3d_data_reduction: {target}")

    opts = get_config(target)
    # TODO: remove this once Gammapy config is more uniform
    opts["emin_tev"] = u.Quantity(opts["emin"]).to_value("TeV")
    opts["emax_tev"] = u.Quantity(opts["emax"]).to_value("TeV")
    opts["lon_deg"] = Angle(opts["lon"]).deg
    opts["lat_deg"] = Angle(opts["lat"]).deg

    txt = Path("config_template.yaml").read_text()
    txt = txt.format_map(opts)
    config = yaml.safe_load(txt)
    config = AnalysisConfig(config)

    analysis = Analysis(config)
    analysis.get_observations()

    log.info("Running data reduction")
    analysis.get_datasets()

    log.info("Running fit ...")
    analysis.set_model(filename=f"{target}/model_3d.yaml")
    logging.info(analysis.model)
    analysis.run_fit()
    logging.info(analysis.fit_result.parameters.to_table())
    path = f"{target}/{target}_3d_bestfit.dat"
    log.info(f"Writing {path}")
    analysis.fit_result.parameters.to_table().write(path, format="ascii", overwrite=True)

    analysis.get_flux_points(source=f"{target}")
    path = f"{target}/{target}_3d_fluxpoints.fits"
    log.info(f"Writing {path}")
    analysis.flux_points.write(path, overwrite=True)


def analysis_3d_modeling(target):
    log.info(f"analysis_3d_modeling: {target}")


def analysis_3d_summary(target):
    log.info(f"analysis_3d_summary: {target}")
    # TODO: make plots
    # TODO: summarise results to `results.md`? Necessary?

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()

