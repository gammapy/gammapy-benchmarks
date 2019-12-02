#!/usr/bin/env python
"""Run Gammapy validation: CTA 1DC"""
import logging
from pathlib import Path
import yaml
import click
import warnings
from astropy.table import Table
from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.modeling.models import SkyModels

log = logging.getLogger(__name__)

AVAILABLE_TARGETS = ["cas_a", "hess_j1702"]


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING"]),
)
@click.option("--show-warnings", is_flag=True, help="Show warnings?")
def cli(log_level, show_warnings):
    logging.basicConfig(level=log_level)
    log.setLevel(level=log_level)

    if not show_warnings:
        warnings.simplefilter("ignore")


@cli.command("run-analyses", help="Run Gammapy validation: CTA 1DC")
@click.argument("targets", type=click.Choice(list(AVAILABLE_TARGETS) + ["all"]))
def run_analyses(targets):
    if targets == "all":
        targets = list(AVAILABLE_TARGETS)
    else:
        targets = [targets]

    for target in targets:
        analysis_3d(target)


def analysis_3d_data_reduction(target):
    log.info(f"analysis_3d_data_reduction: {target}")

    opts = yaml.safe_load(open("targets.yaml"))[target]

    txt = Path("config_template.yaml").read_text()
    txt = txt.format_map(opts)
    config = AnalysisConfig.from_yaml(txt)


    analysis = Analysis(config)
    analysis.get_observations()
    log.info("Running data reduction")
    analysis.get_datasets()

    # TODO: write datasets and separate fitting to next function
    # Not implemented in Gammapy yet, coming very soon.
    log.info("Running fit ...")
    analysis.read_models(f"{target}/model_3d.yaml")
    logging.info(analysis.models)
    analysis.run_fit()
    logging.info(analysis.fit_result.parameters.to_table())
    path = f"{target}/{target}_3d_bestfit.rst"
    log.info(f"Writing {path}")
    analysis.fit_result.parameters.to_table().write(
        path, format="ascii.rst", overwrite=True
    )

    #    analysis.get_flux_points(source=f"{target}")
    #    path = f"{target}/{target}_3d_fluxpoints.fits"
    #    log.info(f"Writing {path}")
    #    analysis.flux_points.write(path, overwrite=True)

    analysis.get_flux_points(source=f"{target}")
    path = f"{target}/{target}_3d_fluxpoints.ecsv"
    log.info(f"Writing {path}")
    keys = ["e_ref", "e_min", "e_max", "dnde", "dnde_errp", "dnde_errn", "is_ul"]
    analysis.flux_points.data.table_formatted[keys].write(
        path, format="ascii.ecsv", overwrite=True
    )


def analysis_3d_modeling(target):
    log.info(f"analysis_3d_modeling: {target}")


def analysis_3d_summary(target):
    log.info(f"analysis_3d_summary: {target}")
    # TODO: make plots
    # TODO: summarise results to `results.md`? Necessary?

    path = f"{target}/{target}_3d_bestfit.rst"
    tab = Table.read(path, format="ascii")
    tab.add_index("name")
    dt = "U30"
    comp_tab = Table(names=("Param", "DC1 Ref", "gammapy 3d"), dtype=[dt, dt, dt])

    ref_model = SkyModels.read(f"{target}/reference/dc1_model_3d.yaml")
    pars = ref_model.parameters.names
    pars.remove("reference")  # need to find a better way to handle this

    for par in pars:
        ref = ref_model.parameters[par].value
        value = tab.loc[par]["value"]
        name = tab.loc[par]["name"]
        error = tab.loc[par]["error"]
        comp_tab.add_row([name, ref, f"{value}±{error}"])

    path = f"{target}/README.md"
    comp_tab.write(path, format="ascii.html", overwrite=True)

def analysis_3d(target):
    log.info(f"analysis_3d: {target}")
    analysis_3d_data_reduction(target)
    analysis_3d_modeling(target)
    analysis_3d_summary(target)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
