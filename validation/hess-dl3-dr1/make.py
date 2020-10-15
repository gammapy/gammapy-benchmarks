import logging
import time
import warnings
from pathlib import Path

import click
import yaml

import numpy as np
from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.modeling.models import Models

log = logging.getLogger(__name__)

AVAILABLE_TARGETS = ["crab", "pks2155", "msh1552"]
AVAILABLE_METHODS = ["1d", "3d"]


@click.group()
@click.option(
    "--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING"])
)
@click.option("--show-warnings", is_flag=True, help="Show warnings?")
def cli(log_level, show_warnings):
    logging.basicConfig(level=log_level)

    if not show_warnings:
        warnings.simplefilter("ignore")


@cli.command("run-analyses", help="Run DL3 analysis validation")
@click.option("--debug", is_flag=True)
@click.option("--skip_flux_points", is_flag=True)
@click.argument("targets", type=click.Choice(list(AVAILABLE_TARGETS) + ["all-targets"]))
@click.argument("methods", type=click.Choice(list(AVAILABLE_METHODS) + ["all-methods"]))
def run_analyses(debug, skip_flux_points, targets, methods):
    start_time = time.time()
    targets = list(AVAILABLE_TARGETS) if targets == "all-targets" else [targets]
    methods = list(AVAILABLE_METHODS) if methods == "all-methods" else [methods]

    with open("targets.yaml", "r") as stream:
        targets_file = yaml.safe_load(stream)

    for target in targets:
        target_filter = filter(lambda _: _["tag"] == target, targets_file)
        target_dict = list(target_filter)[0]

        log.info(f"Processing source: {target}")
        for method in methods:
            run_analysis(method, target_dict, debug, skip_flux_points)
    end_time = time.time()
    duration = end_time - start_time
    log.info(f"The time taken for the validation is: {duration} s ({duration/60} min)")


def write_fit_summary(parameters, outfile):
    """Store fit results with uncertainties"""
    fit_results_dict = {}
    for parameter in parameters:
        value = parameter.value
        error = parameter.error
        name = parameter.name
        fit_results_dict.update({name: value})
        fit_results_dict.update({name + "_err": float(error)})
    with open(str(outfile), "w") as f:
        yaml.dump(fit_results_dict, f)


def run_analysis(method, target_dict, debug, skip_flux_points):
    """If the method is "1d", runs joint spectral analysis for the selected target. If
    instead it is "3d", runs stacked 3D analysis."""
    tag = target_dict["tag"]
    log.info(f"Running {method} analysis, {tag}")
    path_res = Path(tag + "/results/")

    log.info("Reading config")
    txt = Path(f"config_{method}.yaml").read_text()
    txt = txt.format_map(target_dict)
    config = AnalysisConfig.from_yaml(txt)

    # fixme
    config.datasets.safe_mask.methods = ["edisp-bias", "offset-max"]
    config.datasets.safe_mask.parameters = {"offset_max": "2.5 deg"}

    if debug:
        config.observations.obs_ids = [target_dict["debug_run"]]
        config.flux_points.energy.nbins = 1
        if method == "3d":
            config.datasets.geom.axes.energy_true.nbins = 10
    analysis = Analysis(config)

    log.info("Running observations selection")
    analysis.get_observations()

    # there are background rates of zero present which we fix here:

    for obs in analysis.observations:
        bkg = obs.bkg

        for data in bkg.data.data:
            is_zero = (data == 0)
            data[is_zero] = np.min(data)

        obs.bkg = bkg

    log.info(f"Running data reduction")
    analysis.get_datasets()

    log.info(f"Setting the model")
    models = Models.read(f"{tag}/model.yaml")
    analysis.set_models(models)

    if method == "3d":
        analysis.datasets[0].background_model.spectral_model.norm.frozen = False
        analysis.datasets[0].background_model.spectral_model.tilt.frozen = False
        
        # Impose min and max values to ensure position does not diverge
        delta = 1.5
        lon = analysis.models[0].spatial_model.lon_0.value
        lat = analysis.models[0].spatial_model.lat_0.value
        analysis.models[0].spatial_model.lat_0.min = lat - delta
        analysis.models[0].spatial_model.lat_0.max = lat + delta
        analysis.models[0].spatial_model.lon_0.min = lon - delta
        analysis.models[0].spatial_model.lon_0.max = lon + delta

        if target_dict["spatial_model"] == "DiskSpatialModel":
            analysis.models[0].spatial_model.e.frozen = False
            analysis.models[0].spatial_model.phi.frozen = False
            analysis.models[0].spatial_model.r_0.value = 0.3
    log.info(f"Running fit ...")

    analysis.run_fit(optimize_opts={"print_level": 3})

    log.info(f"Writing {path_res}")
    write_fit_summary(
        analysis.models[0].parameters, str(path_res / f"result-{method}.yaml")
    )

    if not skip_flux_points:
        log.info(f"Running flux points estimation")
        # Freeze all parameters except the backround norm
        if method == "3d":
            dataset = analysis.datasets[0]
            for parameter in dataset.models.parameters:
                if parameter is not dataset.background_model.spectral_model.norm:
                    parameter.frozen = True
        analysis.get_flux_points()
        flux_points = analysis.flux_points.data
        flux_points.table["is_ul"] = flux_points.table["ts"] < 4
        keys = [
            "e_ref",
            "e_min",
            "e_max",
            "dnde",
            "dnde_errp",
            "dnde_errn",
            "is_ul",
            "dnde_ul",
        ]
        log.info(f"Writing {path_res}")
        flux_points.table_formatted[keys].write(
            path_res / f"flux-points-{method}.ecsv", format="ascii.ecsv"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
