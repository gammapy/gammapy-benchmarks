import logging
import yaml
import click
import warnings
from pathlib import Path
from gammapy.analysis import Analysis, AnalysisConfig

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
@click.argument("targets", type=click.Choice(list(AVAILABLE_TARGETS) + ["all-targets"]))
@click.argument("methods", type=click.Choice(list(AVAILABLE_METHODS) + ["all-methods"]))
def run_analyses(debug, targets, methods):
    targets = list(AVAILABLE_TARGETS) if targets == "all-targets" else [targets]
    methods = list(AVAILABLE_METHODS) if methods == "all-methods" else [methods]

    with open("targets.yaml", "r") as stream:
        targets_file = yaml.safe_load(stream)

    for target in targets:
        target_filter = filter(lambda _: _["tag"] == target, targets_file)
        target_dict = list(target_filter)[0]

        log.info(f"Processing source: {target}")
        for method in methods:
            run_analysis(method, target_dict, debug)


def write_fit_summary(parameters, outfile):
    """Store fit results with uncertainties"""
    fit_results_dict = {}
    for parameter in parameters:
        value = parameter.value
        error = parameters.error(parameter)
        unit = parameter.unit
        name = parameter.name
        string = "{0:.2e} +- {1:.2e} {2}".format(value, error, unit)
        fit_results_dict.update({name: string})
    with open(str(outfile), "w") as f:
        yaml.dump(fit_results_dict, f)


def run_analysis(method, target_dict, debug):
    """If the method is "1d", runs joint spectral analysis for the selected target. If
    instead it is "3d", runs stacked 3D analysis."""
    tag = target_dict["tag"]
    log.info(f"Running {method} analysis, {tag}")
    path_res = Path(tag + "/results/")

    log.info("Reading config")
    txt = Path(f"config_{method}.yaml").read_text()
    txt = txt.format_map(target_dict)
    config = AnalysisConfig.from_yaml(txt)
    if debug:
        config.observations.obs_ids = [target_dict["debug_run"]]
        config.flux_points.energy.nbins = 1
        if method == "3d":
            config.datasets.geom.axes.energy_true.nbins = 10
    analysis = Analysis(config)

    log.info("Running observations selection")
    analysis.get_observations()

    log.info(f"Running data reduction")
    # TODO: apply the safe mask (run by run). For the 1d analysis, use method="edisp-bias, bias_percent=10.
    #   for the 3d analysis, use method=["edisp-bias", bkg-peak] and bias_percent=10
    analysis.get_datasets()

    log.info(f"Setting the model")
    txt = Path("model_config.yaml").read_text()
    txt = txt.format_map(target_dict)
    analysis.set_models(txt)
    if method == "3d" and target_dict["spatial_model"] == "DiskSpatialModel":
        analysis.models[0].spatial_model.e.frozen = False
        analysis.models[0].spatial_model.phi.frozen = False

    log.info(f"Running fit ...")
    analysis.run_fit()

    # TODO: set covariance automatically
    model = analysis.models[0].spectral_model
    results_joint = analysis.fit_result
    model.parameters.covariance = results_joint.parameters.get_subcovariance(
        model.parameters
    )
    log.info(f"Writing {path_res}")
    write_fit_summary(
        model.parameters, str(path_res / f"results-summary-fit-{method}.yaml")
    )

    log.info(f"Running flux points estimation")
    # TODO:  For the 3D analysis, re-optimize the background norm in each energy
    #  bin. For now, this is not possible from the HLI.
    analysis.get_flux_points(source=tag)
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
