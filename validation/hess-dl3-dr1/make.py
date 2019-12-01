import logging
import yaml
import click
import warnings
from pathlib import Path
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.maps import MapAxis
from gammapy.modeling import Fit
from gammapy.data import DataStore
from gammapy.modeling import Model
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from gammapy.cube import SafeMaskMaker
from gammapy.spectrum import (
    SpectrumDatasetMaker,
    FluxPointsEstimator,
    ReflectedRegionsBackgroundMaker,
)

log = logging.getLogger(__name__)

AVAILABLE_TARGETS = ["crab", "pks2155", "msh1552"]


@click.group()
@click.option(
    "--log-level",
    default="info",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
)
@click.option("--show-warnings", is_flag=True, help="Show warnings?")
def cli(log_level, show_warnings):
    """
    Run validation of DL3 data analysis.
    """
    levels = dict(
        debug=logging.DEBUG,
        info=logging.INFO,
        warning=logging.WARNING,
        error=logging.ERROR,
        critical=logging.CRITICAL,
    )
    logging.basicConfig(level=levels[log_level])
    log.setLevel(level=levels[log_level])

    if not show_warnings:
        warnings.simplefilter("ignore")


@cli.command("run-analyses", help="Run DL3 analysis validation")
@click.option(
    "--debug",
    is_flag=True,
    help="If True, runs faster. In the 1D joint analysis, analyzes only 1 run."
    "For both 1d and 3d, complutes only 1 flux point and does not re-optimize the bkg",
)
@click.argument("targets", type=click.Choice(list(AVAILABLE_TARGETS) + ["all"]))
def run_analyses(debug, targets):
    if targets == "all":
        targets = list(AVAILABLE_TARGETS)
    else:
        targets = [targets]

    e_reco = MapAxis.from_bounds(
        0.1, 100, nbin=24, unit="TeV", name="energy", interp="log"
    ).edges
    nbin = 12 if debug is False else 1
    fluxp_edges = MapAxis.from_bounds(
        0.1, 100, nbin=nbin, unit="TeV", name="energy", interp="log"
    ).edges

    with open("targets.yaml", "r") as stream:
        targets_file = yaml.safe_load(stream)

    for target in targets:
        # read config for the target
        target_filter = filter(lambda _: _["tag"] == target, targets_file)
        target_dict = list(target_filter)[0]

        log.info(f"Processing source: {target}")
        run_analysis_1d(target_dict, e_reco, fluxp_edges, debug)
        run_analysis_3d(target_dict, fluxp_edges, debug)


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


def run_analysis_1d(target_dict, e_reco, fluxp_edges, debug):
    """Run joint spectral analysis for the selected target"""
    tag = target_dict["tag"]
    name = target_dict["name"]

    log.info(f"Running 1d analysis, {tag}")
    path_res = Path(tag + "/results/")

    ra = target_dict["ra"]
    dec = target_dict["dec"]
    on_size = target_dict["on_size"]
    e_decorr = target_dict["e_decorr"]

    target_pos = SkyCoord(ra, dec, frame="icrs")
    on_radius = Angle(on_size)

    log.info(f"Running observations selection")
    data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
    mask = data_store.obs_table["TARGET_NAME"] == name
    obs_table = data_store.obs_table[mask]
    observations = data_store.get_observations(obs_table["OBS_ID"])

    if debug is True:
        observations = [observations[0]]
    log.info(f"Running data reduction")
    # Reflected regions background estimation
    on_region = CircleSkyRegion(center=target_pos, radius=on_radius)
    dataset_maker = SpectrumDatasetMaker(
        region=on_region,
        e_reco=e_reco,
        e_true=e_reco,
        containment_correction=True,
    )
    bkg_maker = ReflectedRegionsBackgroundMaker()
    safe_mask_masker = SafeMaskMaker(methods=["edisp-bias"], bias_percent=10)

    datasets = []

    for observation in observations:
        dataset = dataset_maker.run(observation, selection=["counts", "aeff", "edisp"])
        dataset_on_off = bkg_maker.run(dataset, observation)
        dataset_on_off = safe_mask_masker.run(dataset_on_off, observation)
        datasets.append(dataset_on_off)

    log.info(f"Running fit ...")
    model = PowerLawSpectralModel(
        index=2, amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=e_decorr
    )
    for dataset in datasets:
        dataset.model = SkyModel(spectral_model=model)

    fit_joint = Fit(datasets)
    result_joint = fit_joint.run()
    parameters = model.parameters
    parameters.covariance = result_joint.parameters.covariance
    log.info(f"Writing {path_res}")
    write_fit_summary(parameters, str(path_res / "results-summary-fit-1d.yaml"))

    log.info(f"Running flux points estimation")
    fpe = FluxPointsEstimator(datasets=datasets, e_edges=fluxp_edges)
    flux_points = fpe.run()
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
        path_res / "flux-points-1d.ecsv", format="ascii.ecsv"
    )


def run_analysis_3d(target_dict, fluxp_edges, debug):
    """Run stacked 3D analysis for the selected target.

    Notice that, for the sake of time saving, we run a stacked analysis, as opposed
     to the joint analysis that is performed in the reference paper.
    """
    tag = target_dict["tag"]
    log.info(f"running 3d analysis, {tag}")

    path_res = Path(tag + "/results/")

    txt = Path("config_template.yaml").read_text()
    txt = txt.format_map(target_dict)
    config = AnalysisConfig.from_yaml(txt)

    log.info(f"Running observations selection")
    analysis = Analysis(config)
    analysis.get_observations()

    log.info(f"Running data reduction")
    analysis.get_datasets()


    # TODO: Improve safe mask handling in Analysis. the mask should be applied run-by-run
    maker_safe_mask = SafeMaskMaker(methods=["edisp-bias", "bkg-peak"])
    stacked = maker_safe_mask.run(analysis.datasets[0])

    log.info(f"Running fit ...")
    ra = target_dict["ra"]
    dec = target_dict["dec"]
    e_decorr = target_dict["e_decorr"]
    spectral_model = Model.create("PowerLawSpectralModel", reference=e_decorr)
    spatial_model = Model.create(
        target_dict["spatial_model"], lon_0=ra, lat_0=dec
    )
    if target_dict["spatial_model"] == "DiskSpatialModel":
        spatial_model.e.frozen = False
    sky_model = SkyModel(
        spatial_model=spatial_model, spectral_model=spectral_model, name=tag
    )

    stacked.model = sky_model
    stacked.background_model.norm.frozen = False
    fit = Fit([stacked])
    result = fit.run()

    parameters = stacked.model.parameters
    model_npars = len(sky_model.parameters.names)
    parameters.covariance = result.parameters.covariance[
        0:model_npars, 0:model_npars
    ]
    log.info(f"Writing {path_res}")
    write_fit_summary(parameters, str(path_res / "results-summary-fit-3d.yaml"))

    log.info("Running flux points estimation")
    # TODO: This is a workaround to re-optimize the bkg. Remove it once it's added to the Analysis class
    for par in stacked.parameters:
        if par is not stacked.background_model.norm:
            par.frozen = True

    reoptimize = True if debug is False else False
    fpe = FluxPointsEstimator(
        datasets=[stacked], e_edges=fluxp_edges, source=tag, reoptimize=reoptimize
    )

    flux_points = fpe.run()
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
        path_res / "flux-points-3d.ecsv", format="ascii.ecsv"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
