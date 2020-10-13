import logging
import multiprocessing
import warnings
from itertools import repeat
from pathlib import Path

import astropy.units as u
import click
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from regions import CircleSkyRegion
from scipy.stats import norm

from gammapy.data import GTI, EventList, Observation
from gammapy.datasets import MapDataset, MapDatasetEventSampler
from gammapy.estimators import ExcessMapEstimator
from gammapy.irf import EnergyDispersion2D, load_cta_irfs
from gammapy.makers import MapDatasetMaker
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import Models
from gammapy.utils.table import table_from_row_data

log = logging.getLogger(__name__)

# path config
BASE_PATH = Path(__file__).parent

AVAILABLE_MODELS = [
    "point-pwl",
    "point-ecpl",
    "point-log-parabola",
    "point-pwl2",
    "point-ecpl-3fgl",
    "point-ecpl-4fgl",
    "point-template",
    "diffuse-cube",
    "disk-pwl",
    "gauss-pwl",
]

DPI = 120

# observation config
IRF_FILE = "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
# IRF_FILE = "$GAMMAPY_DATA/cta-prod3b/caldb/data/cta/prod3b-v2/bcf/South_z20_50h/irf_file.fits"

POINTING = SkyCoord(0.0, 0.5, frame="galactic", unit="deg")
LIVETIME = 1 * u.hr
GTI_TABLE = GTI.create(start=0 * u.s, stop=LIVETIME.to(u.s))

# dataset config
ENERGY_AXIS = MapAxis.from_energy_bounds("0.1 TeV", "100 TeV", nbin=10, per_decade=True)
ENERGY_AXIS_TRUE = MapAxis.from_energy_bounds(
    "0.03 TeV", "300 TeV", nbin=20, per_decade=True, name="energy_true"
)
MIGRA_AXIS = MapAxis.from_bounds(0.5, 2, nbin=150, node_type="edges", name="migra")

WCS_GEOM = WcsGeom.create(
    skydir=POINTING, width=(8, 8), binsz=0.02, frame="galactic", axes=[ENERGY_AXIS]
)


def get_filename_dataset(livetime):
    filename = f"data/dataset_{livetime.value:.0f}{livetime.unit}.fits.gz"
    return BASE_PATH / filename


def get_filename_events(filename_dataset, filename_model, obs_id):
    obs_id = int(obs_id)
    model_str = filename_model.name.replace(filename_model.suffix, "")
    filename_events = filename_dataset.name.replace("dataset", "events")
    filename_events = BASE_PATH / f"data/models/{model_str}/" / filename_events
    filename_events = filename_events.name.replace(".fits.gz", f"_{obs_id:04d}.fits.gz")
    return BASE_PATH / f"data/models/{model_str}/" / filename_events


def get_filename_best_fit_model(filename_model, obs_id, livetime):
    obs_id = int(obs_id)
    model_str = filename_model.name.replace(filename_model.suffix, "")

    path = (
        BASE_PATH
        / f"results/models/{model_str}/fit_{livetime.value:.0f}{livetime.unit}/covariance"
    )
    path.mkdir(exist_ok=True, parents=True)
    path = (
        BASE_PATH
        / f"results/models/{model_str}/plots_{livetime.value:.0f}{livetime.unit}"
    )
    path.mkdir(exist_ok=True, parents=True)

    filename = f"results/models/{model_str}/fit_{livetime.value:.0f}{livetime.unit}/best-fit-model_{obs_id:04d}.yaml"
    return BASE_PATH / filename


def get_filename_covariance(filename_best_fit_model):
    filename = filename_best_fit_model.name
    # filename = filename.replace("best-fit-model", "covariance")
    filename = filename.replace(".yaml", "_covariance.dat")
    # return filename_best_fit_model.parent / "covariance" / filename
    return filename_best_fit_model.parent / filename


@click.group()
@click.option(
    "--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING"])
)
@click.option("--show-warnings", is_flag=True, help="Show warnings?")
def cli(log_level, show_warnings):
    logging.basicConfig(level=log_level)
    if not show_warnings:
        warnings.simplefilter("ignore")


@cli.command("all", help="Run all steps")
@click.argument("model", type=click.Choice(list(AVAILABLE_MODELS) + ["all-models"]))
@click.option(
    "--obs_ids", default=1, nargs=1, help="Select a single observation", type=int
)
@click.option(
    "--obs_all",
    default=False,
    nargs=1,
    help="Iterate over all observations",
    is_flag=True,
)
@click.option(
    "--simple",
    default=False,
    nargs=1,
    help="Simplify the dataset preparation",
    type=str,
)
@click.option("--core", default=4, nargs=1, help="Number of cores to be used", type=int)
def all_cmd(model, obs_ids, obs_all, simple, core):
    models = AVAILABLE_MODELS if model == "all-models" else [model]
    log.info(models)
    binned = False
    filename_dataset = get_filename_dataset(LIVETIME)

    log.info(f"Preparing datasets")
    if simple:
        filename_dataset = Path(
            str(filename_dataset).replace("dataset", "dataset_simple")
        )
        prepare_dataset_simple(filename_dataset)
    else:
        prepare_dataset(filename_dataset)

    for model in models:
        log.info(f"Simulating events with model {model}")
        filename_model = BASE_PATH / f"models/{model}.yaml"
        simulate_events(
            filename_model=filename_model,
            filename_dataset=filename_dataset,
            nobs=obs_ids,
        )
        if obs_all:
            obs_ids = f"0:{obs_ids}"
            obs_ids = parse_obs_ids(obs_ids, model)
            with multiprocessing.Pool(processes=core) as pool:
                args = zip(
                    repeat(filename_model),
                    repeat(filename_dataset),
                    obs_ids,
                    repeat(binned),
                    repeat(simple),
                )
                pool.starmap(fit_model, args)

            fit_gather(model, LIVETIME)
            plot_pull_distribution(model, LIVETIME)
        else:
            fit_model(
                filename_model=filename_model,
                filename_dataset=filename_dataset,
                obs_id=obs_ids - 1,
                binned=binned,
                simple=simple,
            )
            plot_results(
                filename_model=filename_model,
                filename_dataset=filename_dataset,
                obs_id=obs_ids - 1,
            )


@cli.command("prepare-dataset", help="Prepare map dataset used for event simulation")
def prepare_dataset_cmd():
    filename_dataset = get_filename_dataset(LIVETIME)
    prepare_dataset(filename_dataset)


def prepare_dataset(filename_dataset):
    """Prepare dataset for a given skymodel."""
    log.info(f"Reading {IRF_FILE}")
    irfs = load_cta_irfs(IRF_FILE)
    observation = Observation.create(
        obs_id=1001, pointing=POINTING, livetime=LIVETIME, irfs=irfs
    )

    empty = MapDataset.create(
        WCS_GEOM, energy_axis_true=ENERGY_AXIS_TRUE, migra_axis=MIGRA_AXIS
    )
    maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
    dataset = maker.run(empty, observation)

    filename_dataset.parent.mkdir(exist_ok=True, parents=True)
    log.info(f"Writing {filename_dataset}")
    dataset.write(filename_dataset, overwrite=True)


def prepare_dataset_simple(filename_dataset):
    """Prepare dataset for a given skymodel."""
    log.info(f"Reading {IRF_FILE}")

    irfs = load_cta_irfs(IRF_FILE)

    edisp_gauss = EnergyDispersion2D.from_gauss(
        e_true=ENERGY_AXIS_TRUE.edges,
        migra=MIGRA_AXIS.edges,
        sigma=0.1,
        bias=0,
        offset=[0, 2, 4, 6, 8] * u.deg,
    )

    irfs["edisp"] = edisp_gauss
    # irfs["aeff"].data.data = np.ones_like(irfs["aeff"].data.data) * 1e6

    observation = Observation.create(
        obs_id=1001, pointing=POINTING, livetime=LIVETIME, irfs=irfs
    )

    empty = MapDataset.create(
        WCS_GEOM, energy_axis_true=ENERGY_AXIS_TRUE, migra_axis=MIGRA_AXIS
    )
    # maker = MapDatasetMaker(selection=["exposure", "edisp"])
    # maker = MapDatasetMaker(selection=["exposure", "edisp", "background"])
    maker = MapDatasetMaker(selection=["exposure", "edisp", "psf", "background"])
    dataset = maker.run(empty, observation)

    filename_dataset.parent.mkdir(exist_ok=True, parents=True)
    log.info(f"Writing {filename_dataset}")
    dataset.write(filename_dataset, overwrite=True)


@cli.command("simulate-events", help="Simulate events for given model and livetime")
@click.argument("model", type=click.Choice(list(AVAILABLE_MODELS) + ["all-models"]))
@click.option("--nobs", default=1, nargs=1, help="How many observations to simulate")
def simulate_events_cmd(model, nobs):
    models = AVAILABLE_MODELS if model == "all-models" else [model]
    filename_dataset = get_filename_dataset(LIVETIME)

    for model in models:
        filename_model = BASE_PATH / f"models/{model}.yaml"
        simulate_events(
            filename_model=filename_model, filename_dataset=filename_dataset, nobs=nobs
        )


def simulate_events(filename_model, filename_dataset, nobs):
    """Simulate events for a given model and dataset.

    Parameters
    ----------
    filename_model : str
        Filename of the model definition.
    filename_dataset : str
        Filename of the dataset to use for simulation.
    nobs : int
        Number of obervations to simulate.
    """
    log.info(f"Reading {IRF_FILE}")
    irfs = load_cta_irfs(IRF_FILE)

    log.info(f"Reading {filename_dataset}")
    dataset = MapDataset.read(filename_dataset)

    log.info(f"Reading {filename_model}")
    models = Models.read(filename_model)
    # dataset.models = models
    dataset.models.extend(models)

    sampler = MapDatasetEventSampler(random_state=0)

    for obs_id in np.arange(nobs):
        observation = Observation.create(
            obs_id=obs_id, pointing=POINTING, livetime=LIVETIME, irfs=irfs
        )

        events = sampler.run(dataset, observation)

        path = get_filename_events(filename_dataset, filename_model, obs_id)
        log.info(f"Writing {path}")
        path.parent.mkdir(exist_ok=True, parents=True)
        events.table.write(str(path), overwrite=True)


def parse_obs_ids(obs_ids_str, model):
    if ":" in obs_ids_str:
        start, stop = obs_ids_str.split(":")
        obs_ids = np.arange(int(start), int(stop))
    elif "," in obs_ids_str:
        obs_ids = [int(_) for _ in obs_ids_str.split(",")]
    elif obs_ids_str == "all":
        n_obs = len(list(BASE_PATH.glob(f"data/models/{model}/events_*.fits.gz")))
        obs_ids = np.arange(n_obs)
    else:
        obs_ids = [int(obs_ids_str)]
    return obs_ids


@cli.command("fit-model", help="Fit given model")
@click.argument("model", type=click.Choice(list(AVAILABLE_MODELS) + ["all-models"]))
@click.option(
    "--obs_ids", default="all", nargs=1, help="Which observation to choose.", type=str
)
@click.option(
    "--binned", default=False, nargs=1, help="Which observation to choose.", type=str
)
@click.option(
    "--simple", default=False, nargs=1, help="Select a single observation", type=str
)
@click.option("--core", default=4, nargs=1, help="Number of cores to be used", type=int)
def fit_model_cmd(model, obs_ids, binned, simple, core):
    models = AVAILABLE_MODELS if model == "all-models" else [model]
    filename_dataset = get_filename_dataset(LIVETIME)

    for model in models:
        obs_ids = parse_obs_ids(obs_ids, model)
        filename_model = BASE_PATH / f"models/{model}.yaml"
        with multiprocessing.Pool(processes=core) as pool:
            args = zip(
                repeat(filename_model),
                repeat(filename_dataset),
                obs_ids,
                repeat(binned),
                repeat(simple),
            )
            pool.starmap(fit_model, args)


def read_dataset(filename_dataset, filename_model, obs_id):
    log.info(f"Reading {filename_dataset}")
    dataset = MapDataset.read(filename_dataset)

    filename_events = get_filename_events(filename_dataset, filename_model, obs_id)
    log.info(f"Reading {filename_events}")
    events = EventList.read(filename_events)

    counts = Map.from_geom(WCS_GEOM)
    counts.fill_events(events)
    dataset.counts = counts
    return dataset


def fit_model(filename_model, filename_dataset, obs_id, binned=False, simple=False):
    """Fit the events using a model.

    Parameters
    ----------
    filename_model : str
        Filename of the model definition.
    filename_dataset : str
        Filename of the dataset to use for simulation.
    obs_id : int
        Observation ID.
    """
    dataset = read_dataset(filename_dataset, filename_model, obs_id)

    log.info(f"Reading {filename_model}")
    models = Models.read(filename_model)

    # dataset.models = models
    dataset.models.extend(models)
    if binned:
        dataset.fake()

    if dataset.background_model:
        dataset.background_model.parameters["norm"].frozen = True

    fit = Fit([dataset])

    result = fit.run(optimize_opts={"print_level": 1})

    log.info(f"Fit info: {result}")

    # write best fit model
    path = get_filename_best_fit_model(filename_model, obs_id, LIVETIME)
    path = path.absolute()
    if binned:
        path = Path(str(path).replace("/fit", "/fit_fake"))
    log.info(f"Writing {path}")
    # write best-fit model and covariance
    dataset.models.write(str(path), overwrite=True)

    # write covariance
    # path = get_filename_covariance(path)
    # if binned:
    #    path = Path(str(path).replace("/fit","/fit_fake"))
    # log.info(f"Writing {path}")

    # TODO: exclude background parameters for now, as they are fixed anyway
    # covariance = result.parameters.get_subcovariance(models.parameters)
    # np.savetxt(path, covariance)


@cli.command("fit-gather", help="Gather fit results from the given model")
@click.argument("model", type=click.Choice(list(AVAILABLE_MODELS) + ["all-models"]))
@click.option(
    "--binned", default=False, nargs=1, help="Which observation to choose.", type=str
)
def fit_gather_cmd(model, binned):
    models = AVAILABLE_MODELS if model == "all-models" else [model]
    for model in models:
        fit_gather(model, LIVETIME, binned)


def fit_gather(model_name, livetime, binned=False):
    rows = []

    path = (
        BASE_PATH
        / f"results/models/{model_name}/fit_{livetime.value:.0f}{livetime.unit}"
    )
    if binned:
        path = Path(str(path).replace("/fit", "/fit_fake"))

    for filename in path.glob("*.yaml"):
        # model_best_fit = read_best_fit_model(filename)
        model_best_fit = Models.read(filename)
        model_best_fit = model_best_fit[model_name]
        row = {}

        for par in model_best_fit.parameters:
            row[par.name] = par.value
            row[par.name + "_err"] = par.error

        rows.append(row)

    table = table_from_row_data(rows)
    name = f"fit-results-all_{livetime.value:.0f}{livetime.unit}"
    if binned:
        name = "fit_binned-results-all"
    filename = f"results/models/{model_name}/{name}.fits.gz"
    log.info(f"Writing {filename}")
    table.write(str(filename), overwrite=True)


@cli.command("plot-results", help="Plot results for given model")
@click.argument("model", type=click.Choice(list(AVAILABLE_MODELS) + ["all-models"]))
@click.option(
    "--obs_ids", default="0", nargs=1, help="Which observation to choose.", type=str
)
def plot_results_cmd(model, obs_ids):
    models = AVAILABLE_MODELS if model == "all-models" else [model]
    filename_dataset = get_filename_dataset(LIVETIME)
    for model in models:
        for obs_id in parse_obs_ids(obs_ids, model):
            filename_model = BASE_PATH / f"models/{model}.yaml"
            plot_results(
                filename_model=filename_model,
                filename_dataset=filename_dataset,
                obs_id=obs_id,
            )


def save_figure(filename):
    path = BASE_PATH / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Writing {path}")
    plt.savefig(path, dpi=DPI)
    plt.clf()
    plt.close()


def plot_spectra(model, model_best_fit, obs_id, livetime):
    """Plot spectral models"""

    if model.tag == "SkyDiffuseCube":
        log.info(f"SkyDiffuseCube: no spectral model to plot")
    else:
        ax = model.spectral_model.plot(
            energy_range=(0.1, 300) * u.TeV, label="Sim. model"
        )
        model_best_fit.spectral_model.plot(
            energy_range=(0.1, 300) * u.TeV, label="Best-fit model", ax=ax,
        )
        model_best_fit.spectral_model.plot_error(energy_range=(0.1, 300) * u.TeV, ax=ax)

        ax.legend()
        obs_id = int(obs_id)
        filename = f"results/models/{model.name}/plots_{livetime.value:.0f}{livetime.unit}/spectra/spectra_{obs_id:04d}.png"
        save_figure(filename)


def plot_residuals(dataset, obs_id, livetime, model_name):
    """Plot residuals"""

    model = dataset.models[model_name]
    if model.tag == "SkyDiffuseCube":
        log.info(f"SkyDiffuseCube: no spectral model to plot")
    else:
        spatial_model = model.spatial_model
        if spatial_model.__class__.__name__ == "PointSpatialModel":
            region = CircleSkyRegion(center=spatial_model.position, radius=0.1 * u.deg)
        else:
            region = spatial_model.to_region()

        dataset.plot_residuals(
            method="diff/sqrt(model)",
            vmin=-0.5,
            vmax=0.5,
            region=region,
            figsize=(10, 4),
        )
        obs_id = int(obs_id)
        filename = f"results/models/{model.name}/plots_{livetime.value:.0f}{livetime.unit}/residuals/residuals_{obs_id:04d}.png"
        save_figure(filename)


def plot_residual_distribution(dataset, obs_id, livetime):
    """Plot residual significance distribution"""
    model = dataset.models[1]

    estimator = ExcessMapEstimator(
        correlation_radius="0.1 deg"
    )

    maps = estimator.run(dataset)
    valid = np.isfinite(maps["ts"].data)
    sig_resid = np.sqrt(maps["ts"].data[valid])

    plt.hist(
        sig_resid, density=True, alpha=0.5, color="red", bins=100,
    )

    mu, std = norm.fit(sig_resid)
    # replace with log.info()
    log.info("Fit results: mu = {:.2f}, std = {:.2f}".format(mu, std))
    x = np.linspace(-8, 8, 50)
    p = norm.pdf(x, mu, std)
    plt.plot(
        x,
        p,
        lw=2,
        color="black",
        label="Fit results: mu = {:.2f}, std = {:.2f}".format(mu, std),
    )
    plt.legend()
    plt.xlabel("Significance")
    plt.yscale("log")
    plt.ylim(1e-5, 1)
    xmin, xmax = np.min(sig_resid), np.max(sig_resid)
    plt.xlim(xmin, xmax)

    obs_id = int(obs_id)
    filename = f"residuals-distribution_{obs_id:04d}.png"
    filepath = f"results/models/{model.name}/plots_{livetime.value:.0f}{livetime.unit}/residuals-distribution/{filename}"
    save_figure(filepath)


# OBSOLETE...
# def read_best_fit_model(filename):
#    log.info(f"Reading {filename}")
#    model_best_fit = Models.read(filename)
#
#    path = get_filename_covariance(filename)
#    log.info(f"Reading {path}")
#    pars = model_best_fit.parameters
#    pars.covariance = np.loadtxt(str(path))
#
#    if model_best_fit[1].tag  == 'SkyDiffuseCube':
#        spectral_model_best_fit = model_best_fit[1]
#        covar = pars.get_subcovariance(spectral_model_best_fit.parameters)
#        spectral_model_best_fit.parameters.covariance = covar
#
#       # spatial_model_best_fit = model_best_fit[0].spatial_model
#       # covar = pars.get_subcovariance(spatial_model_best_fit.parameters)
#       # spatial_model_best_fit.parameters.covariance = covar
#
#    else:
#        spectral_model_best_fit = model_best_fit[1].spectral_model
#        covar = pars.get_subcovariance(spectral_model_best_fit.parameters)
#        spectral_model_best_fit.parameters.covariance = covar
#
#        spatial_model_best_fit = model_best_fit[1].spatial_model
#        covar = pars.get_subcovariance(spatial_model_best_fit.parameters)
#        spatial_model_best_fit.parameters.covariance = covar
#
#    return model_best_fit
#


def plot_results(filename_model, obs_id, filename_dataset=None):
    """Plot the best-fit spectrum, the residual map and the residual significance distribution.

    Parameters
    ----------
    filename_model : str
        Filename of the model definition.
    filename_dataset : str
        Filename of the dataset.
    obs_id : int
        Observation ID.
    """
    log.info(f"Reading {filename_model}")
    model = Models.read(filename_model)

    path = get_filename_best_fit_model(filename_model, obs_id, LIVETIME)
    # model_best_fit = read_best_fit_model(path)
    model_best_fit = Models.read(path)

    plot_spectra(
        model[model.names[0]], model_best_fit[model.names[0]], obs_id, LIVETIME
    )

    dataset = read_dataset(filename_dataset, filename_model, obs_id)
    mod = Models(model_best_fit[model.names[0]])
    dataset.models.extend(mod)
    plot_residuals(dataset, obs_id, LIVETIME, model.names[0])
    plot_residual_distribution(dataset, obs_id, LIVETIME)


@cli.command(
    "plot-pull-distributions", help="Plot pull distributions for the given model"
)
@click.argument("model", type=click.Choice(list(AVAILABLE_MODELS) + ["all-models"]))
@click.option(
    "--binned", default=False, nargs=1, help="Which observation to choose.", type=str
)
def plot_pull_distribution_cmd(model, binned):
    models = AVAILABLE_MODELS if model == "all-models" else [model]
    for model in models:
        plot_pull_distribution(model_name=model, livetime=LIVETIME, binned=binned)


def plot_pull_distribution(model_name, livetime, binned=False):
    name = f"fit-results-all_{livetime.value:.0f}{livetime.unit}"
    if binned:
        name = "fit_binned-results-all"
    filename = BASE_PATH / f"results/models/{model_name}/{name}.fits.gz"
    results = Table.read(str(filename))

    filename_ref = BASE_PATH / f"models/{model_name}.yaml"
    model_ref = Models.read(filename_ref)[0]
    names = [name for name in results.colnames if "err" not in name]

    plots = f"plots_{livetime.value:.0f}{livetime.unit}"
    if binned:
        plots = "plots_fake"
    for name in names:
        # TODO: report mean and stdev here as well
        values = results[name]
        values_err = results[name + "_err"]
        par = model_ref.parameters[name]

        if par.frozen:
            log.info(f"Skipping frozen parameter: {name}")
            continue

        pull = (values - par.value) / values_err

        # print("Number of fits beyond 5 sigmas: ",(np.where( (pull<-5) )))
        plt.hist(pull, bins=21, normed=True, range=(-5, 5))
        plt.xlim(-5, 5)
        plt.xlabel("(value - value_true) / error")
        plt.ylabel("PDF")
        plt.title(f"Pull distribution for {model_name}: {name} ")
        filename = f"results/models/{model_name}/{plots}/pull-distribution-{name}.png"
        save_figure(filename)


if __name__ == "__main__":
    cli()
