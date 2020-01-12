# simulate bright sources
from pathlib import Path
import logging
import warnings
import click

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.cube import (
    MapDataset,
    MapDatasetEventSampler,
    MapDatasetMaker,
    SafeMaskMaker,
)
from gammapy.data import GTI, Observation, EventList
from gammapy.maps import MapAxis, WcsGeom, WcsNDMap, Map
from gammapy.irf import load_cta_irfs
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    PointSpatialModel,
    SkyModel,
    SkyModels,
)
from regions import CircleSkyRegion

log = logging.getLogger(__name__)

AVAILABLE_MODELS = ["point-pwl", "point-ecpow", "point-logparabola",
                    "point-pwltwo", "point-ecpow3fgl", "point-excpow4fgl",
                    "point-compoundmod",
                    "disk-pwl", "gauss-pwl"]
DPI = 300

# observation config
IRF_FILE = "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
POINTING = SkyCoord(0.0, 0.0, frame="galactic", unit="deg")
LIVETIME = 10 * u.hr
GTI_TABLE = GTI.create(start=0 * u.s, stop=LIVETIME.to(u.s))
OBS_ID = '{:04d}'.format(1)

# dataset config
ENERGY_AXIS = MapAxis.from_energy_bounds("0.1 TeV", "100 TeV", nbin=30)
ENERGY_AXIS_TRUE = MapAxis.from_energy_bounds("0.3 TeV", "300 TeV", nbin=30)
WCS_GEOM = WcsGeom.create(
    skydir=POINTING, width=(8, 8), binsz=0.02, coordsys="GAL", axes=[ENERGY_AXIS]
)

# path config
BASE_PATH = Path(__file__).parent


def get_filename_dataset(livetime):
    filename = f"data/dataset_{livetime.value:.0f}{livetime.unit}.fits.gz"
    return BASE_PATH / filename


def get_filename_events(filename_dataset, filename_model, obs_id=OBS_ID):
    model_str = filename_model.name.replace(filename_model.suffix, "")
    filename_events = filename_dataset.name.replace("dataset", "events")
    filename_events = BASE_PATH / f"data/models/{model_str}/" / filename_events
    filename_events = filename_events.name.replace(".fits.gz", f"_{obs_id}.fits.gz")
    path = BASE_PATH / f"data/models/{model_str}/" / filename_events
    return path


def get_filename_best_fit_model(filename_model, obs_id=OBS_ID):
    model_str = filename_model.name.replace(filename_model.suffix, "")
    filename = f"results/models/{model_str}/best-fit-model_{obs_id}.yaml"
    return BASE_PATH / filename


def get_filename_covariance(filename_model, obs_id=OBS_ID):
    model_str = filename_model.name.replace(filename_model.suffix, "")
    filename = f"results/models/{model_str}/covariance_{obs_id}.txt"
    return str(BASE_PATH / filename)

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
@click.argument("model", type=click.Choice(list(AVAILABLE_MODELS)))
def all_cmd(model):
    if model == "all":
        models = AVAILABLE_MODELS
    else:
        models = [model]

    filename_dataset = get_filename_dataset(LIVETIME)

    prepare_dataset(filename_dataset)

    for model in models:
        filename_model = BASE_PATH / f"models/{model}.yaml"
        simulate_events(filename_model=filename_model, filename_dataset=filename_dataset)
        fit_model(filename_model=filename_model, filename_dataset=filename_dataset)
        plot_results(filename_model=filename_model, filename_dataset=filename_dataset)


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

    empty = MapDataset.create(WCS_GEOM)
    maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
    dataset = maker.run(empty, observation)

    filename_dataset.parent.mkdir(exist_ok=True, parents=True)
    log.info(f"Writing {filename_dataset}")
    dataset.write(filename_dataset, overwrite=True)


@cli.command("simulate-events", help="Simulate events for given model and livetime")
@click.argument("model", type=click.Choice(list(AVAILABLE_MODELS) + ["all"]))
def simulate_events_cmd(model):
    if model == "all":
        models = AVAILABLE_MODELS
    else:
        models = [model]

    filename_dataset = get_filename_dataset(LIVETIME)

    for model in models:
        filename_model = BASE_PATH / f"models/{model}.yaml"
        simulate_events(filename_model=filename_model, filename_dataset=filename_dataset)


def simulate_events(filename_model, filename_dataset, obs_id=int(OBS_ID)):
    """Simulate events for a given model and dataset.

    Parameters
    ----------
    filename_model : str
        Filename of the model definition.
    filename_dataset : str
        Filename of the dataset to use for simulation.
    obs_id : int
        Observation ID.
    """
    log.info(f"Reading {IRF_FILE}")
    irfs = load_cta_irfs(IRF_FILE)
    observation = Observation.create(
        obs_id=obs_id, pointing=POINTING, livetime=LIVETIME, irfs=irfs
    )

    log.info(f"Reading {filename_dataset}")
    dataset = MapDataset.read(filename_dataset)

    log.info(f"Reading {filename_model}")
    models = SkyModels.read(filename_model)
    dataset.models = models

    events = MapDatasetEventSampler(random_state=obs_id)
    events = events.run(dataset, observation)

    path = get_filename_events(filename_dataset, filename_model)
    log.info(f"Writing {path}")
    path.parent.mkdir(exist_ok=True, parents=True)
    events.table.write(str(path), overwrite=True)


@cli.command("fit-model", help="Fit given model")
@click.argument("model", type=click.Choice(list(AVAILABLE_MODELS) + ["all"]))
def fit_model_cmd(model):
    if model == "all":
        models = AVAILABLE_MODELS
    else:
        models = [model]

    filename_dataset = get_filename_dataset(LIVETIME)

    for model in models:
        filename_model = BASE_PATH / f"models/{model}.yaml"
        fit_model(filename_model=filename_model, filename_dataset=filename_dataset)


def read_dataset(filename_dataset, filename_model):
    log.info(f"Reading {filename_dataset}")
    dataset = MapDataset.read(filename_dataset)

    filename_events = get_filename_events(filename_dataset, filename_model)
    log.info(f"Reading {filename_events}")
    events = EventList.read(filename_events)

    counts = Map.from_geom(WCS_GEOM)
    counts.fill_events(events)
    dataset.counts = counts
    return dataset


def fit_model(filename_model, filename_dataset):
    """Fit the events using a model.

    Parameters
    ----------
    filename_model : str
        Filename of the model definition.
    filename_dataset : str
        Filename of the dataset to use for simulation.
    """
    dataset = read_dataset(filename_dataset, filename_model)

    log.info(f"Reading {filename_model}")
    models = SkyModels.read(filename_model)

    dataset.models = models
    dataset.background_model.parameters["norm"].frozen = True

    fit = Fit([dataset])
    result = fit.run(optimize_opts={"print_level": 1})

    log.info(f"Fit info: {result}")

    # write best fit model
    path = get_filename_best_fit_model(filename_model)
    log.info(f"Writing {path}")
    models.write(str(path), overwrite=True)

    # write covariance
    path = get_filename_covariance(filename_model)
    log.info(f"Writing {path}")

    # TODO: exclude background parameters for now, as they are fixed anyway
    covariance = result.parameters.get_subcovariance(models.parameters)
    np.savetxt(path, covariance)


@cli.command("plot-results", help="Plot results for given model")
@click.argument("model", type=click.Choice(list(AVAILABLE_MODELS) + ["all"]))
def plot_results_cmd(model):
    if model == "all":
        models = AVAILABLE_MODELS
    else:
        models = [model]

    filename_dataset = get_filename_dataset(LIVETIME)

    for model in models:
        filename_model = BASE_PATH / f"models/{model}.yaml"
        plot_results(filename_model=filename_model, filename_dataset=filename_dataset)


def save_figure(filename):
    path = BASE_PATH / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Writing {path}")
    plt.savefig(path, dpi=DPI)
    plt.clf()


def plot_spectra(model, model_best_fit):
    """Plot spectral models"""
    # plot spectral models
    ax = model.spectral_model.plot(
        energy_range=(0.1, 300) * u.TeV, label="Sim. model"
    )
    model_best_fit.spectral_model.plot(
        energy_range=(0.1, 300) * u.TeV, label="Best-fit model", ax=ax,
    )
    model_best_fit.spectral_model.plot_error(energy_range=(0.1, 300) * u.TeV, ax=ax)
    ax.legend()

    filename = f"results/models/{model.name}/plots/spectra_{OBS_ID}.png"
    save_figure(filename)


def plot_residuals(dataset):
    # plot residuals
    model = dataset.models[0]
    spatial_model = model.spatial_model
    if spatial_model.__class__.__name__ == "PointSpatialModel":
        region = CircleSkyRegion(center=spatial_model.position, radius=0.1 * u.deg)
    else:
        region = spatial_model.to_region()

    dataset.plot_residuals(method="diff/sqrt(model)", vmin=-0.5, vmax=0.5, region=region, figsize=(10, 4))
    filename = f"results/models/{model.name}/plots/residuals_{OBS_ID}.png"
    save_figure(filename)


def plot_residual_distribution(dataset):
    # plot residual significance distribution
    model = dataset.models[0]
    resid = dataset.residuals()
    sig_resid = resid.data[np.isfinite(resid.data)]

    plt.hist(
        sig_resid, density=True, alpha=0.5, color="red", bins=100,
    )

    mu, std = norm.fit(sig_resid)
    # replace with log.info()
    print("Fit results: mu = {:.2f}, std = {:.2f}".format(mu, std))
    x = np.linspace(-8, 8, 50)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, lw=2, color="black")
    plt.legend()
    plt.xlabel("Significance")
    plt.yscale("log")
    plt.ylim(1e-5, 1)
    xmin, xmax = np.min(sig_resid), np.max(sig_resid)
    plt.xlim(xmin, xmax)

    filename = f"results/models/{model.name}/plots/residuals-distribution_{OBS_ID}.png"
    save_figure(filename)


def read_best_fit_model(path, obs_id=OBS_ID):
    log.info(f"Reading {path}")
    model_best_fit = SkyModels.read(path)

    path = path.parent / f"covariance_{obs_id}.txt"
    log.info(f"Reading {path}")
    pars = model_best_fit.parameters
    pars.covariance = np.loadtxt(str(path))

    spectral_model_best_fit = model_best_fit[0].spectral_model
    covar = pars.get_subcovariance(spectral_model_best_fit.parameters)
    spectral_model_best_fit.parameters.covariance = covar
    return model_best_fit


def plot_results(filename_model, filename_dataset=None):
    """Plot the best-fit spectrum, the residual map and the residual significance distribution.

    Parameters
    ----------
    filename_model : str
        Filename of the model definition.
    filename_dataset : str
        Filename of the dataset.
    """
    log.info(f"Reading {filename_model}")
    model = SkyModels.read(filename_model)

    path = get_filename_best_fit_model(filename_model)
    model_best_fit = read_best_fit_model(path)

    plot_spectra(model[0], model_best_fit[0])

    dataset = read_dataset(filename_dataset, filename_model)
    dataset.models = model_best_fit
    plot_residuals(dataset)
    plot_residual_distribution(dataset)


if __name__ == "__main__":
    cli()
