# simulate bright sources
from pathlib import Path
import logging
import warnings
import click
import multiprocessing
from itertools import repeat

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.convolution import Tophat2DKernel
from astropy.coordinates import SkyCoord
from astropy.table import Table
from gammapy.cube import (
    MapDataset,
    MapDatasetEventSampler,
    MapDatasetMaker,
)
from gammapy.data import GTI, Observation, EventList
from gammapy.detect import compute_lima_image as lima
from gammapy.maps import MapAxis, WcsGeom, Map
from gammapy.irf import EnergyDispersion2D, load_cta_irfs
from gammapy.modeling import Fit
from gammapy.modeling.models import Models
from gammapy.utils.table import table_from_row_data
from regions import CircleSkyRegion

log = logging.getLogger(__name__)

# path config
BASE_PATH = Path(__file__).parent

AVAILABLE_MODELS = ["point-pwl", "point-ecpl", "point-log-parabola",
                    "point-pwl2", "point-ecpl-3fgl", "point-ecpl-4fgl",
                    "point-template", "diffuse-cube",
                    "disk-pwl", "gauss-pwl", "gauss-pwlsimple"]

DPI = 120

# observation config
IRF_FILE = "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"

POINTING = SkyCoord(0.0, 0.5, frame="galactic", unit="deg")
LIVETIME = 10 * u.hr
GTI_TABLE = GTI.create(start=0 * u.s, stop=LIVETIME.to(u.s))

# dataset config
ENERGY_AXIS = MapAxis.from_energy_bounds("0.1 TeV", "100 TeV", nbin=10, per_decade=True)
ENERGY_AXIS_TRUE = MapAxis.from_energy_bounds("0.03 TeV", "300 TeV", nbin=20, per_decade=True)
MIGRA_AXIS = MapAxis.from_bounds(0.5, 2, nbin=150, node_type="edges", name="migra")

WCS_GEOM = WcsGeom.create(
    skydir=POINTING, width=(4, 4), binsz=0.02, frame="galactic", axes=[ENERGY_AXIS]
)


def get_filename_dataset(livetime):
    filename = f"data/dataset_{livetime.value:.0f}{livetime.unit}.fits.gz"
    return BASE_PATH / filename


def get_filename_events(filename_dataset, filename_model, obs_id):
    obs_id=int(obs_id)
    model_str = filename_model.name.replace(filename_model.suffix, "")
    filename_events = filename_dataset.name.replace("dataset", "events")
    filename_events = BASE_PATH / f"data/models/{model_str}/" / filename_events
    filename_events = filename_events.name.replace(".fits.gz", f"_{obs_id:04d}.fits.gz")
    path = BASE_PATH / f"data/models/{model_str}/" / filename_events
    return path


def get_filename_best_fit_model(filename_model, obs_id):
    obs_id=int(obs_id)
    model_str = filename_model.name.replace(filename_model.suffix, "")
    filename = f"results/models/{model_str}/fit/best-fit-model_{obs_id:04d}.yaml"
    return BASE_PATH / filename


def get_filename_covariance(filename_best_fit_model):
    filename = filename_best_fit_model.name
    filename = filename.replace("best-fit-model", "covariance")
    filename = filename.replace(".yaml", ".txt")
    return filename_best_fit_model.parent / "covariance" / filename


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
@click.option(
              "--obs_ids", default=1, nargs=1, help="Select a single observation", type=int
              )
@click.option(
              "--obs_all", default=False, nargs=1, help="Iterate over all observations", is_flag=True
              )
@click.option(
              "--simple", default=False, nargs=1, help="Simplify the dataset preparation", type=str
              )
def all_cmd(model, obs_ids, obs_all, simple):
    if model == "all":
        models = AVAILABLE_MODELS
    else:
        models = [model]

    binned = False
    filename_dataset = get_filename_dataset(LIVETIME)
    filename_model = BASE_PATH / f"models/{model}.yaml"

    if simple:
        filename_dataset = Path(str(filename_dataset).replace("dataset","dataset_simple"))
        prepare_dataset_simple(filename_dataset)

    else:
        prepare_dataset(filename_dataset)

    if obs_all:
        for model in models:
            simulate_events(filename_model=filename_model, filename_dataset=filename_dataset, nobs=obs_ids)
            obs_ids = f"0:{obs_ids}"
            obs_ids = parse_obs_ids(obs_ids, model)
            with multiprocessing.Pool(processes=4) as pool:
                args = zip(repeat(filename_model), repeat(filename_dataset), obs_ids, repeat(binned), repeat(simple))
                results = pool.starmap(fit_model, args)

            fit_gather(model)
            plot_pull_distribution(model)
    else:
        for model in models:
            simulate_events(filename_model=filename_model, filename_dataset=filename_dataset, nobs=obs_ids)
            fit_model(filename_model=filename_model, filename_dataset=filename_dataset, obs_id=str(obs_ids-1), binned=binned, simple=simple)
            plot_results(filename_model=filename_model, filename_dataset=filename_dataset, obs_id=str(obs_ids-1))


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

    empty = MapDataset.create(WCS_GEOM, energy_axis_true=ENERGY_AXIS_TRUE, migra_axis=MIGRA_AXIS)
    maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
    dataset = maker.run(empty, observation)

    filename_dataset.parent.mkdir(exist_ok=True, parents=True)
    log.info(f"Writing {filename_dataset}")
    dataset.write(filename_dataset, overwrite=True)


def prepare_dataset_simple(filename_dataset):
    """Prepare dataset for a given skymodel."""
    log.info(f"Reading {IRF_FILE}")

    irfs = load_cta_irfs(IRF_FILE)

    edisp_gauss = EnergyDispersion2D.from_gauss(e_true=ENERGY_AXIS_TRUE.edges,
                                            migra=MIGRA_AXIS.edges,
                                            sigma=0.1, bias=0,
                                            offset=[0, 2, 4, 6, 8] * u.deg)

    irfs["edisp"] = edisp_gauss
    irfs["aeff"].data.data = np.ones_like(irfs["aeff"].data.data) * 1e6

    observation = Observation.create(
                                     obs_id=1001, pointing=POINTING, livetime=LIVETIME, irfs=irfs
                                     )

    empty = MapDataset.create(WCS_GEOM, energy_axis_true=ENERGY_AXIS_TRUE, migra_axis=MIGRA_AXIS)
    maker = MapDatasetMaker(selection=["exposure", "edisp"])
    dataset = maker.run(empty, observation)

    filename_dataset.parent.mkdir(exist_ok=True, parents=True)
    log.info(f"Writing {filename_dataset}")
    dataset.write(filename_dataset, overwrite=True)


@cli.command("simulate-events", help="Simulate events for given model and livetime")
@click.argument("model", type=click.Choice(list(AVAILABLE_MODELS) + ["all"]))
@click.option(
              "--nobs", default=1, nargs=1, help="How many observations to simulate"
              )
def simulate_events_cmd(model, nobs):
    if model == "all":
        models = AVAILABLE_MODELS
    else:
        models = [model]

    filename_dataset = get_filename_dataset(LIVETIME)

    for model in models:
        filename_model = BASE_PATH / f"models/{model}.yaml"
        simulate_events(filename_model=filename_model, filename_dataset=filename_dataset, nobs=nobs)


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
    dataset.models = models

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
@click.argument("model", type=click.Choice(list(AVAILABLE_MODELS) + ["all"]))
@click.option(
              "--obs_ids", default="all", nargs=1, help="Which observation to choose.", type=str
              )
@click.option(
              "--binned", default=False, nargs=1, help="Which observation to choose.", type=str
              )
@click.option(
              "--simple", default=False, nargs=1, help="Select a single observation", type=str
              )
def fit_model_cmd(model, obs_ids, binned, simple):
    if model == "all":
        models = AVAILABLE_MODELS
    else:
        models = [model]

    filename_dataset = get_filename_dataset(LIVETIME)

    for model in models:
        obs_ids = parse_obs_ids(obs_ids, model)
        filename_model = BASE_PATH / f"models/{model}.yaml"
        with multiprocessing.Pool(processes=4) as pool:
            args = zip(repeat(filename_model), repeat(filename_dataset), obs_ids, repeat(binned), repeat(simple))
            results = pool.starmap(fit_model, args)


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

    dataset.models = models
    if binned:
        dataset.fake()
    
    if dataset.background_model:
        dataset.background_model.parameters["norm"].frozen = True

    fit = Fit([dataset])
    
    result = fit.run(optimize_opts={"print_level": 1})

    log.info(f"Fit info: {result}")

    # write best fit model
    path = get_filename_best_fit_model(filename_model, obs_id)
    if binned:
        path = Path(str(path).replace("/fit/","/fit_fake/"))
    log.info(f"Writing {path}")
    models.write(str(path), overwrite=True)

    # write covariance
    path = get_filename_covariance(path)
    if binned:
        path = Path(str(path).replace("/fit/","/fit_fake/"))
    log.info(f"Writing {path}")

    # TODO: exclude background parameters for now, as they are fixed anyway
    covariance = result.parameters.get_subcovariance(models.parameters)
    np.savetxt(path, covariance)


@cli.command("fit-gather", help="Gather fit results from the given model")
@click.argument("model", type=click.Choice(list(AVAILABLE_MODELS) + ["all"]))
@click.option(
              "--binned", default=False, nargs=1, help="Which observation to choose.", type=str
              )
def fit_gather_cmd(model, binned):
    if model == "all":
        models = AVAILABLE_MODELS
    else:
        models = [model]

    for model in models:
        fit_gather(model, binned)


def fit_gather(model_name, binned=False):
    rows = []

    path = (BASE_PATH / f"results/models/{model_name}/fit")
    if binned:
        path = Path(str(path).replace("/fit","/fit_fake"))

    for filename in path.glob("*.yaml"):
        model_best_fit = read_best_fit_model(filename)
        row = {}

        for par in model_best_fit.parameters:
            row[par.name] = par.value
            row[par.name + "_err"] = model_best_fit.parameters.error(par)

        rows.append(row)

    table = table_from_row_data(rows)
    name = "fit-results-all"
    if binned:
        name = "fit_binned-results-all"
    filename = f"results/models/{model_name}/{name}.fits.gz"
    log.info(f"Writing {filename}")
    table.write(str(filename), overwrite=True)


@cli.command("plot-results", help="Plot results for given model")
@click.argument("model", type=click.Choice(list(AVAILABLE_MODELS) + ["all"]))
@click.option(
              "--obs_ids", default="0", nargs=1, help="Which observation to choose.", type=str
              )
def plot_results_cmd(model, obs_ids):
    if model == "all":
        models = AVAILABLE_MODELS
    else:
        models = [model]

    filename_dataset = get_filename_dataset(LIVETIME)

    for model in models:
        for obs_id in parse_obs_ids(obs_ids, model):
            filename_model = BASE_PATH / f"models/{model}.yaml"
            plot_results(filename_model=filename_model, filename_dataset=filename_dataset, obs_id=obs_id)


def save_figure(filename):
    path = BASE_PATH / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Writing {path}")
    plt.savefig(path, dpi=DPI)
    plt.clf()
    plt.close()



def plot_spectra(model, model_best_fit, obs_id):
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
    obs_id = int(obs_id)
    filename = f"results/models/{model.name}/plots/spectra/spectra_{obs_id:04d}.png"
    save_figure(filename)


def plot_residuals(dataset, obs_id):
    # plot residuals
    model = dataset.models[0]
    spatial_model = model.spatial_model
    if spatial_model.__class__.__name__ == "PointSpatialModel":
        region = CircleSkyRegion(center=spatial_model.position, radius=0.1 * u.deg)
    else:
        region = spatial_model.to_region()

    dataset.plot_residuals(method="diff/sqrt(model)", vmin=-0.5, vmax=0.5, region=region, figsize=(10, 4))
    obs_id = int(obs_id)
    filename = f"results/models/{model.name}/plots/residuals/residuals_{obs_id:04d}.png"
    save_figure(filename)


def plot_residual_distribution(dataset, obs_id):
    # plot residual significance distribution
    model = dataset.models[0]

    tophat_2D_kernel = Tophat2DKernel(5)
    l_m = lima(dataset.counts.sum_over_axes(keepdims=False), dataset.npred().sum_over_axes(keepdims=False), tophat_2D_kernel)
    sig_resid = l_m["significance"].data[np.isfinite(l_m["significance"].data)]

#    resid = dataset.residuals()
#    sig_resid = resid.data[np.isfinite(resid.data)]

    plt.hist(
        sig_resid, density=True, alpha=0.5, color="red", bins=100,
    )

    mu, std = norm.fit(sig_resid)
    # replace with log.info()
    print("Fit results: mu = {:.2f}, std = {:.2f}".format(mu, std))
    x = np.linspace(-8, 8, 50)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, lw=2, color="black", label="Fit results: mu = {:.2f}, std = {:.2f}".format(mu, std))
    plt.legend()
    plt.xlabel("Significance")
    plt.yscale("log")
    plt.ylim(1e-5, 1)
    xmin, xmax = np.min(sig_resid), np.max(sig_resid)
    plt.xlim(xmin, xmax)

    obs_id = int(obs_id)
    filename = f"results/models/{model.name}/plots/residuals-distribution/residuals-distribution_{obs_id:04d}.png"
    save_figure(filename)


def read_best_fit_model(filename):
    log.info(f"Reading {filename}")
    model_best_fit = Models.read(filename)

    path = get_filename_covariance(filename)
    log.info(f"Reading {path}")
    pars = model_best_fit.parameters
    pars.covariance = np.loadtxt(str(path))

    spectral_model_best_fit = model_best_fit[0].spectral_model
    covar = pars.get_subcovariance(spectral_model_best_fit.parameters)
    spectral_model_best_fit.parameters.covariance = covar

    spatial_model_best_fit = model_best_fit[0].spatial_model
    covar = pars.get_subcovariance(spatial_model_best_fit.parameters)
    spatial_model_best_fit.parameters.covariance = covar
    return model_best_fit


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

    path = get_filename_best_fit_model(filename_model, obs_id)
    model_best_fit = read_best_fit_model(path)

    plot_spectra(model[0], model_best_fit[0], obs_id)

    dataset = read_dataset(filename_dataset, filename_model, obs_id)
    dataset.models = model_best_fit
    plot_residuals(dataset, obs_id)
    plot_residual_distribution(dataset, obs_id)


@cli.command("plot-pull-distributions", help="Plot pull distributions for the given model")
@click.argument("model", type=click.Choice(list(AVAILABLE_MODELS) + ["all"]))
@click.option(
              "--binned", default=False, nargs=1, help="Which observation to choose.", type=str
              )
def plot_pull_distribution_cmd(model, binned):
    if model == "all":
        models = AVAILABLE_MODELS
    else:
        models = [model]

    for model in models:
        plot_pull_distribution(model_name=model, binned=binned)


def plot_pull_distribution(model_name, binned=False):
    name = "fit-results-all"
    if binned:
        name = "fit_binned-results-all"
    filename = BASE_PATH / f"results/models/{model_name}/{name}.fits.gz"
    results = Table.read(str(filename))

    filename_ref = BASE_PATH / f"models/{model_name}.yaml"
    model_ref = Models.read(filename_ref)[0]
    names = [name for name in results.colnames if "err" not in name]

    plots = "plots"
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

        plt.hist(pull, bins=21, normed=True)
        plt.xlim(-5, 5)
        plt.xlabel("(value - value_true) / error")
        plt.ylabel("PDF")
        plt.title(f"Pull distribution for {model_name}: {name} ")
        filename = f"results/models/{model_name}/{plots}/pull-distribution-{name}.png"
        save_figure(filename)


if __name__ == "__main__":
    cli()
