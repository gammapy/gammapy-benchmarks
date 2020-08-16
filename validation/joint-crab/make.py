#!/usr/bin/env python
"""Run Gammapy validation: CTA 1DC"""
import logging
import warnings
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table
from regions import CircleAnnulusSkyRegion, CircleSkyRegion

from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.datasets import Datasets
from gammapy.maps import MapAxis
from gammapy.modeling import Fit
from gammapy.modeling.models import LogParabolaSpectralModel, SkyModel
from gammapy.utils.scripts import make_path
from joint_crab.extract_fermi import extract_spectrum_fermi

log = logging.getLogger(__name__)

# AVAILABLE_DATA = ["hess", "magic", "veritas", "fact", "fermi", "joint"]
AVAILABLE_DATA = ["hess", "magic", "veritas", "fact", "joint"]

DATASETS = [
    {
        "name": "joint",
        "label": "joint fit",
        "energy_range": {"min": "0.03 TeV", "max": "30 TeV"},
        "color": "crimson",
    },
]

instrument_opts = dict(
    hess={
        "on_radius": "0.11 deg",
        "stack": False,
        "containment": True,
        "emin": "0.66 TeV",
        "emax": "30 TeV",
        "color": "#002E63",
    },
    magic={
        "on_radius": "0.14 deg",
        "stack": False,
        "containment": False,
        "emin": "0.08 TeV",
        "emax": "30 TeV",
        "color": "#FF9933",
    },
    veritas={
        "on_radius": "0.1 deg",
        "stack": False,
        "containment": False,
        "emin": "0.15 TeV",
        "emax": "30 TeV",
        "color": "#893F45",
    },
    fact={
        "on_radius": "0.1732 deg",
        "stack": True,
        "containment": False,
        "emin": "0.4 TeV",
        "emax": "30 TeV",
        "color": "#3EB489",
    },
    fermi={
        "on_radius": "0.3 deg",
        "stack": False,
        "containment": True,
        "emin": "0.03 TeV",
        "emax": "2 TeV",
        "color": "#21ABCD",
    },
)


@click.group()
@click.option(
    "--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING"]),
)
@click.option("--show-warnings", is_flag=True, help="Show warnings?")
def cli(log_level, show_warnings):
    logging.basicConfig(level=log_level)
    log.setLevel(level=log_level)

    if not show_warnings:
        warnings.simplefilter("ignore")


@cli.command("run-analyses", help="Run Gammapy validation: joint Crab")
@click.argument("instruments", type=click.Choice(list(AVAILABLE_DATA) + ["all"]))
@click.argument("npoints", required=False, type=int, default=10)
def run_analyses(instruments, npoints=10):
    # Loop over instruments
    if instruments == "all":
        instruments = list(AVAILABLE_DATA)
    else:
        instruments = [instruments]

    joint = []
    for instrument in instruments:
        # First perform data reduction and save to disk
        if instrument != "joint":
            if instrument != "fermi":
                data_reduction(instrument)
            else:
                data_reduction_fermi()

        # data_fitting(instrument, npoints)
        # make_summary(instrument)


@cli.command("run-fit", help="Run Gammapy fit: joint Crab")
@click.argument("instruments", type=click.Choice(list(AVAILABLE_DATA) + ["all"]))
@click.argument("npoints", required=False, type=int, default=10)
def run_fit(instruments, npoints=10):
    # Loop over instruments
    if instruments == "all":
        instruments = list(AVAILABLE_DATA)
    else:
        instruments = [instruments]

    for instrument in instruments:
        data_fitting(instrument, npoints)
        make_summary(instrument)


def data_reduction(instrument):
    log.info(f"data_reduction: {instrument}")
    config = AnalysisConfig.read(f"config.yaml")
    config.observations.datastore = str(
        Path().resolve().parent / "data" / "joint-crab" / instrument
    )
    config.datasets.stack = instrument_opts[instrument]["stack"]
    config.datasets.containment_correction = instrument_opts[instrument]["containment"]
    config.datasets.on_region.radius = instrument_opts[instrument]["on_radius"]

    if instrument == "fact":
        config.datasets.safe_mask.methods = ["aeff-default"]

    analysis = Analysis(config)
    analysis.get_observations()

    analysis.get_datasets()
    if instrument == "fact":
        counts = analysis.datasets[0].counts
        data = counts.geom.energy_mask(emin=0.4 * u.TeV)
        analysis.datasets[0].mask_safe = counts.copy(data=data)

    analysis.datasets.write(f"reduced_{instrument}", overwrite=True)


def data_reduction_fermi():
    log.info(f"data_reduction: fermi")
    containment_correction = instrument_opts["fermi"]["containment"]
    radius = instrument_opts["fermi"]["on_radius"]
    emin = u.Quantity(instrument_opts["fermi"]["emin"]).to_value("TeV")
    emax = u.Quantity(instrument_opts["fermi"]["emax"]).to_value("TeV")

    crab_pos = SkyCoord(ra=83.63, dec=22.01, unit="deg", frame="icrs")
    on_region = CircleSkyRegion(crab_pos, radius=Angle(radius))
    off_region = CircleAnnulusSkyRegion(
        crab_pos, inner_radius=1 * u.deg, outer_radius=2 * u.deg
    )

    energy = MapAxis.from_bounds(
        emin, emax, 36, unit="TeV", name="energy", interp="log"
    )
    dataset = extract_spectrum_fermi(
        on_region, off_region, energy, containment_correction
    )
    datasets = Datasets([dataset])

    datasets.write(f"reduced_fermi", overwrite=True)


def define_model():
    crab_spectrum = LogParabolaSpectralModel(
        amplitude=3e-11 / u.cm ** 2 / u.s / u.TeV,
        reference=1 * u.TeV,
        alpha=2.3,
        beta=0.2,
    )

    crab_spectrum.beta.min = -0.5
    crab_spectrum.beta.max = 1

    crab_model = SkyModel(spatial_model=None, spectral_model=crab_spectrum, name="crab")

    return crab_model


def make_contours(fit, result, npoints):
    log.info(f"Running contours with {npoints} points...")
    contours = dict()
    contour = fit.minos_contour(
        result.parameters["alpha"],
        result.parameters["beta"],
        numpoints=npoints,
        sigma=np.sqrt(2.3),
    )
    contours["contour_alpha_beta"] = {
        "alpha": contour["x"].tolist(),
        "beta": (contour["y"] * np.log(10)).tolist(),
    }

    contour = fit.minos_contour(
        result.parameters["amplitude"],
        result.parameters["beta"],
        numpoints=npoints,
        sigma=np.sqrt(2.3),
    )
    contours["contour_amplitude_beta"] = {
        "amplitude": contour["x"].tolist(),
        "beta": (contour["y"] * np.log(10)).tolist(),
    }

    contour = fit.minos_contour(
        result.parameters["amplitude"],
        result.parameters["alpha"],
        numpoints=npoints,
        sigma=np.sqrt(2.3),
    )
    contours["contour_amplitude_alpha"] = {
        "amplitude": contour["x"].tolist(),
        "alpha": contour["y"].tolist(),
    }

    return contours


def read_datasets_and_set_model(instrument, model):
    # Read from disk
    datasets = Datasets.read(f"reduced_{instrument}", f"_datasets.yaml", "_models.yaml")

    e_min = u.Quantity(instrument_opts[instrument]["emin"])
    e_max = u.Quantity(instrument_opts[instrument]["emax"])

    # Set model and fit range
    for ds in datasets:
        ds.models = model
        ds.mask_fit = ds.counts.geom.energy_mask(e_min, e_max)

    return datasets


def data_fitting(instrument, npoints):
    log.info("Running fit ...")
    # First define model
    crab_model = define_model()

    if instrument != "joint":
        datasets = read_datasets_and_set_model(instrument, crab_model)
    else:
        log.info("Performing joint analysis")
        ds_list = []
        for inst in AVAILABLE_DATA[:-1]:
            datasets = read_datasets_and_set_model(inst, crab_model)
            ds_list = [*ds_list, *datasets]
        datasets = Datasets(ds_list)

    # Perform fit
    fit = Fit(datasets)
    result = fit.run(optimize_opts={"tol": 0.1, "strategy": 1})
    log.info(result.parameters.to_table())

    path = f"results/fit_{instrument}.rst"
    log.info(f"Writing {path}")
    result.parameters.to_table().write(path, format="ascii.rst", overwrite=True)

    contours = make_contours(fit, result, npoints)
    with open(f"results/contours_{instrument}.yaml", "w") as file:
        yaml.dump(contours, file)


def plot_contour_line(ax, x, y, **kwargs):
    """Plot smooth curve from points.

    There is some noise in the contour points from MINUIT,
    which throws off most interpolation schemes.

    The LSQUnivariateSpline as used here gives good results.

    It could probably be simplified, or Bezier curve plotting via
    matplotlib could also be tried:
    https://matplotlib.org/gallery/api/quad_bezier.html
    """
    from scipy.interpolate import LSQUnivariateSpline

    x = np.hstack([x, x, x])
    y = np.hstack([y, y, y])

    t = np.linspace(-1, 2, len(x), endpoint=False)
    tp = np.linspace(0, 1, 50)

    t_knots = np.linspace(t[1], t[-2], 10)
    xs = LSQUnivariateSpline(t, x, t_knots, k=5)(tp)
    ys = LSQUnivariateSpline(t, y, t_knots, k=5)(tp)

    ax.plot(xs, ys, **kwargs)


def plot_contours(instrument):
    log.info(f"Plotting contours: {instrument}")

    filename = make_path(
        str(Path().resolve().parent / "data" / "joint-crab" / "published" / "fit")
    )
    filename = filename / f"contours_{instrument}.yaml"
    with open(filename, "r") as file:
        paper_contours = yaml.safe_load(file)

    with open(f"results/contours_{instrument}.yaml", "r") as file:
        contours = yaml.safe_load(file)

    pars = {
        "phi": {
            "label": r"$\phi_0 \,/\,(10^{-11}\,{\rm TeV}^{-1} \, {\rm cm}^{-2} {\rm s}^{-1})$",
            "lim": [2.6, 5.8],
            "ticks": [3, 4, 5],
        },
        "gamma": {
            "label": r"$\Gamma$",
            "lim": [1.9, 3.0],
            "ticks": [2.0, 2.3, 2.6, 2.9],
        },
        "beta": {
            "label": r"$\beta$",
            "lim": [-0.1, 1.0],
            "ticks": [0.0, 0.3, 0.6, 0.9],
        },
    }

    panels = [
        {
            "x": "phi",
            "y": "gamma",
            "cx": np.array(1e11) * contours["contour_amplitude_alpha"]["amplitude"],
            "cy": contours["contour_amplitude_alpha"]["alpha"],
            "pcx": np.array(1e11)
            * paper_contours["contour_amplitude_alpha"]["amplitude"],
            "pcy": paper_contours["contour_amplitude_alpha"]["alpha"],
        },
        {
            "x": "phi",
            "y": "beta",
            "cx": np.array(1e11) * contours["contour_amplitude_beta"]["amplitude"],
            "cy": contours["contour_amplitude_beta"]["beta"],
            "pcx": np.array(1e11)
            * paper_contours["contour_amplitude_beta"]["amplitude"],
            "pcy": paper_contours["contour_amplitude_beta"]["beta"],
        },
        {
            "x": "gamma",
            "y": "beta",
            "cx": contours["contour_alpha_beta"]["alpha"],
            "cy": contours["contour_alpha_beta"]["beta"],
            "pcx": paper_contours["contour_alpha_beta"]["alpha"],
            "pcy": paper_contours["contour_alpha_beta"]["beta"],
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for p, ax in zip(panels, axes):
        x = pars[p["x"]]
        y = pars[p["y"]]
        plot_contour_line(
            ax, p["cx"], p["cy"], ls="-", lw=2.5, color="b", label="gammapy"
        )
        plot_contour_line(
            ax, p["pcx"], p["pcy"], ls="-", lw=2.5, color="r", label="ref"
        )
        ax.set_xlabel(x["label"])
        ax.set_ylabel(y["label"])
        #        ax.set_xlim(x["lim"])
        #        ax.set_ylim(y["lim"])
        ax.set_xticks(x["ticks"])
        ax.set_yticks(y["ticks"])
        plt.legend()

    plt.savefig(f"results/contours_{instrument}.png", bbox_inches="tight")
    plt.close()


def make_summary(instrument):
    log.info(f"Preparing summary: {instrument}")

    # Data info

    # Fit results
    path = f"results/fit_{instrument}.rst"
    tab = Table.read(path, format="ascii")
    tab.add_index("name")
    dt = "U30"
    comp_tab = Table(names=("Param", "joint crab paper", "gammapy"), dtype=[dt, dt, dt])

    filename = make_path(
        str(Path().resolve().parent / "data" / "joint-crab" / "published" / "fit")
    )
    filename = filename / f"fit_{instrument}.yaml"
    with open(filename, "r") as file:
        paper_result = yaml.safe_load(file)

    for par in paper_result["parameters"]:
        if par["name"] is not "reference":
            name = par["name"]
            ref = par["value"]
            ref_error = par["error"]
            if name == "beta":
                factor = np.log(10)
            else:
                factor = 1
            value = tab.loc[name]["value"] * factor
            error = tab.loc[name]["error"] * factor
            comp_tab.add_row(
                [name, f"{ref:.3e} ± {ref_error:.3e}", f"{value:.3e} ± {error:.3e}"]
            )

    # Generate README.md file with table and plots
    path = f"results/{instrument}_comparison_table.md"
    comp_tab.write(path, format="ascii.html", overwrite=True)

    txt = Path(f"results/{instrument}_comparison_table.md").read_text()

    plot_contours(instrument)
    im1 = f"\n ![Contours](contours_{instrument}.png)"

    out = txt + im1
    Path(f"results/{instrument}_summary.md").write_text(out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
