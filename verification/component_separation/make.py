import json
import logging
import sys
import time
import warnings
from pathlib import Path

import click

import numpy as np
import astropy.units as u

from gammapy.utils.parallel import run_multiprocessing, multiprocessing_manager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import build_observation, build_dataset_3d
from cs_utils import (
    build_scene_model, fake_scene_dataset_3d, fit_big_source,
    detect_and_characterize_small_source, run_flux_collection,
    simulate_and_characterize, summarize_component_separation, create_pull_figure,
)

log = logging.getLogger(__name__)


@click.group()
@click.option(
    "--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING"])
)
@click.option("--show-warnings", is_flag=True, help="Show warnings?")
def cli(log_level, show_warnings):
    logging.basicConfig(level=log_level)

    if not show_warnings:
        warnings.simplefilter("ignore")


@cli.command("cs", help="Run component separation validation")
@click.option("--livetime", type=str, default="5 h")
@click.option("--big_sigma", type=str, default="0.5 deg")
@click.option("--big_crab_fraction", type=float, default=0.2)
@click.option("--big_index", type=float, default=2.5)
@click.option("--small_sigma", type=str, default="0.05 deg")
@click.option("--small_crab_fraction", type=float, default=0.01)
@click.option("--small_index", type=float, default=1.8)
@click.option("--separation", type=str, default="0.2 deg")
@click.option("--dataset_width", type=str, default="3 deg")
@click.option(
    "--n_sigma", type=float, default=5.0,
    help="Small-source detection significance threshold.",
)
@click.option("--n_samples", type=int, default=100)
@click.option("--n_jobs", type=int, default=4)
def run_cs(
    livetime, big_sigma, big_crab_fraction, big_index,
    small_sigma, small_crab_fraction, small_index, separation,
    dataset_width, n_sigma, n_samples, n_jobs,
):
    """Run component separation validation."""
    start_time = time.time()

    livetime = u.Quantity(livetime)

    log.info("Building observation, dataset and scene model.")
    obs = build_observation(livetime=livetime)
    dataset = build_dataset_3d(obs, width=dataset_width)
    models = build_scene_model(
        big_sigma=big_sigma, big_crab_fraction=big_crab_fraction, big_index=big_index,
        small_sigma=small_sigma, small_crab_fraction=small_crab_fraction, small_index=small_index,
        separation=separation,
    )

    config = {
        "big_sigma_init": big_sigma,
        "small_sigma_init": small_sigma,
        "n_sigma": n_sigma,
    }

    log.info("Starting simulations.")
    with multiprocessing_manager(backend="multiprocessing", pool_kwargs=dict(processes=n_jobs)):
        results = perform_cs_simulation(n_samples, dataset, models, config)

    log.info("Compute summary and plot result.")
    dir = Path("results")
    dir.mkdir(exist_ok=True)

    summary = summarize_component_separation(results)
    summary.update({
        "big_crab_fraction": big_crab_fraction,
        "big_sigma_deg": u.Quantity(big_sigma).to_value("deg"),
        "big_index": big_index,
        "small_crab_fraction": small_crab_fraction,
        "small_sigma_deg": u.Quantity(small_sigma).to_value("deg"),
        "small_index": small_index,
        "separation_deg": u.Quantity(separation).to_value("deg"),
        "livetime_h": livetime.to_value("h"),
        "dataset_width_deg": u.Quantity(dataset_width).to_value("deg"),
        "n_sigma": n_sigma,
    })

    stem = (
        f"component_separation_big{big_crab_fraction}crab_small{small_crab_fraction}"
        f"crab_{livetime.to_value('h')}h"
    )

    pulls_filename = dir / f"{stem}.png"
    create_pull_figure(summary, pulls_filename)

    json_filename = dir / f"{stem}.json"
    log.info(f"Write summary to {json_filename}.")
    with json_filename.open("w") as fh:
        json.dump(summary, fh, indent=2)

    log.info("Running one reference realization for FitResult/FluxPoints products.")
    write_reference_realization(dataset, models, config, dir, stem)

    end_time = time.time()
    duration = end_time - start_time
    log.info(
        f"The total time taken for the component separation validation is: "
        f"{duration} s ({duration/60} min)"
    )


def perform_cs_simulation(nsim, dataset, models, config):
    indices = np.arange(nsim)
    inputs = [(dataset, models, config) for _ in indices]
    return run_multiprocessing(simulate_and_characterize, inputs, task_name="simulation")


def write_reference_realization(dataset, models, config, dir, stem):
    """Run one additional realization end-to-end and write its FitResult/FluxPoints products.

    These per-component artifacts (the joint fit result, per-component flux
    points) are heavy and only meaningful for a single representative
    realization, not the full `n_samples` Monte Carlo batch -- re-running one
    extra realization here (cheap next to the batch itself) keeps
    `simulate_and_characterize`, the per-realization worker, free of this
    write-to-disk side effect and its non-JSON-serializable return values.
    """
    faked_dataset = fake_scene_dataset_3d(dataset, models)
    fitted_big_dataset, big_model, big_fit_success = fit_big_source(
        faked_dataset, sigma_init=config.get("big_sigma_init", "0.5 deg"),
    )
    if big_model is None or not big_fit_success:
        log.warning(
            "Reference realization: big source not detected/fit -- "
            "skipping FitResult/FluxPoints products."
        )
        return

    joint_dataset, fit_result, small_detected = detect_and_characterize_small_source(
        fitted_big_dataset, big_model,
        sigma_init=config.get("small_sigma_init", "0.05 deg"),
        n_sigma=config.get("n_sigma", 5),
    )
    if not small_detected:
        log.warning(
            "Reference realization: small source not detected -- "
            "skipping FitResult/FluxPoints products."
        )
        return

    fit_result_filename = dir / f"{stem}_fit_result.fits"
    log.info(f"Write reference fit result to {fit_result_filename}.")
    fit_result.write(fit_result_filename, overwrite=True)

    flux_points = run_flux_collection(
        joint_dataset, fit_result.models["big"], fit_result.models["small"]
    )
    for name, fp in flux_points.items():
        flux_points_filename = dir / f"{stem}_flux_points_{name}.fits"
        log.info(f"Write {name} flux points to {flux_points_filename}.")
        fp.write(flux_points_filename, overwrite=True)


if __name__ == "__main__":
    cli()
