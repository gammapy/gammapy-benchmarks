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

from utils import build_observation, build_dataset_3d, build_extended_model
from isd_utils import (
    fake_detect_fit_recompute, summarize_isd, create_residual_figure, create_pull_figure,
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


@cli.command("isd", help="Run isolated source detection validation")
@click.option("--livetime", type=str, default="5 h")
@click.option("--crab_fraction", type=float, default=1.0)
@click.option("--sigma", type=str, default="0.1 deg")
@click.option("--correlation_radius", type=str, default="0.2 deg")
@click.option("--detection_threshold", type=float, default=5.0)
@click.option("--dataset_width", type=str, default="3 deg")
@click.option(
    "--exclusion_radius", type=str, default=None,
    help="Region around the fitted source excluded from the residual mean/std "
         "(default: 3x correlation_radius). The fit consumes degrees of freedom "
         "concentrated there, which suppresses the local residual variance.",
)
@click.option("--n_samples", type=int, default=100)
@click.option("--n_jobs", type=int, default=4)
def run_isd(
    livetime, crab_fraction, sigma, correlation_radius, detection_threshold,
    dataset_width, exclusion_radius, n_samples, n_jobs
):
    """Run isolated source detection validation."""
    start_time = time.time()

    livetime = u.Quantity(livetime)

    log.info("Building observation and dataset.")
    obs = build_observation(livetime=livetime)
    dataset = build_dataset_3d(obs, width=dataset_width)
    model = build_extended_model(percent_crab=crab_fraction, sigma=sigma)

    config = {
        "correlation_radius": correlation_radius,
        "detection_threshold": detection_threshold,
        "min_distance": correlation_radius,
        "sigma_init": sigma,
    }
    if exclusion_radius is not None:
        config["exclusion_radius"] = exclusion_radius

    log.info(f"Starting simulations.")
    with multiprocessing_manager(backend="multiprocessing", pool_kwargs=dict(processes=n_jobs)):
        results = perform_isd_simulation(n_samples, dataset, model, config)

    log.info(f"Compute summary and plot result.")
    dir = Path("results")
    dir.mkdir(exist_ok=True)

    summary = summarize_isd(results)
    summary.update({
        "crab_fraction": crab_fraction,
        "livetime_h": livetime.to_value("h"),
        "sigma_deg": u.Quantity(sigma).to_value("deg"),
        "correlation_radius_deg": u.Quantity(correlation_radius).to_value("deg"),
        "dataset_width_deg": u.Quantity(dataset_width).to_value("deg"),
        "exclusion_radius_deg": u.Quantity(config.get("exclusion_radius") or 3 * u.Quantity(correlation_radius)).to_value("deg"),
    })

    filename = dir / f"isolated_source_detection_{crab_fraction}crab_{livetime.to_value('h')}h.png"
    create_residual_figure(summary, filename)

    pulls_filename = dir / f"isolated_source_detection_pulls_{crab_fraction}crab_{livetime.to_value('h')}h.png"
    create_pull_figure(summary, pulls_filename)

    json_filename = dir / f"isolated_source_detection_{crab_fraction}crab_{livetime.to_value('h')}h.json"
    log.info(f"Write summary to {json_filename}.")
    with json_filename.open("w") as fh:
        json.dump(summary, fh, indent=2)

    end_time = time.time()
    duration = end_time - start_time
    log.info(f"The total time taken for the isolated source detection validation is: {duration} s ({duration/60} min)")


def perform_isd_simulation(nsim, dataset, model, config):
    indices = np.arange(nsim)
    inputs = [(dataset, model, config) for _ in indices]
    return run_multiprocessing(fake_detect_fit_recompute, inputs, task_name="simulation")


if __name__ == "__main__":
    cli()
