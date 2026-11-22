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

from utils import build_observation, build_transient_model
from tc_utils import simulate_and_characterize, summarize_transient, create_pull_figure

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


@cli.command("tc", help="Run transient characterization validation")
@click.option("--livetime", type=str, default="30 min")
@click.option("--t_peak_offset", type=str, default="10 min", help="Time of the flare peak after the observation start.")
@click.option("--t_decay", type=str, default="200 s")
@click.option("--t_rise", type=str, default="10 s")
@click.option("--crab_fraction", type=float, default=5.0)
@click.option("--correlation_radius", type=str, default="0.2 deg")
@click.option("--detection_threshold", type=float, default=5.0)
@click.option("--bin_width", type=str, default="30 s")
@click.option("--on_region_radius", type=str, default="0.11 deg")
@click.option("--n_samples", type=int, default=20)
@click.option("--n_jobs", type=int, default=4)
def run_tc(
    livetime, t_peak_offset, t_decay, t_rise, crab_fraction, correlation_radius,
    detection_threshold, bin_width, on_region_radius, n_samples, n_jobs
):
    """Run transient characterization validation."""
    start_time = time.time()

    livetime = u.Quantity(livetime)

    log.info("Building observation and transient model.")
    obs = build_observation(livetime=livetime)
    t_peak = obs.tstart + u.Quantity(t_peak_offset)
    model = build_transient_model(
        t_peak=t_peak, t_decay=t_decay, t_rise=t_rise, crab_fraction=crab_fraction
    )

    config = {
        "correlation_radius": correlation_radius,
        "detection_threshold": detection_threshold,
        "min_distance": correlation_radius,
        "bin_width": bin_width,
        "on_region_radius": on_region_radius,
        "t_rise": t_rise,
    }

    log.info(f"Starting simulations.")
    with multiprocessing_manager(backend="multiprocessing", pool_kwargs=dict(processes=n_jobs)):
        results = perform_tc_simulation(n_samples, obs, model, config)

    log.info(f"Compute summary and plot result.")
    dir = Path("results")
    dir.mkdir(exist_ok=True)

    summary = summarize_transient(results)
    summary.update({
        "crab_fraction": crab_fraction,
        "livetime_h": livetime.to_value("h"),
        "t_peak_offset_s": u.Quantity(t_peak_offset).to_value("s"),
        "t_decay_s": u.Quantity(t_decay).to_value("s"),
        "t_rise_s": u.Quantity(t_rise).to_value("s"),
        "correlation_radius_deg": u.Quantity(correlation_radius).to_value("deg"),
        "bin_width_s": u.Quantity(bin_width).to_value("s"),
        "on_region_radius_deg": u.Quantity(on_region_radius).to_value("deg"),
    })

    pulls_filename = dir / f"transient_characterization_{crab_fraction}crab_{livetime.to_value('h')}h.png"
    create_pull_figure(summary, pulls_filename)

    json_filename = dir / f"transient_characterization_{crab_fraction}crab_{livetime.to_value('h')}h.json"
    log.info(f"Write summary to {json_filename}.")
    with json_filename.open("w") as fh:
        json.dump(summary, fh, indent=2)

    end_time = time.time()
    duration = end_time - start_time
    log.info(f"The total time taken for the transient characterization validation is: {duration} s ({duration/60} min)")


def perform_tc_simulation(nsim, obs, model, config):
    indices = np.arange(nsim)
    inputs = [(obs, model, config) for _ in indices]
    return run_multiprocessing(simulate_and_characterize, inputs, task_name="simulation")


if __name__ == "__main__":
    cli()
