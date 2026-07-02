import logging
import time
import warnings
from pathlib import Path

import click

import numpy as np
import astropy.units as u
from astropy.table import Table

from gammapy.estimators import FluxPoints
from gammapy.maps import LabelMapAxis
from gammapy.utils.parallel import run_multiprocessing, multiprocessing_manager

from utils import (build_observation, build_dataset_1d, build_dataset_3d,
                   build_model, fake_and_apply_fpe, create_coverage_figure, fake_and_apply_fe,
                   create_sensitivity_figure)
AVAILABLE_GEOMS = ["1d", "3d"]

log = logging.getLogger(__name__)

SHOW_PROGRESS_BAR=True

@click.group()
@click.option(
    "--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING"])
)
@click.option("--show-warnings", is_flag=True, help="Show warnings?")
def cli(log_level, show_warnings):
    logging.basicConfig(level=log_level)

    if not show_warnings:
        warnings.simplefilter("ignore")


@cli.command("fp_coverage", help="Run flux point coverage validation")
@click.argument("geometries", type=click.Choice(list(AVAILABLE_GEOMS) + ["all"]))
@click.option("--livetime", type=str, default="1 h")
@click.option("--crab_fraction", type=float, default=0.1)
@click.option("--n_samples", type=int, default=1000)
@click.option("--n_sigma", type=float, default=1)
@click.option("--n_sigma_ul", type=float, default=1.6448)
@click.option("--n_jobs", type=int, default=4)
@click.option("--save_results", is_flag=True)
def run_fp_coverage(geometries, livetime, crab_fraction, n_samples, n_sigma, n_sigma_ul, n_jobs, save_results):
    """Run coverage validation."""
    start_time = time.time()

    livetime = u.Quantity(livetime)
    geometries = list(AVAILABLE_GEOMS) if geometries == "all" else [geometries]

    for geometry in geometries:
        log.info(f"Perform CI coverage on {geometry} dataset.")

        obs = build_observation(livetime=livetime)

        if geometry == "1d":
            dataset = build_dataset_1d(obs)
        elif geometry == "3d":
            dataset = build_dataset_3d(obs)

        model = build_model(percent_crab=crab_fraction)

        energy_edges = dataset.counts.geom.axes["energy"].downsample(2).edges

        fpe_config = {
            "source" : "source",
            "energy_edges": energy_edges,
            "selection_optional": ["errn-errp", "ul"],
            "n_sigma": n_sigma,
            "n_sigma_ul": n_sigma_ul,
        }

        log.info(f"Starting simulations.")
        with multiprocessing_manager(backend="multiprocessing", pool_kwargs=dict(processes=n_jobs)):
            result = perform_fpe_simulation(n_samples, dataset, model, fpe_config)
            
        log.info(f"Compute coverage and plot result.")
        dir = Path("results")
        dir.mkdir(exist_ok=True)
        filename = dir / f"test_coverage_crab_{100*crab_fraction}percent_{str(livetime.to_value('h'))}h.png"
        create_coverage_figure(result, filename)

        if save_results:
            filename = dir / f"flux_points_crab_{100*crab_fraction}percent_{str(livetime.to_value('h'))}h.fits"
            log.info(f"Write result flux points to {filename}.")
            super(FluxPoints,result).write(filename)

    end_time = time.time()
    duration = end_time - start_time
    log.info(f"The total time taken for the coverage validation is: {duration} s ({duration/60} min)")


@cli.command("sensitivity", help="Run sensitivity evaluation validation")
@click.argument("geometries", type=click.Choice(list(AVAILABLE_GEOMS) + ["all"]))
@click.option("--livetime", type=str, default="1 h")
@click.option("--scan_range", type=(float, float, int), default=(0.5, 2., 10))
@click.option("--n_samples", type=int, default=1000)
@click.option("--n_sigma", type=float, default=3)
@click.option("--n_jobs", type=int, default=4)
@click.option("--save_results", is_flag=True)
def run_sensitivity_coverage(geometries, livetime, scan_range, n_samples, n_sigma, n_jobs, save_results):
    """Run coverage validation."""
    start_time = time.time()

    livetime = u.Quantity(livetime)
    geometries = list(AVAILABLE_GEOMS) if geometries == "all" else [geometries]

    fe_config = {"source": 0, "selection_optional": ["sensitivity"], "n_sigma_sensitivity": n_sigma}

    for geometry in geometries:
        log.info(f"Perform sensitivity validation on {geometry} dataset.")

        obs = build_observation(livetime=livetime)

        if geometry == "1d":
            dataset = build_dataset_1d(obs)
            dataset.mask_fit = dataset.counts.geom.energy_mask(0.1 * u.TeV, 100 * u.TeV)
        elif geometry == "3d":
            dataset = build_dataset_3d(obs)

        log.info(f"Compute expected sensitivity with Asimov dataset.")
        from gammapy.estimators.flux import FluxEstimator
        ref_fraction = 0.1
        model = build_model(percent_crab=ref_fraction)
        fe = FluxEstimator(**fe_config)
        dataset.models = model
        res = fe.run([dataset])
        sensitivity_asimov = res['norm_sensitivity']*ref_fraction
        log.info(f"Asimov sensitivity for observation is {sensitivity_asimov:.3f} crab.")
        sensitivity_amplitude = build_model(percent_crab=sensitivity_asimov).spectral_model.amplitude.quantity

        crab_fractions = np.geomspace(scan_range[0]*sensitivity_asimov, 
                                      scan_range[1]*sensitivity_asimov, 
                                      scan_range[2]
                                      )
        
        log.info(f"Start simulation loop over source flux.")

        results = []
        for crab_fraction in crab_fractions:
            model = build_model(percent_crab=crab_fraction)

            log.info(f"Starting simulations for {1e3*crab_fraction:.2f} mCrab.")
            with multiprocessing_manager(backend="multiprocessing", pool_kwargs=dict(processes=n_jobs)):
                simu = perform_sensitivity_simulation(n_samples, dataset, model, fe_config)
  
            result = dict()
            ref_amplitude = model.spectral_model.amplitude.quantity
            result["ref_amplitude"] = ref_amplitude
            result["flux"] = np.array([res['norm'] for res in simu]) * ref_amplitude
            result["sensitivity"] = np.array([res['norm_sensitivity'] for res in simu]) * ref_amplitude
            result["excess"] = np.array([res['npred_excess'][0] for res in simu])
            result["sqrt_ts"] = np.sqrt(np.array([res['ts'] for res in simu])) * np.sign(result["excess"])

            results.append(result)

        table = Table(results)

        log.info(f"Compute sensitivity and plot result.")
        dir = Path("results")
        dir.mkdir(exist_ok=True)
        filename = dir / f"sensitivity_{geometry}_{livetime.to_value('h')}h.png"
        create_sensitivity_figure(table, n_sigma, sensitivity_amplitude, filename)
        log.info(f"Saved sensitivity plot to {filename}.")

    end_time = time.time()
    duration = end_time - start_time
    log.info(f"The total time taken for the sensitivity validation is: {duration} s ({duration/60} min)")


def perform_fpe_simulation(nsim, dataset, model, fpe_config):
    indices = np.arange(nsim)

    # Force n_jobs to one to avoid multiprocessing in subprocess
    fpe_config["n_jobs"] = 1

    inputs = [(dataset, model, fpe_config)  for _ in indices]

    fps = run_multiprocessing(fake_and_apply_fpe, inputs, task_name="simulation")

    axis = LabelMapAxis(indices, name='index')
    result = FluxPoints.from_stack(
                maps=fps,
                axis=axis,
            )
    return result

def perform_sensitivity_simulation(nsim, dataset, model, fe_config):
    #fe_config["n_jobs"] = 1

    indices = np.arange(nsim)

    inputs = [(dataset, model, fe_config)  for _ in indices]

    return run_multiprocessing(fake_and_apply_fe, inputs, task_name="simulation")



if __name__ == "__main__":
    cli()
