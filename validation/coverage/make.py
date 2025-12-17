import logging
import time
import warnings
from pathlib import Path

import click
import yaml

import numpy as np
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt

from gammapy.data import FixedPointingInfo, Observation, observatory_locations
from gammapy.datasets import Datasets, SpectrumDataset, SpectrumDatasetOnOff
from gammapy.estimators import FluxPointsEstimator, FluxPoints
from gammapy.irf import load_irf_dict_from_file
from gammapy.makers import SpectrumDatasetMaker
from gammapy.maps import MapAxis, RegionGeom, LabelMapAxis, Map
from gammapy.modeling.models import SkyModel, create_crab_spectral_model
from gammapy.analysis import Analysis, AnalysisConfig

AVAILABLE_GEOMS = ["1d", "3d"]

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


@cli.command("run", help="Run flux point coverage validation")
@click.argument("geometries", type=click.Choice(list(AVAILABLE_GEOMS) + ["all"]))
@click.option("--livetime", type=str, default="1 h")
@click.option("--percent_crab", type=float, default=0.1)
@click.option("--n_samples", type=int, default=1000)
@click.option("--n_sigma", type=float, default=1)
@click.option("--n_sigma_ul", type=float, default=1.6448)
@click.option("--n_jobs", type=int, default=4)
def run(geometries, livetime, percent_crab, n_samples, n_sigma, n_sigma_ul, n_jobs):
    """Run coverage validation."""
    start_time = time.time()

    livetime = u.Quantity(livetime)
    geometries = list(AVAILABLE_GEOMS) if geometries == "all" else [geometries]

    for geometry in geometries:
        log.info(f"Perform CI coverage on {geometry} dataset.")

        obs = build_observation(livetime=livetime)

        dataset = build_dataset_1d(obs)

        model = build_model(percent_crab=percent_crab)

        energy_edges = dataset.counts.geom.axes["energy"].downsample(2).edges

        fpe = FluxPointsEstimator(
            energy_edges=energy_edges,
            selection_optional=["errn-errp", "ul"],
            n_sigma=n_sigma,
            n_sigma_ul=n_sigma_ul,
            n_jobs=n_jobs
        )

        log.info(f"Starting simulations.")
        result = perform_simulation(n_samples, dataset, model, fpe)

        log.info(f"Compute coverage and plot result.")
        dir = Path("results")
        dir.mkdir(exist_ok=True)
        filename = dir / f"test_coverage_crab_{100*percent_crab}percent_{str(livetime.to_value('h'))}h.png"
        create_coverage_figure(result, filename)

    end_time = time.time()
    duration = end_time - start_time
    log.info(f"The total time taken for the coverage validation is: {duration} s ({duration/60} min)")


def build_observation(livetime="1 h"):
    # Define simulation parameters parameters
    livetime = u.Quantity(livetime)

    pointing_position = SkyCoord(0, 0, unit="deg", frame="galactic")
    # We want to simulate an observation pointing at a fixed position in the sky.
    # For this, we use the `FixedPointingInfo` class
    pointing = FixedPointingInfo(
        fixed_icrs=pointing_position.icrs,
    )

    irfs = load_irf_dict_from_file(
        "$GAMMAPY_DATA/cta-caldb/Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits.gz"
    )

    location = observatory_locations["ctao_south"]
    return Observation.create(
        pointing=pointing,
        livetime=livetime,
        irfs=irfs,
        location=location,
    )

def build_dataset_1d(obs, offset="0.5 deg"):
    offset = u.Quantity(offset)

    # Reconstructed and true energy axis
    energy_axis = MapAxis.from_energy_bounds(0.1, 100, 6, per_decade=True, unit="TeV")
    energy_axis_true = MapAxis.from_energy_bounds(0.05, 200, 12, per_decade=True, unit="TeV", name="energy_true")

    on_region_radius = Angle("0.11 deg")

    pointing_position = obs.get_pointing_icrs(obs.tmid)
    center = pointing_position.directional_offset_by(
        position_angle=0 * u.deg, separation=offset
    )

    on_region = CircleSkyRegion(center=center, radius=on_region_radius)

    # Make the SpectrumDataset
    geom = RegionGeom.create(region=on_region, axes=[energy_axis])

    dataset_empty = SpectrumDataset.create(
        geom=geom, energy_axis_true=energy_axis_true, name="obs-0"
    )
    maker = SpectrumDatasetMaker(selection=["exposure", "edisp", "background"])

    return maker.run(dataset_empty, obs)

def build_model(percent_crab=0.1):
    model_simu = create_crab_spectral_model('magic_lp')
    model_simu.amplitude.value *= percent_crab
    return SkyModel(spectral_model=model_simu, name="source")

def fake_dataset(dataset, model):
    dataset_on_off = SpectrumDatasetOnOff.from_spectrum_dataset(
        dataset=dataset, acceptance=1, acceptance_off=10
    )
    dataset_on_off.models = model.copy()

    dataset_on_off.fake(npred_background=dataset.npred_background())
    return dataset_on_off

def reduce_dimensionality_flux_points(flux_points):
    flux_points = flux_points.copy()
    for name, quantity in flux_points._data.items():
        if "dataset" in quantity.geom.axes.names:
            map_obj = quantity.sum_over_axes(axes_names=["dataset"], keepdims=False)
            flux_points._data[name] = map_obj
    return flux_points

def perform_simulation(nsim, dataset, model, estimator):
    fps  = []
    indices = np.arange(nsim)

    for i in tqdm(indices, total=len(indices)):
        dataset_on_off = fake_dataset(dataset, model)
        fp = estimator.run([dataset_on_off])
        fps.append(reduce_dimensionality_flux_points(fp))

    axis = LabelMapAxis(indices, name='index')
    result = FluxPoints.from_stack(
                maps=fps,
                axis=axis,
            )
    return result


def compute_ci_coverage(result_fp, remove_ul=False):
    energy_axis = result_fp.geom.axes["energy"]

    weights = ~result_fp.is_ul * 1.0 if remove_ul else np.ones(result_fp.is_ul.data.shape)

    ci_min = result_fp.norm - result_fp.norm_errn
    ci_max = result_fp.norm + result_fp.norm_errp

    in_ci = (ci_min < 1.0) & (ci_max > 1.0)
    geom = in_ci.geom.to_image().to_cube([energy_axis])

    return Map.from_geom(geom, data=np.average(in_ci, axis=0, weights=weights))


def compute_ul_coverage(result_fp):
    energy_axis = result_fp.geom.axes["energy"]
    in_ul = result_fp.norm_ul > 1.
    geom = in_ul.geom.to_image().to_cube([energy_axis])
    return Map.from_geom(geom, data=np.mean(in_ul, axis=0))

def create_coverage_figure(result, filename):
    coverage_ci = compute_ci_coverage(result, False)
    coverage_ul = compute_ul_coverage(result)

    nsim = result.geom.axes['index'].nbin

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    coverage_ci.plot(ax=ax1, color="k")
    ax1.set_yscale("linear")

    ref_val = 1 - 2 * norm.sf(result.n_sigma)
    ref_min = ref_val * (1 - nsim ** -0.5)
    ref_max = ref_val * (1 + nsim ** -0.5)
    ax1.axhline(ref_val, color='k')
    ax1.axhspan(ref_min, ref_max, color='b', alpha=0.2)

    ax2 = fig.add_subplot(122)
    coverage_ul.plot(ax=ax2)
    ax2.set_yscale("linear")

    ref_val = norm.cdf(result.n_sigma_ul)
    ref_min = ref_val * (1 - nsim ** -0.5)
    ref_max = ref_val * (1 + nsim ** -0.5)

    ax2.axhline(ref_val, color='k')
    ax2.axhspan(ref_min, ref_max, color='b', alpha=0.2)

    plt.savefig(filename)

if __name__ == "__main__":
    cli()
