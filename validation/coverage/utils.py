import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from scipy.stats import norm
import matplotlib.pyplot as plt

from gammapy.data import FixedPointingInfo, Observation, observatory_locations
from gammapy.datasets import Datasets, SpectrumDataset, SpectrumDatasetOnOff
from gammapy.estimators import FluxPointsEstimator, FluxPoints
from gammapy.estimators.flux import FluxEstimator
from gammapy.irf import load_irf_dict_from_file
from gammapy.makers import SpectrumDatasetMaker
from gammapy.maps import MapAxis, RegionGeom, LabelMapAxis, Map
from gammapy.modeling.models import SkyModel, create_crab_spectral_model


def build_observation(livetime="1 h"):
    """Build an observation using CTAO south Prod 5 IRFs.

    Pointing is assumed to be fixed on the GC direction.

    Parameters
    ----------
    livetime : `~astropy.quantity`, optional
        The observation livetime. Default is 1h.
    """
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
    """Build dataset.

    TODO: use configuration to build dataset
    """
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
        dataset=dataset, acceptance=1, acceptance_off=10,
        name=dataset.name   # keeping the same name is necessary to keep flux points geometries aligned
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

def fake_and_apply_fpe(dataset, model, fpe_config):
    fpe = FluxPointsEstimator(**fpe_config)
    dataset_on_off = fake_dataset(dataset, model)
    fp = fpe.run([dataset_on_off])
    return fp

def fake_and_apply_fe(dataset, model, fe_config):
    dataset_on_off = fake_dataset(dataset, model)
    return FluxEstimator(**fe_config).run([dataset_on_off])

def compute_ci_coverage(result_fp, use_covar=False, remove_ul=False):
    energy_axis = result_fp.geom.axes["energy"]

    weights = ~result_fp.is_ul * 1.0 if remove_ul else np.ones(result_fp.is_ul.data.shape)

    if use_covar:
        ci_min = result_fp.norm - result_fp.norm_err
        ci_max = result_fp.norm + result_fp.norm_err
    else:
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
    coverage_ci = compute_ci_coverage(result, False, False)
    coverage_covar_ci = compute_ci_coverage(result, True, False)
    coverage_ul = compute_ul_coverage(result)

    nsim = result.geom.axes['index'].nbin

    fig = plt.figure(figsize=(12, 4))
    ax0 = fig.add_subplot(131)
    ax0.set_title("err")
    coverage_covar_ci.plot(ax=ax0, color="k")
    ax0.set_yscale("linear")

    ref_val = 1 - 2 * norm.sf(result.n_sigma)
    ref_min = ref_val * (1 - nsim ** -0.5)
    ref_max = ref_val * (1 + nsim ** -0.5)
    ax0.axhline(ref_val, color='k')
    ax0.axhspan(ref_min, ref_max, color='b', alpha=0.2)

    ax1 = fig.add_subplot(132)
    ax1.set_title("errn-errp")
    coverage_ci.plot(ax=ax1, color="k")
    ax1.set_yscale("linear")

    ref_val = 1 - 2 * norm.sf(result.n_sigma)
    ref_min = ref_val * (1 - nsim ** -0.5)
    ref_max = ref_val * (1 + nsim ** -0.5)
    ax1.axhline(ref_val, color='k')
    ax1.axhspan(ref_min, ref_max, color='b', alpha=0.2)

    ax2 = fig.add_subplot(133)
    ax2.set_title("UL")
    coverage_ul.plot(ax=ax2)
    ax2.set_yscale("linear")

    ref_val = norm.cdf(result.n_sigma_ul)
    ref_min = ref_val * (1 - nsim ** -0.5)
    ref_max = ref_val * (1 + nsim ** -0.5)

    ax2.axhline(ref_val, color='k')
    ax2.axhspan(ref_min, ref_max, color='b', alpha=0.2)

    plt.savefig(filename)