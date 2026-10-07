import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import Angle
from regions import CircleSkyRegion
from scipy.stats import norm

from gammapy.datasets import SpectrumDataset
from gammapy.estimators import FluxPointsEstimator
from gammapy.estimators.flux import FluxEstimator
from gammapy.maps import Map

from utils import fake_dataset_3d, fake_dataset_on_off


def reduce_dimensionality_flux_points(flux_points):
    flux_points = flux_points.copy()
    for name, quantity in flux_points._data.items():
        if "dataset" in quantity.geom.axes.names:
            map_obj = quantity.sum_over_axes(axes_names=["dataset"], keepdims=False)
            flux_points._data[name] = map_obj
    return flux_points

def fake_and_apply_fpe(dataset, model, fpe_config):
    fpe = FluxPointsEstimator(**fpe_config)
    if isinstance(dataset, SpectrumDataset):
        faked_dataset = fake_dataset_on_off(dataset, model)
    else:
        faked_dataset = fake_dataset_3d(dataset, model)
    fp = fpe.run([faked_dataset])
    return fp


def fake_and_apply_fe(dataset, model, fe_config):
    fe = FluxEstimator(**fe_config)
    if isinstance(dataset, SpectrumDataset):
        faked_dataset = fake_dataset_on_off(dataset, model)
    else:
        faked_dataset = fake_dataset_3d(dataset, model)

    result = fe.run([faked_dataset])

    if dataset.tag == "MapDataset":
        # Project to a spectrum dataset to compute npred_excess as a 1-element array,
        on_region = CircleSkyRegion(center=model.spatial_model.position, radius=Angle("0.11 deg"))
        dataset_spec = dataset.to_spectrum_dataset(on_region=on_region, name="obs-3d-spec")
        excess = (dataset_spec.counts.data - dataset_spec.npred_background().data).sum()
        # TODO: Add info on result

    return result

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


def _coverage_per_energy_bin(coverage_map):
    """Reduce a coverage map to one fraction per energy bin.

    For 1d results the map already holds a single spatial pixel; for 3d
    results this additionally averages over the sky positions in the map,
    since the coverage test cares about overall statistical coverage rather
    than per-pixel differences.
    """
    data = coverage_map.data
    return np.nanmean(data.reshape(data.shape[0], -1), axis=1)


def summarize_coverage(result, n_sigma, n_sigma_ul):
    """Build a JSON-serializable summary of a coverage simulation result.

    Used to persist coverage fractions per energy bin, together with their
    nominal reference value, so they can be checked by an automated test
    without needing to re-run the Monte Carlo simulation.
    """
    energy_axis = result.geom.axes["energy"]
    nsim = result.geom.axes["index"].nbin

    coverage_ci = _coverage_per_energy_bin(compute_ci_coverage(result, use_covar=False, remove_ul=False))
    coverage_ci_covar = _coverage_per_energy_bin(compute_ci_coverage(result, use_covar=True, remove_ul=False))
    coverage_ul = _coverage_per_energy_bin(compute_ul_coverage(result))

    return {
        "nsim": int(nsim),
        "n_sigma": float(n_sigma),
        "n_sigma_ul": float(n_sigma_ul),
        "energy_edges_tev": energy_axis.edges.to_value("TeV").tolist(),
        "coverage_ci": coverage_ci.tolist(),
        "coverage_ci_covar": coverage_ci_covar.tolist(),
        "coverage_ul": coverage_ul.tolist(),
        "ref_ci": float(1 - 2 * norm.sf(n_sigma)),
        "ref_ul": float(norm.cdf(n_sigma_ul)),
    }

def create_sensitivity_figure(table, n_sigma, sensitivity_amplitude, filename):
    x = table["ref_amplitude"].quantity
    sqrt_ts = np.asarray(table["sqrt_ts"])
    n_samples = sqrt_ts.shape[1]
    p16, median, p84 = np.percentile(sqrt_ts, [16, 50, 84], axis=1)
    # Approximate the standard error on the median from the 16-84% half-width,
    # scaled by sqrt(n_samples), as already done for coverage bands elsewhere.
    median_err = (p84 - p16) / 2 * n_samples ** -0.5

    fig, ax = plt.subplots()
    ax.fill_between(x.value, p16, p84, alpha=0.3, color="C0", label="16%-84% containment")
    ax.fill_between(
        x.value, median - median_err, median + median_err, alpha=0.6, color="C1",
        label="median uncertainty"
    )
    ax.plot(x.value, median, color="C0", marker="o", label="median")
    ax.axhline(n_sigma, color="k", linestyle="--", label=f"{n_sigma}"+r"$\sigma$")
    ax.axvline(
        sensitivity_amplitude.to_value(x.unit), color="r", linestyle="--", label=f"Asimov sensitivity ({n_sigma}"+r"$\sigma$)"
    )

    ax.set_xscale("log")
    ax.set_xlabel(f"Reference amplitude ({x.unit})")
    ax.set_ylabel(r"$\sqrt{TS}$")
    ax.legend()

    fig.savefig(filename)
    plt.close(fig)


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
