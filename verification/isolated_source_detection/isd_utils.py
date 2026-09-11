import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import Angle, SkyCoord

from gammapy.estimators import ExcessMapEstimator
from gammapy.estimators.utils import find_peaks
from gammapy.modeling import Fit
from gammapy.modeling.models import GaussianSpatialModel, Models, PowerLawSpectralModel, SkyModel

from utils import fake_dataset_3d

PULL_PARAMETERS = ["lon_0", "lat_0", "sigma", "index", "amplitude"]


def detect_peak(dataset, correlation_radius, detection_threshold, min_distance):
    """Run the ExcessMapEstimator and return the brightest peak, or None."""
    estimator = ExcessMapEstimator(correlation_radius=correlation_radius)
    result = estimator.run(dataset)
    peaks = find_peaks(result["sqrt_ts"], threshold=detection_threshold, min_distance=min_distance)
    if len(peaks) == 0:
        return None
    return peaks[0]


def build_candidate_model(peak, sigma_init="0.1 deg"):
    """Build a free-parameter source model seeded at a detected peak position.

    Amplitude and index are bounded to the physical region (positive flux,
    reasonable spectral index), and every free parameter gets an explicit
    initial step (`error`) -- without it MIGRAD's default step-size guess is
    badly scaled across parameters as different as amplitude (~1e-12) and
    lon_0 (~1e2 deg), which derails the Hessian estimate and reliably fails
    to converge on datasets wider than ~3 deg.
    """
    position = SkyCoord(peak["ra"], peak["dec"], unit="deg", frame="icrs")
    spatial = GaussianSpatialModel(
        lon_0=position.ra, lat_0=position.dec, sigma=sigma_init, frame="icrs"
    )
    spatial.lon_0.error = 0.01
    spatial.lat_0.error = 0.01
    spatial.sigma.error = 0.01

    spectral = PowerLawSpectralModel(index=2, amplitude="1e-12 cm-2 s-1 TeV-1")
    spectral.amplitude.min = 0
    spectral.amplitude.error = 1e-13
    spectral.index.min = 1
    spectral.index.max = 5
    spectral.index.error = 0.1

    return SkyModel(spatial_model=spatial, spectral_model=spectral, name="candidate")


def fit_candidate(dataset, candidate_model):
    """Fit `candidate_model` (position, sigma, amplitude, index all free) against `dataset`."""
    dataset = dataset.copy(name=dataset.name)
    dataset.models = Models([candidate_model])
    fit = Fit()
    result = fit.run(datasets=[dataset])
    return dataset, result.success


def residual_stats(dataset, correlation_radius, exclusion_position=None, exclusion_radius=None):
    """Mean/std of the residual significance map, over unmasked pixels.

    If `exclusion_position`/`exclusion_radius` are given, pixels within
    `exclusion_radius` of `exclusion_position` are dropped first. The fit
    consumes a handful of degrees of freedom concentrated in the correlated
    region around the fitted source, which measurably suppresses the local
    residual variance there (the map only has a few dozen independent
    correlated regions, so losing ~5 to the fit is not negligible) --
    excluding that region keeps the check limited to genuinely source-free,
    white-noise pixels.
    """
    estimator = ExcessMapEstimator(correlation_radius=correlation_radius)
    result = estimator.run(dataset)
    sqrt_ts = result["sqrt_ts"]
    data = sqrt_ts.data.copy()
    if exclusion_position is not None and exclusion_radius is not None:
        separation = sqrt_ts.geom.to_image().separation(exclusion_position)
        data[:, separation < Angle(exclusion_radius)] = np.nan
    return float(np.nanmean(data)), float(np.nanstd(data))


def compute_pulls(fitted_model, true_model):
    """Pull, (fitted - true) / fitted_error, for each parameter in `PULL_PARAMETERS`."""
    pulls = {}
    for name in PULL_PARAMETERS:
        fitted = fitted_model.parameters[name]
        true_value = true_model.parameters[name].value
        pulls[name] = (fitted.value - true_value) / fitted.error
    return pulls


def fake_detect_fit_recompute(dataset, model, config):
    """Run one Monte Carlo realization of the detect -> fit -> residual workflow.

    Parameters
    ----------
    dataset : `~gammapy.datasets.MapDataset`
        Empty (background-only) dataset to fake counts into.
    model : `~gammapy.modeling.models.SkyModel`
        True source model to inject.
    config : dict
        `correlation_radius`, and optionally `detection_threshold` (default 5),
        `min_distance` (default `correlation_radius`), `sigma_init` (default "0.1 deg"),
        `exclusion_radius` (default `3 * correlation_radius`) -- the region
        around the fitted source excluded from the residual mean/std.

    Returns
    -------
    result : dict
        `detected` and `fit_success` (bool), `mean`/`std` of the residual
        sqrt_ts map, and `pulls` (dict of `(fitted - true) / fitted_error`
        per `PULL_PARAMETERS`) -- `None` if detection or the fit did not
        succeed.
    """
    correlation_radius = config["correlation_radius"]
    detection_threshold = config.get("detection_threshold", 5)
    min_distance = config.get("min_distance", correlation_radius)

    faked_dataset = fake_dataset_3d(dataset, model)

    detection_dataset = faked_dataset.copy(name=faked_dataset.name)
    detection_dataset.models = None
    peak = detect_peak(detection_dataset, correlation_radius, detection_threshold, min_distance)

    if peak is None:
        return {"detected": False, "fit_success": False, "mean": None, "std": None, "pulls": None}

    candidate_model = build_candidate_model(peak, sigma_init=config.get("sigma_init", "0.1 deg"))
    fitted_dataset, fit_success = fit_candidate(faked_dataset, candidate_model)

    if not fit_success:
        return {"detected": True, "fit_success": False, "mean": None, "std": None, "pulls": None}

    exclusion_radius = config.get("exclusion_radius", 3 * Angle(correlation_radius))
    mean, std = residual_stats(
        fitted_dataset,
        correlation_radius,
        exclusion_position=fitted_dataset.models[0].position,
        exclusion_radius=exclusion_radius,
    )
    pulls = compute_pulls(fitted_dataset.models[0], model)
    return {"detected": True, "fit_success": True, "mean": mean, "std": std, "pulls": pulls}


def summarize_isd(results):
    """Build a summary of an isolated-source-detection Monte Carlo run.

    Aggregates the per-realization residual mean/std and parameter pulls
    (only over realizations where the source was detected and the fit
    converged).
    """
    nsim = len(results)
    detected = [r["detected"] for r in results]
    fit_success = [r["fit_success"] for r in results]
    means = [r["mean"] for r in results if r["mean"] is not None]
    stds = [r["std"] for r in results if r["std"] is not None]
    n_valid = len(means)

    pulls_list = [r["pulls"] for r in results if r.get("pulls") is not None]
    pulls = {}
    for name in PULL_PARAMETERS:
        values = [p[name] for p in pulls_list]
        pulls[name] = {
            "values": values,
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
        }

    return {
        "nsim": nsim,
        "n_valid": n_valid,
        "detection_efficiency": float(np.mean(detected)),
        "fit_convergence_rate": float(np.mean(fit_success)),
        "means": means,
        "stds": stds,
        "mean_of_means": float(np.mean(means)) if n_valid else None,
        "std_of_means": float(np.std(means)) if n_valid else None,
        "mean_of_stds": float(np.mean(stds)) if n_valid else None,
        "std_of_stds": float(np.std(stds)) if n_valid else None,
        "pulls": pulls,
    }


def create_residual_figure(summary, filename):
    """Plot the distribution of per-realization residual mean/std against their nominal values."""
    means = np.array(summary["means"])
    stds = np.array(summary["stds"])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))

    ax0.hist(means, bins=20, color="C0")
    ax0.axvline(0, color="k", linestyle="--")
    ax0.set_title("Residual mean(sqrt_ts)")

    ax1.hist(stds, bins=20, color="C1")
    ax1.axvline(1, color="k", linestyle="--")
    ax1.set_title("Residual std(sqrt_ts)")

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def create_pull_figure(summary, filename):
    """Plot each fitted parameter's pull distribution against the standard normal reference."""
    from scipy.stats import norm

    x = np.linspace(-4, 4, 200)
    fig, axes = plt.subplots(1, len(PULL_PARAMETERS), figsize=(3 * len(PULL_PARAMETERS), 3.5))

    for ax, name in zip(axes, PULL_PARAMETERS):
        values = np.array(summary["pulls"][name]["values"])
        ax.hist(values, bins=20, density=True, color="C0", alpha=0.7)
        ax.plot(x, norm.pdf(x), color="k", linestyle="--")
        ax.set_title(f"pull({name})")

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
