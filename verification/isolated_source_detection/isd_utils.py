import numpy as np
from astropy.coordinates import SkyCoord

from gammapy.estimators import ExcessMapEstimator
from gammapy.estimators.utils import find_peaks
from gammapy.modeling import Fit
from gammapy.modeling.models import GaussianSpatialModel, Models, PowerLawSpectralModel, SkyModel

from utils import fake_dataset_3d


def detect_peak(dataset, correlation_radius, detection_threshold, min_distance):
    """Run the ExcessMapEstimator on a source-free dataset and return the brightest peak, or None."""
    estimator = ExcessMapEstimator(correlation_radius=correlation_radius, sum_over_energy_groups=True)
    result = estimator.run(dataset)
    peaks = find_peaks(result["sqrt_ts"], threshold=detection_threshold, min_distance=min_distance)
    if len(peaks) == 0:
        return None
    return peaks[0]


def build_candidate_model(peak, sigma_init="0.1 deg"):
    """Build a free-parameter source model seeded at a detected peak position."""
    position = SkyCoord(peak["ra"], peak["dec"], unit="deg", frame="icrs")
    spatial = GaussianSpatialModel(
        lon_0=position.ra, lat_0=position.dec, sigma=sigma_init, frame="icrs"
    )
    spectral = PowerLawSpectralModel(index=2, amplitude="1e-12 cm-2 s-1 TeV-1")
    return SkyModel(spatial_model=spatial, spectral_model=spectral, name="candidate")


def fit_candidate(dataset, candidate_model):
    """Fit `candidate_model` (position, sigma, amplitude, index all free) against `dataset`."""
    dataset = dataset.copy(name=dataset.name)
    dataset.models = Models([candidate_model])
    fit = Fit()
    result = fit.run(datasets=[dataset])
    return dataset, result.success


def residual_stats(dataset, correlation_radius):
    """Mean/std of the residual significance map, over unmasked pixels."""
    estimator = ExcessMapEstimator(correlation_radius=correlation_radius, sum_over_energy_groups=True)
    result = estimator.run(dataset)
    sqrt_ts = result["sqrt_ts"].data
    return float(np.nanmean(sqrt_ts)), float(np.nanstd(sqrt_ts))


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
        `min_distance` (default `correlation_radius`), `sigma_init` (default "0.1 deg").

    Returns
    -------
    result : dict
        `detected` and `fit_success` (bool), and `mean`/`std` of the residual
        sqrt_ts map -- `None` if detection or the fit did not succeed.
    """
    correlation_radius = config["correlation_radius"]
    detection_threshold = config.get("detection_threshold", 5)
    min_distance = config.get("min_distance", correlation_radius)

    faked_dataset = fake_dataset_3d(dataset, model)

    detection_dataset = faked_dataset.copy(name=faked_dataset.name)
    detection_dataset.models = None
    peak = detect_peak(detection_dataset, correlation_radius, detection_threshold, min_distance)

    if peak is None:
        return {"detected": False, "fit_success": False, "mean": None, "std": None}

    candidate_model = build_candidate_model(peak, sigma_init=config.get("sigma_init", "0.1 deg"))
    fitted_dataset, fit_success = fit_candidate(faked_dataset, candidate_model)

    if not fit_success:
        return {"detected": True, "fit_success": False, "mean": None, "std": None}

    mean, std = residual_stats(fitted_dataset, correlation_radius)
    return {"detected": True, "fit_success": True, "mean": mean, "std": std}
