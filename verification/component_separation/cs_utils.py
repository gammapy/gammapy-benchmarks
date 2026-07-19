import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord

from gammapy.datasets import Datasets
from gammapy.estimators import ExcessMapEstimator
from gammapy.estimators.points import FluxCollectionEstimator
from gammapy.estimators.utils import find_peaks
from gammapy.modeling import Fit
from gammapy.modeling.selection import NestedModelSelection
from gammapy.modeling.models import GaussianSpatialModel, Models, PowerLawSpectralModel, SkyModel

from utils import build_energy_axis, CRAB_POSITION

# Crab's hess_pl differential flux at 1 TeV -- same "crab_fraction of Crab's 1 TeV
# flux" convention as `utils.build_model`/`build_extended_model`, generalized here
# to an arbitrary spectral index (those scale a fixed Crab reference shape instead).
CRAB_1TEV_FLUX = 3.45e-11 * u.Unit("cm-2 s-1 TeV-1")

PULL_PARAMETERS = ["lon_0", "lat_0", "sigma", "index", "amplitude"]


def build_component_model(position, sigma, crab_fraction, index, name):
    """Build a single Gaussian+PowerLaw component model with an explicit spectral index.

    Unlike `utils.build_extended_model` (which scales a fixed Crab reference
    spectral shape and can't take an arbitrary index), amplitude here is
    `crab_fraction * CRAB_1TEV_FLUX` directly -- same normalization
    convention, generalized to an index passed in explicitly.
    """
    spatial = GaussianSpatialModel(
        lon_0=position.ra, lat_0=position.dec, sigma=Angle(sigma), frame="icrs"
    )
    spatial.freeze()
    spectral = PowerLawSpectralModel(index=index, amplitude=crab_fraction * CRAB_1TEV_FLUX)
    return SkyModel(spatial_model=spatial, spectral_model=spectral, name=name)


def build_scene_model(
    position=CRAB_POSITION,
    big_sigma="0.5 deg", big_crab_fraction=0.2, big_index=2.5,
    small_sigma="0.05 deg", small_crab_fraction=0.01, small_index=1.8,
    separation="0.2 deg", position_angle="0 deg",
):
    """Build the two-component scene model: a big, moderately extended source
    ("big") and a smaller, fainter one ("small") offset from its center by
    `separation`. Defaults match the morphology-extraction scenario in
    `verification/CLAUDE.md`'s `component_separation` section.
    """
    small_position = position.directional_offset_by(
        position_angle=u.Quantity(position_angle), separation=u.Quantity(separation)
    )
    big_model = build_component_model(position, big_sigma, big_crab_fraction, big_index, name="big")
    small_model = build_component_model(
        small_position, small_sigma, small_crab_fraction, small_index, name="small"
    )
    return Models([big_model, small_model])


def fake_scene_dataset_3d(dataset, models):
    """Fake counts for a multi-component `Models` list into a copy of `dataset`.

    Local replacement for `utils.fake_dataset_3d`, which only accepts a single
    model and renames it to `"source"` -- that would collapse this use case's
    `"big"`/`"small"` names into one (see `verification/CLAUDE.md`'s
    `component_separation` step 1 finding). Attaches `models` as-is (each
    component deep-copied, names preserved) instead.

    Note: `SkyModel.copy()` generates a *random* new name unless `name` is
    passed explicitly -- it does not default to preserving the original.
    """
    dataset = dataset.copy(name=dataset.name)
    dataset.models = Models([m.copy(name=m.name) for m in models])
    dataset.fake()
    return dataset


def detect_peak(dataset, correlation_radius, detection_threshold, min_distance):
    """Run the ExcessMapEstimator and return the brightest peak, or None.

    If `dataset.models` is set (e.g. to the fitted big source), the excess
    map is computed net of that model's predicted counts -- see
    `detect_and_characterize_small_source`, which relies on this to find the
    small source in the big source's residual.
    """
    estimator = ExcessMapEstimator(correlation_radius=correlation_radius)
    result = estimator.run(dataset)
    peaks = find_peaks(result["sqrt_ts"], threshold=detection_threshold, min_distance=min_distance)
    if len(peaks) == 0:
        return None
    return peaks[0]


def _build_candidate_model(peak, sigma_init, name):
    """Free-parameter Gaussian+PowerLaw model seeded at a detected peak position.

    Every free parameter gets an explicit initial step (`error`) -- without
    it MIGRAD's default step-size guess is badly scaled across parameters as
    different as amplitude (~1e-12) and lon_0 (~1e2 deg), same lesson as
    `isolated_source_detection/isd_utils.py`'s `build_candidate_model`.
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

    return SkyModel(spatial_model=spatial, spectral_model=spectral, name=name)


def fit_big_source(dataset, correlation_radius="0.3 deg", detection_threshold=5,
                    min_distance=None, sigma_init="0.5 deg"):
    """Detect and fit the big source alone against `dataset`.

    ExcessMapEstimator + find_peaks seeds the position, then a free
    Gaussian+PowerLaw fit (position, sigma, amplitude, index all free)
    against `dataset` -- the small source is not yet accounted for, so its
    counts are absorbed as noise in this step.

    Returns
    -------
    fitted_dataset : `~gammapy.datasets.MapDataset` or None
        Copy of `dataset` with the fitted big-source model attached, or
        `None` if no peak was found.
    big_model : `~gammapy.modeling.models.SkyModel` or None
        The fitted big-source model (named `"big"`), or `None` if no peak
        was found.
    fit_success : bool
        Whether the fit converged. `False` (with `big_model` still returned)
        if no peak was found -- callers should check `big_model is None`
        first.
    """
    if min_distance is None:
        min_distance = correlation_radius

    detection_dataset = dataset.copy(name=dataset.name)
    detection_dataset.models = None
    peak = detect_peak(detection_dataset, correlation_radius, detection_threshold, min_distance)
    if peak is None:
        return None, None, False

    candidate = _build_candidate_model(peak, sigma_init, name="big")
    fitted_dataset = dataset.copy(name=dataset.name)
    fitted_dataset.models = Models([candidate])
    fit = Fit()
    result = fit.run(datasets=[fitted_dataset])
    return fitted_dataset, fitted_dataset.models["big"], result.success


def detect_and_characterize_small_source(
    dataset, big_model, correlation_radius="0.1 deg", detection_threshold=5,
    min_distance=None, sigma_init="0.05 deg", n_sigma=5,
):
    """Detect the small source in the big source's residual, then jointly
    characterize both components in one step.

    `big_model` (already fit alone by `fit_big_source`) is attached alongside
    the small-source candidate and both are left free in the joint fit --
    resolving their blending is the point of this use case, not something to
    sidestep by freezing the big source. `NestedModelSelection` (the class
    backing `select_nested_models`) is used instead of a plain `Fit()`
    because its `n_sigma`-thresholded test *is* the small-source detection
    step: passing the small candidate's full free-parameter set (not just
    `amplitude`) means the alternative hypothesis's fit is already the fully
    free, jointly-fit characterization -- no separate fit call needed once
    detection succeeds. Using the class directly (rather than the
    `select_nested_models` convenience function) exposes `ts_threshold`,
    needed to tell whether detection actually succeeded (the function only
    returns the raw `ts`).

    Returns
    -------
    joint_dataset : `~gammapy.datasets.MapDataset` or None
        Copy of `dataset` with both components attached, or `None` if no
        residual peak was found.
    fit_result : `~gammapy.modeling.FitResult` or None
        The alternative hypothesis's fit result -- `.models` holds the
        jointly-fit `"big"`/`"small"` models -- or `None` unless the small
        source was detected. Returned whole (not just `.models`) so callers
        can write it directly (`fit_result.write(...)`).
    detected : bool
        Whether the small source was detected (TS above the `n_sigma`
        threshold).
    """
    if min_distance is None:
        min_distance = correlation_radius

    residual_dataset = dataset.copy(name=dataset.name)
    residual_dataset.models = Models([big_model.copy(name=big_model.name)])
    peak = detect_peak(residual_dataset, correlation_radius, detection_threshold, min_distance)
    if peak is None:
        return None, None, False

    candidate = _build_candidate_model(peak, sigma_init, name="small")

    joint_dataset = dataset.copy(name=dataset.name)
    joint_dataset.models = Models([big_model.copy(name=big_model.name), candidate])

    free_params = list(candidate.parameters.free_parameters)
    null_values = [p.value for p in free_params]
    # amplitude=0 already zeroes the candidate's contribution under the null,
    # so the other entries (lon_0, lat_0, sigma, index) are irrelevant.
    null_values[[p.name for p in free_params].index("amplitude")] = 0

    selector = NestedModelSelection(
        parameters=free_params, null_values=null_values, n_sigma=n_sigma, fit=Fit()
    )
    test_result = selector.run(Datasets([joint_dataset]))
    detected = test_result["ts"] > selector.ts_threshold
    if not detected:
        return joint_dataset, None, False

    return joint_dataset, test_result["fit_results"], True


def run_flux_collection(dataset, big_model, small_model):
    """Per-energy-bin flux points for both components, fit jointly.

    Meant to be called once, on a single reference realization -- not inside
    the per-realization Monte Carlo loop (`simulate_and_characterize`) --
    see `verification/CLAUDE.md`'s `component_separation` scientific
    workflow. `solver=Fit()` (MIGRAD/HESSE) is used instead of
    `FluxCollectionEstimator`'s default nested-sampling `Sampler`, which
    needs the `ultranest` package (not a project dependency) and would break
    the `(fitted-true)/fitted_error` pull convention used everywhere else.

    Returns
    -------
    flux_points : dict
        `{"big": FluxPoints, "small": FluxPoints}`.
    """
    dataset = dataset.copy(name=dataset.name)
    dataset.models = Models(
        [big_model.copy(name=big_model.name), small_model.copy(name=small_model.name)]
    )

    estimator = FluxCollectionEstimator(
        energy_edges=build_energy_axis().edges,
        models=dataset.models,
        solver=Fit(),
    )
    result = estimator.run(Datasets([dataset]))
    return result["flux_points"]


def compute_pulls(fitted_model, true_model):
    """Pull, (fitted - true) / fitted_error, for each parameter in `PULL_PARAMETERS`."""
    pulls = {}
    for name in PULL_PARAMETERS:
        fitted = fitted_model.parameters[name]
        true_value = true_model.parameters[name].value
        pulls[name] = (fitted.value - true_value) / fitted.error
    return pulls


def simulate_and_characterize(dataset, models, config):
    """Run one Monte Carlo realization of the fake -> fit-big -> detect/characterize-small workflow.

    Parameters
    ----------
    dataset : `~gammapy.datasets.MapDataset`
        Empty (background-only) dataset to fake counts into.
    models : `~gammapy.modeling.models.Models`
        True two-component model to inject (see `build_scene_model`).
    config : dict
        `big_correlation_radius` (default "0.3 deg"), `big_detection_threshold`
        (default 5), `big_sigma_init` (default "0.5 deg"),
        `small_correlation_radius` (default "0.1 deg"),
        `small_detection_threshold` (default 5), `small_sigma_init`
        (default "0.05 deg"), `n_sigma` (default 5, the small-source
        detection significance threshold).

    Returns
    -------
    result : dict
        `big_detected`, `big_fit_success`, `small_detected` (bool), and
        `pulls` (dict with `"big"`/`"small"` sub-dicts of per-parameter
        pulls, `None` unless every stage succeeded).
    """
    big_true, small_true = models["big"], models["small"]

    result = {
        "big_detected": False, "big_fit_success": False,
        "small_detected": False, "pulls": None,
    }

    faked_dataset = fake_scene_dataset_3d(dataset, models)

    fitted_big_dataset, big_model, big_fit_success = fit_big_source(
        faked_dataset,
        correlation_radius=config.get("big_correlation_radius", "0.3 deg"),
        detection_threshold=config.get("big_detection_threshold", 5),
        sigma_init=config.get("big_sigma_init", "0.5 deg"),
    )
    if big_model is None:
        return result
    result["big_detected"] = True
    if not big_fit_success:
        return result
    result["big_fit_success"] = True

    joint_dataset, fit_result, small_detected = detect_and_characterize_small_source(
        fitted_big_dataset, big_model,
        correlation_radius=config.get("small_correlation_radius", "0.1 deg"),
        detection_threshold=config.get("small_detection_threshold", 5),
        sigma_init=config.get("small_sigma_init", "0.05 deg"),
        n_sigma=config.get("n_sigma", 5),
    )
    if not small_detected:
        return result
    result["small_detected"] = True

    result["pulls"] = {
        "big": compute_pulls(fit_result.models["big"], big_true),
        "small": compute_pulls(fit_result.models["small"], small_true),
    }
    return result


def summarize_component_separation(results):
    """Build a JSON-serializable summary of a component-separation Monte Carlo run."""
    nsim = len(results)
    big_detected = [r["big_detected"] for r in results]
    big_fit_success = [r["big_fit_success"] for r in results]
    small_detected = [r["small_detected"] for r in results]

    pulls_list = [r["pulls"] for r in results if r.get("pulls") is not None]
    n_valid = len(pulls_list)

    pulls = {}
    for component in ("big", "small"):
        pulls[component] = {}
        for name in PULL_PARAMETERS:
            values = [p[component][name] for p in pulls_list]
            pulls[component][name] = {
                "values": values,
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values)) if values else None,
            }

    return {
        "nsim": nsim,
        "n_valid": n_valid,
        "big_detection_efficiency": float(np.mean(big_detected)),
        "big_fit_convergence_rate": float(np.mean(big_fit_success)),
        "small_detection_efficiency": float(np.mean(small_detected)),
        "pulls": pulls,
    }


def create_pull_figure(summary, filename):
    """Plot each component's pull distribution against the standard normal reference."""
    from scipy.stats import norm

    x = np.linspace(-4, 4, 200)
    components = ("big", "small")
    fig, axes = plt.subplots(
        len(components), len(PULL_PARAMETERS),
        figsize=(3 * len(PULL_PARAMETERS), 3.5 * len(components)),
    )

    for row, component in zip(axes, components):
        for ax, name in zip(row, PULL_PARAMETERS):
            values = np.array(summary["pulls"][component][name]["values"])
            ax.hist(values, bins=20, density=True, color="C0", alpha=0.7)
            ax.plot(x, norm.pdf(x), color="k", linestyle="--")
            ax.set_title(f"{component}: pull({name})")

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
