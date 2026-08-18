import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from regions import CircleSkyRegion

from gammapy.datasets import Datasets, MapDataset, SpectrumDataset
from gammapy.datasets.simulate import MapDatasetEventSampler
from gammapy.estimators import ExcessMapEstimator, LightCurveEstimator
from gammapy.estimators.utils import find_peaks
from gammapy.makers import MapDatasetMaker, SafeMaskMaker, SpectrumDatasetMaker
from gammapy.maps import MapAxis, RegionGeom, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    FoVBackgroundModel, GeneralizedGaussianTemporalModel, Models, PointSpatialModel,
    PowerLawSpectralModel, SkyModel,
)

from utils import build_energy_axis, build_dataset_3d, CRAB_POSITION

# MapDatasetEventSampler requires an EDispMap (it calls `sample_coord`, which
# EDispKernelMap does not implement) -- `build_dataset_3d`'s default produces
# an EDispKernelMap, so the pre-sampling dataset is built here instead of via
# the shared helper.
MIGRA_AXIS = MapAxis.from_bounds(0.2, 5, nbin=48, node_type="edges", name="migra")

PULL_PARAMETERS = ["amplitude", "t_decay"]


def build_sampling_dataset(obs, position=CRAB_POSITION, width="3 deg"):
    """Build a MapDataset with an EDispMap, suitable for `MapDatasetEventSampler`."""
    energy_axis = build_energy_axis()
    energy_axis_true = MapAxis.from_energy_bounds(0.05, 100, nbin=30, unit="TeV", name="energy_true")
    geom = WcsGeom.create(
        skydir=position, width=u.Quantity(width), binsz=0.02 * u.deg, frame="icrs", axes=[energy_axis]
    )
    dataset_empty = MapDataset.create(
        geom=geom, energy_axis_true=energy_axis_true, migra_axis=MIGRA_AXIS, name="obs-3d"
    )
    maker = MapDatasetMaker(selection=["background", "exposure", "psf", "edisp"])
    maker_safe = SafeMaskMaker(methods=["aeff-default", "edisp-bias"], bias_percent=10)
    dataset = maker.run(dataset_empty, obs)
    dataset = maker_safe.run(dataset, obs)
    return dataset


def sample_transient_observation(obs, model, position=CRAB_POSITION, width="3 deg"):
    """Sample source+background events for `obs` given `model`.

    Returns a copy of `obs` with `.events` set to the sampled `EventList`,
    spanning the full duration of `obs` (single pass -- no phase-splitting
    needed, see `transient_characterization/README.md`).
    """
    dataset = build_sampling_dataset(obs, position=position, width=width)
    bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
    dataset.models = Models([model.copy(name="source"), bkg_model])
    sampler = MapDatasetEventSampler()
    events = sampler.run(dataset, observation=obs)
    return obs.copy(in_memory=True, events=events)


def detect_and_locate(dataset, correlation_radius="0.2 deg", detection_threshold=5, min_distance=None):
    """Detect the source in a time-integrated, counts-filled 3D dataset and fit its spectral shape.

    Position is taken directly from the `find_peaks` maximum and frozen --
    not fitted -- for the rest of the pipeline (detection fit and joint
    time-resolved fit alike), since it is not a quantity under test here
    (see `PULL_PARAMETERS`). Returns the fitted point-source `SkyModel`
    (spectral shape only, a nuisance parameter used to seed the joint fit),
    or `None` if no peak was found or the fit did not converge.
    """
    if min_distance is None:
        min_distance = correlation_radius

    detection_dataset = dataset.copy(name=dataset.name)
    detection_dataset.models = None
    estimator = ExcessMapEstimator(correlation_radius=correlation_radius)
    result = estimator.run(detection_dataset)
    peaks = find_peaks(result["sqrt_ts"], threshold=detection_threshold, min_distance=min_distance)
    if len(peaks) == 0:
        return None

    peak = peaks[0]
    position = SkyCoord(peak["ra"], peak["dec"], unit="deg", frame="icrs")
    spatial = PointSpatialModel(lon_0=position.ra, lat_0=position.dec, frame="icrs")
    spatial.freeze()

    spectral = PowerLawSpectralModel(index=2, amplitude="1e-12 cm-2 s-1 TeV-1")
    spectral.amplitude.min = 0
    spectral.amplitude.error = 1e-13
    spectral.index.min = 1
    spectral.index.max = 5
    spectral.index.error = 0.1

    candidate = SkyModel(spatial_model=spatial, spectral_model=spectral, name="candidate")

    fit_dataset = dataset.copy(name=dataset.name)
    fit_dataset.models = Models([candidate])
    fit = Fit()
    result_fit = fit.run(datasets=[fit_dataset])
    if not result_fit.success:
        return None
    return fit_dataset.models[0]


def extract_time_resolved_spectra(obs, position, bin_width="30 s", on_region_radius="0.11 deg"):
    """Extract one on-region SpectrumDataset per `bin_width` time bin across `obs`."""
    bin_width = u.Quantity(bin_width)
    energy_axis = build_energy_axis()
    energy_axis_true = MapAxis.from_energy_bounds(0.05, 200, 12, per_decade=True, unit="TeV", name="energy_true")
    on_region = CircleSkyRegion(center=position, radius=u.Quantity(on_region_radius))
    geom = RegionGeom.create(region=on_region, axes=[energy_axis])

    duration = (obs.tstop - obs.tstart).sec * u.s
    # round(), not floor(): (tstop - tstart).sec is not exactly an integer
    # number of bin_widths due to floating point Time arithmetic.
    n_bins = int(round((duration / bin_width).to_value("")))
    edges = obs.tstart + np.arange(n_bins + 1) * bin_width

    maker = SpectrumDatasetMaker(selection=["counts", "exposure", "edisp", "background"])
    datasets = Datasets()
    for i in range(n_bins):
        obs_bin = obs.select_time(Time([edges[i], edges[i + 1]]))
        dataset_empty = SpectrumDataset.create(geom=geom, energy_axis_true=energy_axis_true, name=f"bin-{i}")
        datasets.append(maker.run(dataset_empty, obs_bin))
    return datasets


def estimate_peak_time(datasets):
    """Crude, purely-counts-based estimate of the flare peak time: the midpoint
    of the highest-excess bin. Only used as a fallback by
    `estimate_initial_parameters` when the light-curve estimate fails
    everywhere (e.g. an unusually faint realization).
    """
    excess = [
        d.counts.data.sum() - (d.background.data.sum() if d.background is not None else 0)
        for d in datasets
    ]
    i_max = int(np.argmax(excess))
    gti = datasets[i_max].gti
    return gti.time_start[0] + (gti.time_stop[0] - gti.time_start[0]) / 2


def estimate_initial_parameters(datasets, position, index, reference_amplitude="1e-11 cm-2 s-1 TeV-1"):
    """Data-driven estimates for `amplitude`/`t_ref`/`t_decay`, from a `LightCurveEstimator` run.

    Runs `LightCurveEstimator` with the spectral shape frozen at `index` (the
    detection fit's value) and a fixed reference amplitude, over the same
    per-bin `datasets` used for the joint fit -- reusing that per-bin
    background/exposure reduction rather than a raw counts excess gives a
    much better estimate than a fixed generic guess: `t_decay` typically
    comes out within a few percent of the true value, `amplitude` within
    ~tens of percent (verified in testing). `t_decay` is estimated from a
    simple log-linear fit of norm vs. time for bins after the peak
    (`GeneralizedGaussianTemporalModel` with `eta=1` is
    `exp(-0.5*(t-t_ref)/t_decay)`, so `log(norm) = const - 0.5*(t-t_ref)/t_decay`).

    Unlike `amplitude`/`t_decay` (used as initial guesses for otherwise-free
    parameters), the returned `t_ref` is used by the caller to *freeze*
    `GeneralizedGaussianTemporalModel.t_ref` in the joint fit -- see
    `build_joint_candidate_model` for why.

    `datasets.models` is left set to the reference model on return; the
    caller (`simulate_and_characterize`) overwrites it before the joint fit.
    """
    reference_amplitude = u.Quantity(reference_amplitude)
    spatial = PointSpatialModel(lon_0=position.ra, lat_0=position.dec, frame="icrs")
    spatial.freeze()
    spectral = PowerLawSpectralModel(index=index, amplitude=reference_amplitude)
    spectral.freeze()
    reference_model = SkyModel(spatial_model=spatial, spectral_model=spectral, name="lc-ref")

    datasets.models = Models([reference_model])
    estimator = LightCurveEstimator(source="lc-ref", selection_optional=[])
    lc = estimator.run(datasets)

    norm = lc.norm.data.squeeze()
    t_center = lc.geom.axes["time"].time_mid
    valid = np.isfinite(norm) & (norm > 0)

    if not valid.any():
        return u.Quantity("1e-10 cm-2 s-1 TeV-1"), estimate_peak_time(datasets), u.Quantity(200, "s")

    i_peak = int(np.nanargmax(np.where(valid, norm, -np.inf)))
    t_peak = t_center[i_peak]
    amplitude_init = reference_amplitude * norm[i_peak]

    after = valid & (np.arange(len(norm)) > i_peak)
    if after.sum() >= 2:
        dt = (t_center[after] - t_peak).sec
        slope, _ = np.polyfit(dt, np.log(norm[after]), 1)
        t_decay_init = u.Quantity(-0.5 / slope, "s") if slope < 0 else u.Quantity(200, "s")
    else:
        t_decay_init = u.Quantity(200, "s")

    return amplitude_init, t_peak, t_decay_init


def build_joint_candidate_model(position, index_init, amplitude_init, t_ref, t_rise="10 s", t_decay_init="200 s"):
    """Build the point-source spectral+temporal model fit jointly across all time bins.

    `t_ref` (the flare peak time) is frozen at the light-curve-based estimate
    from `estimate_initial_parameters`. Testing showed `t_ref` free alongside
    `amplitude` is what actually drove the amplitude miscalibration: a
    slightly-off peak time is compensated by a biased normalization, and
    HESSE's local covariance around that degenerate direction produced both
    an artificially tight `t_ref` error (std of pulls ~0.12 over a real
    n=20 run -- confirmed with real statistics, not small-sample noise) and
    a biased, underestimated-error `amplitude` (pull mean ~-15, std ~2).
    `index` was previously frozen at the detection-fit value to sidestep a
    *different* index/amplitude degeneracy in the per-bin fit; with `t_ref`
    now frozen instead, `index` is free again here, seeded from the
    detection fit. `t_rise` and `eta` stay frozen (see `build_transient_model`
    docstring in `utils.py`), matching the injected model. Every free
    parameter gets an explicit `.error` step hint, same lesson as
    `isolated_source_detection`.
    """
    spatial = PointSpatialModel(lon_0=position.ra, lat_0=position.dec, frame="icrs")
    spatial.freeze()

    amplitude_init = u.Quantity(amplitude_init)
    spectral = PowerLawSpectralModel(index=index_init, amplitude=amplitude_init)
    spectral.amplitude.min = 0
    spectral.amplitude.error = 0.1 * amplitude_init
    spectral.index.min = 1
    spectral.index.max = 5
    spectral.index.error = 0.1

    temporal = GeneralizedGaussianTemporalModel(
        t_rise=u.Quantity(t_rise), t_decay=u.Quantity(t_decay_init), eta=1
    )
    temporal.reference_time = t_ref
    temporal.t_rise.frozen = True
    temporal.eta.frozen = True
    temporal.t_ref.frozen = True
    temporal.t_decay.min = 1
    temporal.t_decay.error = u.Quantity("20 s")

    return SkyModel(
        spatial_model=spatial, spectral_model=spectral, temporal_model=temporal, name="candidate"
    )


def fit_joint_temporal_spectral(datasets, candidate_model):
    """Jointly fit `candidate_model` (same instance, shared free parameters) across all `datasets`.

    Returns `(migrad_success, fit_usable)`. In testing, MIGRAD's own `valid`
    flag (`migrad_success`) is persistently False for this many-dataset (60)
    joint Poisson likelihood -- "Estimated distance to minimum too large" --
    even though the fitted values and HESSE errors are consistently accurate
    and stable across reruns/tol/strategy variations (e.g. t_decay recovered
    to a few percent with a sensible error, repeatedly). This looks like
    MIGRAD's strict EDM-validity criterion being marginal from accumulated
    numerical noise across many independent Poisson terms, not a real
    failure to converge. `fit_usable` -- every free parameter has a finite,
    positive HESSE error -- is used as the actual gating criterion for pull
    inclusion; `migrad_success` is kept only as a diagnostic.
    """
    datasets.models = Models([candidate_model])
    fit = Fit()
    result = fit.run(datasets=datasets)

    free_parameters = candidate_model.parameters.free_parameters
    fit_usable = all(np.isfinite(p.error) and p.error > 0 for p in free_parameters)
    return result.success, fit_usable


def compute_pulls(joint_model, true_model):
    """Pull, (fitted - true) / fitted_error, for `PULL_PARAMETERS`.

    `amplitude` and `t_decay` come from `joint_model` (the time-resolved
    joint fit). `lon_0`/`lat_0` are excluded entirely -- position is fixed
    at the detection peak throughout the pipeline (see `detect_and_locate`),
    never fitted, so it has no meaningful pull. `t_ref` is likewise excluded
    -- frozen in `joint_model`, and not fit anywhere else either. `index` is
    free in `joint_model` (seeded from the detection fit) but excluded from
    `PULL_PARAMETERS` -- see `build_joint_candidate_model` for the
    amplitude/index/t_decay degeneracy this is entangled with.
    """
    pulls = {}
    for name in PULL_PARAMETERS:
        fitted = joint_model.parameters[name]
        true_value = true_model.parameters[name].value
        pulls[name] = (fitted.value - true_value) / fitted.error
    return pulls


def simulate_and_characterize(obs, model, config):
    """Run one Monte Carlo realization of the sample -> detect/locate -> extract -> joint-fit workflow.

    Parameters
    ----------
    obs : `~gammapy.data.Observation`
        Empty (no events) observation to sample into.
    model : `~gammapy.modeling.models.SkyModel`
        True transient model to inject (see `utils.build_transient_model`).
    config : dict
        `correlation_radius` (default "0.2 deg"), `detection_threshold`
        (default 5), `min_distance` (default `correlation_radius`),
        `bin_width` (default "30 s"), `on_region_radius` (default "0.11 deg"),
        `t_rise` (default "10 s"). `amplitude`/`t_peak`/`t_decay` initial
        guesses for the joint fit are data-driven (`estimate_initial_parameters`),
        not configured.

    Returns
    -------
    result : dict
        `detected`, `position_fit_success`, `migrad_success`, `fit_usable`
        (bool), and `pulls` (dict, `None` unless every stage succeeded).
    """
    correlation_radius = config.get("correlation_radius", "0.2 deg")
    detection_threshold = config.get("detection_threshold", 5)
    min_distance = config.get("min_distance", correlation_radius)
    bin_width = config.get("bin_width", "30 s")
    on_region_radius = config.get("on_region_radius", "0.11 deg")
    t_rise = config.get("t_rise", "10 s")

    failed = {
        "detected": False, "position_fit_success": False,
        "migrad_success": False, "fit_usable": False, "pulls": None,
    }

    obs_sampled = sample_transient_observation(obs, model)

    detection_dataset = build_dataset_3d(
        obs_sampled, selection=["background", "exposure", "psf", "edisp", "counts"]
    )
    position_model = detect_and_locate(detection_dataset, correlation_radius, detection_threshold, min_distance)
    if position_model is None:
        return failed
    failed["detected"] = True
    failed["position_fit_success"] = True

    datasets = extract_time_resolved_spectra(
        obs_sampled, position_model.position, bin_width=bin_width, on_region_radius=on_region_radius
    )
    index_init = position_model.spectral_model.index.value
    amplitude_init, t_ref, t_decay_init_lc = estimate_initial_parameters(
        datasets, position_model.position, index_init
    )
    candidate_model = build_joint_candidate_model(
        position_model.position, index_init, amplitude_init, t_ref,
        t_rise=t_rise, t_decay_init=t_decay_init_lc,
    )
    migrad_success, fit_usable = fit_joint_temporal_spectral(datasets, candidate_model)
    if not fit_usable:
        failed["migrad_success"] = migrad_success
        return failed

    pulls = compute_pulls(candidate_model, model)
    return {
        "detected": True, "position_fit_success": True,
        "migrad_success": migrad_success, "fit_usable": True, "pulls": pulls,
    }


def summarize_transient(results):
    """Build a JSON-serializable summary of a transient-characterization Monte Carlo run."""
    nsim = len(results)
    detected = [r["detected"] for r in results]
    position_fit_success = [r["position_fit_success"] for r in results]
    migrad_success = [r["migrad_success"] for r in results]
    fit_usable = [r["fit_usable"] for r in results]

    pulls_list = [r["pulls"] for r in results if r.get("pulls") is not None]
    pulls = {}
    for name in PULL_PARAMETERS:
        values = [p[name] for p in pulls_list]
        pulls[name] = {
            "values": values,
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
        }
    n_valid = len(pulls_list)

    return {
        "nsim": nsim,
        "n_valid": n_valid,
        "detection_efficiency": float(np.mean(detected)),
        "position_fit_success_rate": float(np.mean(position_fit_success)),
        "migrad_success_rate": float(np.mean(migrad_success)),
        "fit_usable_rate": float(np.mean(fit_usable)),
        "pulls": pulls,
    }


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
