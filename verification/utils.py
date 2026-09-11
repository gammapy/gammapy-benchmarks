import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion

from gammapy.data import FixedPointingInfo, Observation, observatory_locations
from gammapy.datasets import SpectrumDataset, SpectrumDatasetOnOff, MapDataset
from gammapy.irf import load_irf_dict_from_file
from gammapy.makers import SpectrumDatasetMaker, MapDatasetMaker, SafeMaskMaker
from gammapy.maps import MapAxis, RegionGeom, WcsGeom
from gammapy.modeling.models import (
    SkyModel, Models, create_crab_spectral_model, PointSpatialModel, GaussianSpatialModel,
    GeneralizedGaussianTemporalModel,
)

CRAB_POSITION = SkyCoord(83.233, 22.214, unit="deg", frame="icrs")

# Default transient position: 1 deg away from CRAB_POSITION, so the injected
# source is decoupled from both the map center and the pointing target used
# elsewhere in this module (`build_dataset_3d`/`build_observation` default to
# CRAB_POSITION), testing position recovery off-center rather than dead-center.
TRANSIENT_POSITION = CRAB_POSITION.directional_offset_by(position_angle=50 * u.deg, separation=1 * u.deg)

IRF_FILENAMES = {
    "south": "Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits.gz",
    "north": "Prod5-North-20deg-AverageAz-4LSTs09MSTs.180000s-v0.1.fits.gz",
}


def build_observation(livetime="1 h", offset="0.5 deg", position=CRAB_POSITION, site="south"):
    """Build an observation using CTAO Prod 5 IRFs.

    Pointing is offset from `position` by a fixed separation.

    Parameters
    ----------
    livetime : `~astropy.units.Quantity`, optional
        The observation livetime. Default is 1h.
    offset : `~astropy.units.Quantity`, optional
        Offset of the pointing from `position`. Default is 0.5 deg.
    position : `~astropy.coordinates.SkyCoord`, optional
        Sky position the observation target. Default is `CRAB_POSITION`.
    site : {"south", "north"}, optional
        CTAO site the IRFs and observatory location are taken from. Default is "south".
    """
    # Define simulation parameters parameters
    livetime = u.Quantity(livetime)
    offset = u.Quantity(offset)

    pointing_position = position.directional_offset_by(
        position_angle=0 * u.deg, separation=offset
    )

    # We want to simulate an observation pointing at a fixed position in the sky.
    # For this, we use the `FixedPointingInfo` class
    pointing = FixedPointingInfo(
        fixed_icrs=pointing_position.icrs,
    )

    irfs = load_irf_dict_from_file(
        f"$GAMMAPY_DATA/cta-caldb/{IRF_FILENAMES[site]}"
    )

    location = observatory_locations[f"ctao_{site}"]
    return Observation.create(
        pointing=pointing,
        livetime=livetime,
        irfs=irfs,
        location=location,
    )


def build_energy_axis():
    return MapAxis.from_energy_bounds(0.1, 100, 6, per_decade=True, unit="TeV")


def build_dataset_1d(obs, position=CRAB_POSITION):
    """Build a SpectrumDataset from a single Observation.

    TODO: use configuration to build dataset
    """
    # Reconstructed and true energy axis
    energy_axis = build_energy_axis()
    energy_axis_true = MapAxis.from_energy_bounds(0.05, 200, 12, per_decade=True, unit="TeV", name="energy_true")

    on_region_radius = Angle("0.11 deg")

    on_region = CircleSkyRegion(center=position, radius=on_region_radius)

    # Make the SpectrumDataset
    geom = RegionGeom.create(region=on_region, axes=[energy_axis])

    dataset_empty = SpectrumDataset.create(
        geom=geom, energy_axis_true=energy_axis_true, name="obs-0"
    )
    maker = SpectrumDatasetMaker(selection=["exposure", "edisp", "background"])

    return maker.run(dataset_empty, obs)


def build_dataset_3d(obs, position=CRAB_POSITION, width="3 deg", selection=None):
    """Build a MapDataset from a single Observation for 3D coverage tests.

    `selection` defaults to `["background", "exposure", "psf", "edisp"]`, i.e.
    no `"counts"` -- the dataset is meant to be filled via `dataset.fake()`.
    Pass `selection` explicitly (with `"counts"` included) to bin real,
    already-sampled events from `obs.events` instead.
    """
    energy_axis = build_energy_axis()
    energy_axis_true = MapAxis.from_energy_bounds(
        0.05, 100, nbin=30, unit="TeV", name="energy_true"
    )

    geom = WcsGeom.create(
        skydir=position,
        width=u.Quantity(width),
        binsz=0.02 * u.deg,
        frame="icrs",
        axes=[energy_axis],
    )
    geom_true = geom.to_image().to_cube([energy_axis_true])
    dataset_empty = MapDataset.create(
        geom=geom,
        geom_exposure=geom_true,
        name="obs-3d",
    )
    if selection is None:
        selection = ["background", "exposure", "psf", "edisp"]
    maker = MapDatasetMaker(selection=selection)
    maker_safe = SafeMaskMaker(methods=["aeff-default", "edisp-bias"], bias_percent=10)
    dataset = maker.run(dataset_empty, obs)
    dataset = maker_safe.run(dataset, obs)
    return dataset


def build_model(crab_fraction=0.1, position=CRAB_POSITION):
    spectral = create_crab_spectral_model('magic_lp')
    spectral.amplitude.value *= crab_fraction
    spatial = PointSpatialModel(lon_0=position.ra, lat_0=position.dec, frame="icrs")
    spatial.freeze()
    return SkyModel(spatial_model=spatial, spectral_model=spectral, name="source")


def build_extended_model(crab_fraction=0.1, sigma="0.1 deg", position=CRAB_POSITION):
    """Build a moderately extended source model with a power-law spectrum.

    Uses the same Crab-normalization convention as `build_model`: amplitude is
    scaled from a reference Crab power-law spectrum by `crab_fraction`.
    """
    spectral = create_crab_spectral_model('hess_pl')
    spectral.amplitude.value *= crab_fraction
    spatial = GaussianSpatialModel(
        lon_0=position.ra, lat_0=position.dec, sigma=Angle(sigma), frame="icrs"
    )
    spatial.freeze()
    return SkyModel(spatial_model=spatial, spectral_model=spectral, name="source")


def build_transient_model(t_peak, t_decay="200 s", t_rise="10 s", crab_fraction=5.0, position=TRANSIENT_POSITION):
    """Build a point-source transient model with a power-law spectrum and a
    generalized-Gaussian temporal profile.

    `eta` is fixed to 1 (a two-sided exponential) and `t_rise` is frozen to a
    short value -- both are numerical devices for a sharp, well-posed peak at
    `t_peak`, not quantities under test. `amplitude` is the *peak* flux
    (the temporal factor is 1 at `t_peak`), scaled from a reference Crab
    power-law spectrum by `crab_fraction`, same convention as `build_model`.

    Parameters
    ----------
    t_peak : `~astropy.time.Time`
        Time of the pulse's peak.
    t_decay : `~astropy.units.Quantity`, optional
        Decay time constant. Default is 200 s.
    t_rise : `~astropy.units.Quantity`, optional
        Rise time constant (frozen). Default is 10 s.
    """
    spectral = create_crab_spectral_model('hess_pl')
    spectral.amplitude.value *= crab_fraction

    spatial = PointSpatialModel(lon_0=position.ra, lat_0=position.dec, frame="icrs")
    spatial.freeze()

    temporal = GeneralizedGaussianTemporalModel(
        t_rise=u.Quantity(t_rise), t_decay=u.Quantity(t_decay), eta=1
    )
    temporal.reference_time = t_peak
    temporal.t_rise.frozen = True
    temporal.eta.frozen = True

    return SkyModel(
        spatial_model=spatial, spectral_model=spectral, temporal_model=temporal, name="source"
    )


def fake_dataset_3d(dataset, model):
    dataset = dataset.copy(name=dataset.name)
    dataset.models = Models([model.copy(name="source")])
    dataset.fake()
    return dataset


def fake_dataset_on_off(dataset, model):
    dataset_on_off = SpectrumDatasetOnOff.from_spectrum_dataset(
        dataset=dataset, acceptance=1, acceptance_off=10,
        name=dataset.name   # keeping the same name is necessary to keep flux points geometries aligned
    )
    dataset_on_off.models = model.copy(name="source")

    dataset_on_off.fake(npred_background=dataset.npred_background())
    return dataset_on_off
