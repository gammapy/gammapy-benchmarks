import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.data import DataStore
from gammapy.modeling.models import PowerLawSpectralModel, PointSpatialModel, SkyModel
from gammapy.cube import MapDatasetMaker, MapDataset
from gammapy.maps import WcsGeom, MapAxis
from gammapy.time import LightCurveEstimator

N_OBS = 20
OBS_ID = 23523


def run_benchmark():
    data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
    obs_ids = OBS_ID * np.ones(N_OBS)
    observations = data_store.get_observations(obs_ids)

    time_intervals = [(obs.tstart, obs.tstop) for obs in observations]
    target_position = SkyCoord(ra=83.63308, dec=22.01450, unit="deg")

    emin, emax = [0.7, 10] * u.TeV
    energy_axis = MapAxis.from_bounds(
        emin.value, emax.value, 10, unit="TeV", name="energy", interp="log"
    )
    geom = WcsGeom.create(
        skydir=target_position,
        binsz=0.02,
        width=(2, 2),
        coordsys="CEL",
        proj="CAR",
        axes=[energy_axis],
    )

    energy_axis_true = MapAxis.from_bounds(
        0.1, 20, 20, unit="TeV", name="energy", interp="log"
    )

    offset_max = 2 * u.deg
    spatial_model = PointSpatialModel(
        lon_0=target_position.ra, lat_0=target_position.dec, frame="icrs"
    )
    spatial_model.lon_0.frozen = True
    spatial_model.lat_0.frozen = True

    spectral_model = PowerLawSpectralModel(
        index=2.6, amplitude=2.0e-11 * u.Unit("1 / (cm2 s TeV)"), reference=1 * u.TeV
    )
    spectral_model.index.frozen = False

    sky_model = SkyModel(
        spatial_model=spatial_model, spectral_model=spectral_model, name=""
    )

    datasets = []

    maker = MapDatasetMaker(
        geom=geom, energy_axis_true=energy_axis_true, offset_max=offset_max
    )

    for time_interval in time_intervals:
        observations = observations.select_time(time_interval)

        # Proceed with further analysis only if there are observations
        # in the selected time window
        if len(observations) == 0:
            log.warning(f"No observations in time interval: {time_interval}")
            continue

        stacked = MapDataset.create(geom=geom, energy_axis_true=energy_axis_true)

        for obs in observations:
            dataset = maker.run(obs)
            stacked.stack(dataset)

        # TODO: remove once IRF maps are handled correctly in fit
        stacked.edisp = stacked.edisp.get_energy_dispersion(
            position=target_position, e_reco=energy_axis.edges
        )

        stacked.psf = stacked.psf.get_psf_kernel(
            position=target_position, geom=stacked.exposure.geom, max_radius="0.3 deg"
        )

        stacked.counts.meta["t_start"] = time_interval[0]
        stacked.counts.meta["t_stop"] = time_interval[1]
        datasets.append(stacked)

    for dataset in datasets:
        model = sky_model.copy(name="crab")
        dataset.model = model

    lc_maker = LightCurveEstimator(datasets, source="crab", reoptimize=True)

    lc = lc_maker.run(e_ref=1 * u.TeV, e_min=1.0 * u.TeV, e_max=10.0 * u.TeV)


if __name__ == "__main__":
    run_benchmark()
