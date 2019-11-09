import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.modeling.models import (
    SkyModel,
    ExpCutoffPowerLawSpectralModel,
    PointSpatialModel,
)
from gammapy.spectrum import FluxPointsEstimator
from gammapy.modeling import Fit
from gammapy.data import DataStore
from gammapy.maps import MapAxis, WcsGeom
from gammapy.cube import MapDatasetMaker


N_OBS = 20
OBS_ID = 110380


def run_benchmark():
    data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")
    obs_ids = OBS_ID * np.ones(N_OBS)
    observations = data_store.get_observations(obs_ids)

    energy_axis = MapAxis.from_bounds(
        0.1, 10, nbin=10, unit="TeV", name="energy", interp="log"
    )
    geom = WcsGeom.create(
        skydir=(0, 0),
        binsz=0.02,
        width=(10, 8),
        coordsys="GAL",
        proj="CAR",
        axes=[energy_axis],
    )

    src_pos = SkyCoord(0, 0, unit="deg", frame="galactic")
    offset_max = 4 * u.deg

    spatial_model = PointSpatialModel(
        lon_0="-0.05 deg", lat_0="-0.05 deg", frame="galactic"
    )
    spectral_model = ExpCutoffPowerLawSpectralModel(
        index=2,
        amplitude=3e-12 * u.Unit("cm-2 s-1 TeV-1"),
        reference=1.0 * u.TeV,
        lambda_=0.1 / u.TeV,
    )
    model = SkyModel(
        spatial_model=spatial_model, spectral_model=spectral_model, name="gc-source"
    )

    datasets = []
    for obs in observations:
        maker = MapDatasetMaker(geom=geom, offset_max=offset_max)
        dataset = maker.run(obs)
        dataset.edisp = dataset.edisp.get_energy_dispersion(
            position=src_pos, e_reco=energy_axis.edges
        )
        dataset.psf = dataset.psf.get_psf_kernel(
            position=src_pos, geom=geom, max_radius="0.3 deg"
        )
        dataset.model = model
        datasets.append(dataset)

    fit = Fit(datasets)
    result = fit.run()

    e_edges = [0.3, 1, 3, 10] * u.TeV
    fpe = FluxPointsEstimator(datasets=datasets, e_edges=e_edges, source="gc-source")

    fpe.run()


if __name__ == "__main__":
    run_benchmark()
