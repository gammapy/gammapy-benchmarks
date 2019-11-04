import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.cube import MapDataset
from gammapy.modeling.models import SkyModel, SkyDiffuseCube
from gammapy.modeling.models.spectral import ExpCutoffPowerLawSpectralModel
from gammapy.spectrum import FluxPointsEstimator
from gammapy.modeling.models.spatial import PointSpatialModel
from gammapy.modeling import Fit
from gammapy.data import DataStore
from gammapy.maps import MapAxis, WcsGeom
from gammapy.cube import MapDataset, MapDatasetMaker


N_OBS = 100
OBS_ID = 110380


def run_benchmark():

    data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")
    obs_ids = OBS_ID * np.ones(N_OBS)
    observations = data_store.get_observations(obs_ids)

    energy_axis = MapAxis.from_edges(
        np.logspace(-1.0, 1.0, 10), unit="TeV", name="energy", interp="log"
    )
    geom = WcsGeom.create(
        skydir=(0, 0),
        binsz=0.05,
        width=(10, 8),
        coordsys="GAL",
        proj="CAR",
        axes=[energy_axis],
    )

    stacked = MapDataset.create(geom)
    for obs in observations:
        maker = MapDatasetMaker(geom, offset_max=4.0 * u.deg)
        dataset = maker.run(obs)
        stacked.stack(dataset)

    stacked.edisp = stacked.edisp.get_energy_dispersion(
        position=SkyCoord(0, 0, unit="deg", frame="galactic"), e_reco=energy_axis.edges
    )

    stacked.psf = stacked.psf.get_psf_kernel(
        position=SkyCoord(0, 0, unit="deg", frame="galactic"),
        geom=geom,
        max_radius="0.3 deg",
    )

    spatial_model = PointSpatialModel(
        lon_0="0.01 deg", lat_0="0.01 deg", frame="galactic"
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

    stacked.model = model

    fit = Fit(stacked)
    result = fit.run(optimize_opts={"print_level": 1})

    spec = model.spectral_model

    covariance = result.parameters.covariance
    spec.parameters.covariance = covariance[2:6, 2:6]

    e_edges = [0.3, 1, 3, 10] * u.TeV
    fpe = FluxPointsEstimator(datasets=[stacked], e_edges=e_edges, source="gc-source")

    fpe.run()


if __name__ == "__main__":
    run_benchmark()
