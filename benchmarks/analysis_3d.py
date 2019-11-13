import numpy as np
import astropy.units as u
import time
import yaml
from astropy.coordinates import SkyCoord
from gammapy.modeling.models import (
    SkyModel,
    ExpCutoffPowerLawSpectralModel,
    PointSpatialModel,
)
from gammapy.modeling import Fit
from gammapy.spectrum import FluxPointsEstimator
from gammapy.data import DataStore
from gammapy.maps import MapAxis, WcsGeom
from gammapy.cube import MapDataset, MapDatasetMaker


N_OBS = 100
OBS_ID = 110380


def data_prep():
    # Create maps

    data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")
    obs_ids = OBS_ID * np.ones(N_OBS)
    observations = data_store.get_observations(obs_ids)

    energy_axis = MapAxis.from_bounds(
        0.1, 10, nbin=10, unit="TeV", name="energy", interp="log"
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

    return stacked


def write(stacked, filename):
    stacked.write(filename, overwrite=True)


def read(filename):
    return MapDataset.read(filename)


def data_fit(stacked):
    # Data fitting

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

    fit = Fit([stacked])
    result = fit.run(optimize_opts={"print_level": 1})


def flux_point(stacked):
    e_edges = [0.3, 1, 3, 10] * u.TeV
    fpe = FluxPointsEstimator(datasets=[stacked], e_edges=e_edges, source="gc-source")
    fpe.run()


def run_benchmark():
    info = {}
    filename = "stacked_3d.fits.gz"

    t = time.time()

    stacked = data_prep()
    info["data_preparation"] = time.time() - t
    t = time.time()

    write(stacked, filename)
    info["writing"] = time.time() - t
    t = time.time()

    stacked = read(filename)
    info["reading"] = time.time() - t
    t = time.time()

    data_fit(stacked)
    info["data_fitting"] = time.time() - t
    t = time.time()

    flux_point(stacked)
    info["flux_point"] = time.time() - t

    results_folder = "results/analysis_3d/"
    subtimes_filename = results_folder + "/subtimings.yaml"
    with open(subtimes_filename, "w") as fh:
        yaml.dump(info, fh, default_flow_style=False)


if __name__ == "__main__":
    run_benchmark()
