import time
import astropy.units as u
import numpy as np
import yaml
from pathlib import Path
from gammapy.maps import Map
from gammapy.estimators import TSMapEstimator, ASmoothMapEstimator
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    PointSpatialModel,
    SkyModel,
)
from gammapy.irf import PSFMap, EDispKernelMap
from gammapy.datasets import MapDataset


def data_prep():
    counts = Map.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-counts-cube.fits.gz")
    background = Map.read(
        "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-background-cube.fits.gz"
    )

    exposure = Map.read(
        "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-exposure-cube.fits.gz"
    )
    # unit is not properly stored on the file. We add it manually
    exposure = Map.from_geom(exposure.geom, data=exposure.data, unit="cm2s")

    psfmap = PSFMap.read(
        "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-psf-cube.fits.gz", format="gtpsf"
    )

    edisp = EDispKernelMap.from_diagonal_response(
        energy_axis=counts.geom.axes["energy"],
        energy_axis_true=exposure.geom.axes["energy_true"],
    )

    dataset = MapDataset(
        counts=counts,
        background=background,
        exposure=exposure,
        psf=psfmap,
        name="fermi-3fhl-gc",
        edisp=edisp,
    )
    return dataset


def run_asmooth(dataset):
    scales = u.Quantity(np.arange(0.05, 1, 0.05), unit="deg")
    smooth = ASmoothMapEstimator(threshold=3, scales=scales, energy_edges=[10, 300] * u.GeV)
    images = smooth.run(dataset)
    return images


def fit_estimator(dataset):
    spatial_model = PointSpatialModel()
    spectral_model = PowerLawSpectralModel(index=2)
    model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)
    estimator = TSMapEstimator(
        model, kernel_width="1 deg", energy_edges=[10, 30, 300] * u.GeV
    )
    images = estimator.run(dataset)
    return images


def run_benchmark():
    info = {}

    t = time.time()

    data = data_prep()
    info["data_preparation"] = time.time() - t

    t = time.time()

    run_asmooth(data)
    info["run_asmooth"] = time.time() - t

    t = time.time()

    fit_estimator(data)
    info["TSmap_estimator"] = time.time() - t

    Path("bench.yaml").write_text(yaml.dump(info, sort_keys=False, indent=4))


if __name__ == "__main__":
    run_benchmark()
