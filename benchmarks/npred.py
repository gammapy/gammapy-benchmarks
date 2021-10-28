import os
import time
from pathlib import Path

import yaml
import numpy as np
import astropy.units as u

from gammapy.datasets import Datasets, MapDataset
from gammapy.irf import PSFMap, EDispKernelMap
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling.models import (GaussianSpatialModel, Models,
                                     PowerLawSpectralModel, SkyModel,
                                    )

N_ITER = int(os.environ.get("GAMMAPY_BENCH_N_ITER", 10))
N_SRC = int(os.environ.get("GAMMAPY_BENCH_N_SRC", 10))

def prepare_dataset():
    energy = MapAxis.from_energy_bounds(0.1, 100, 5, per_decade=True, unit="TeV")
    energy_true = MapAxis.from_energy_bounds(0.1, 100, 5, unit="TeV", per_decade=True, name="energy_true")
    geom = WcsGeom.create(npix=500, binsz=0.01, axes=[energy])

    dataset = MapDataset.create(geom, energy_axis_true=energy_true)

    dataset.exposure += "1 m2 s"
    dataset.psf = PSFMap.from_gauss(energy_true)
    dataset.edisp = EDispKernelMap.from_gauss(energy, energy_true, 0.1, 0.)

    return Datasets([dataset])

def compute_npreds(datasets, n_iter, n_src):
    models = Models()
    positions = np.random.uniform(-4., 4., (n_src, 2))
    for pos in positions:
        pos = u.Quantity(pos, "deg")
        model = SkyModel(
            spectral_model=PowerLawSpectralModel(),
            spatial_model=GaussianSpatialModel(lon_0=pos[0], lat_0=pos[1], sigma="0.5 deg")
        )
        models.append(model)

    for i in range(n_iter):
        datasets.models = models
        tmp = datasets[0].npred()


def run_benchmark():
    info = {"n_iterations": N_ITER, "n_sources": N_SRC}

    t = time.time()

    datasets = prepare_dataset()
    info["simulations"] = time.time() - t
    t = time.time()

    compute_npreds(datasets, N_ITER, N_SRC)
    info["npred computation"] = time.time() - t
    t = time.time()

    Path("bench.yaml").write_text(yaml.dump(info, sort_keys=False, indent=4))


if __name__ == "__main__":
    run_benchmark()
