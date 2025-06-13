"""Data reduction, fitting, and flux point estimation for LST-1 Crab sample.

It includes functions for data preparation, writing and reading datasets,
fitting models, and benchmarking the entire process.
"""

import time
from pathlib import Path

import astropy.units as u
import yaml
from astropy.coordinates import SkyCoord
from gammapy.data import DataStore
from gammapy.datasets import Datasets, SpectrumDataset
from gammapy.estimators import FluxPointsEstimator
from gammapy.makers import (
    ReflectedRegionsBackgroundMaker,
    SafeMaskMaker,
    SpectrumDatasetMaker,
    WobbleRegionsFinder,
)
from gammapy.maps import MapAxis, RegionGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    LogParabolaSpectralModel,
    SkyModel,
)
from regions import PointSkyRegion

PATH_RESULTS = Path(__file__).parent / "lst1_results"

def data_prep():
    """Prepare the data for analysis.

    This function sets up the target region, creates the necessary geometry,
    and reduces the observations to datasets.

    Returns
    -------
    datasets : `~gammapy.datasets.Datasets`
        A collection of datasets ready for analysis.
    
    """
    data_store = DataStore.from_dir("$GAMMAPY_DATA/lst1_crab_data")
    observations = data_store.get_observations(required_irf="point-like")

    target_position = SkyCoord(ra=83.63, dec=22.01, unit="deg", frame="icrs")
    on_region = PointSkyRegion(target_position)

    e_reco = MapAxis.from_bounds(
        0.05, 50, nbin=60, interp="log", unit="TeV", name="energy",
    )
    e_true = MapAxis.from_bounds(
        0.01, 100, nbin=100, interp="log", unit="TeV", name="energy_true",
    )

    geom = RegionGeom(on_region, axes=[e_reco])

    dataset_empty = SpectrumDataset.create(geom=geom, energy_axis_true=e_true)

    dataset_maker = SpectrumDatasetMaker(
        containment_correction=False, selection=["counts", "exposure", "edisp"],
    )

    region_finder = WobbleRegionsFinder(n_off_regions=1)
    bkg_maker = ReflectedRegionsBackgroundMaker(region_finder=region_finder)

    safe_mask_masker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)

    spectral_model = LogParabolaSpectralModel(
        alpha=2, beta=0.1, amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=0.4 * u.TeV,
    )
    sky_model = SkyModel(
        spectral_model=spectral_model, name="crab",
    )

    datasets = Datasets()

    for observation in observations:
        dataset = dataset_empty.copy(name=f"dataset-{observation.obs_id}")
        dataset = dataset_maker.run(dataset=dataset, observation=observation)
        dataset_on_off = bkg_maker.run(dataset, observation)
        dataset_on_off = safe_mask_masker.run(dataset_on_off, observation)
        datasets.append(dataset_on_off)

    datasets.models = sky_model
    return datasets


def write_dataset(datasets, filename):
    """Write the datasets and its models to YAML files.

    Parameters
    ----------
    datasets : `~gammapy.datasets.Datasets`
        The datasets to be written.
    filename : str
        The base filename for the output YAML files.
    
    """
    datasets.write(
        PATH_RESULTS / f"{filename}_datasets.yaml",
        filename_models= PATH_RESULTS / f"{filename}_models.yaml",
        overwrite=True,
    )


def read_dataset(filename):
    """Read the datasets and its models from YAML files.

    Parameters
    ----------
    filename : str
        The base filename for the input YAML files.

    Returns
    -------
    datasets : `~gammapy.datasets.Datasets`
        The datasets read from the YAML files.
    
    """
    return Datasets.read(
        PATH_RESULTS / f"{filename}_datasets.yaml",
        filename_models= PATH_RESULTS / f"{filename}_models.yaml",
    )


def spectral_fitting(datasets):
    """Perform spectral model fitting on the datasets.

    Parameters
    ----------
    datasets : `~gammapy.datasets.Datasets`
        The datasets to be fitted.

    """
    fit = Fit(optimize_opts={"print_level": 1})
    result = fit.run(datasets)
    print("Fit success:", result.success)


def estimate_flux_points(datasets):
    """Estimate flux points for the given datasets.

    Parameters
    ----------
    datasets : `~gammapy.datasets.Datasets`
        The datasets for which flux points are to be estimated.
    
    """
    e_edges = MapAxis.from_bounds(0.05, 30, nbin=11, interp="log", unit="TeV").edges
    fpe = FluxPointsEstimator(energy_edges=e_edges)
    fpe.run(datasets=datasets)


def run_benchmark():
    """Run the full benchmark process.
    
    It includes data preparation, writing/reading datasets,
    spectral model fitting, and flux point estimation.
    """
    filename = "lst1"

    PATH_RESULTS.mkdir(exist_ok=True, parents=True)

    t = time.time()

    datasets = data_prep()
    info = {"n_obs": len(datasets)}
    info["data_preparation"] = time.time() - t
    t = time.time()

    write_dataset(datasets, filename)
    info["writing"] = time.time() - t
    t = time.time()

    datasets = read_dataset(filename)
    info["reading"] = time.time() - t
    t = time.time()

    spectral_fitting(datasets)
    info["data_fitting"] = time.time() - t
    t = time.time()

    estimate_flux_points(datasets)
    info["flux_point"] = time.time() - t

    Path("bench.yaml").write_text(yaml.dump(info, sort_keys=False, indent=4))


if __name__ == "__main__":
    run_benchmark()
