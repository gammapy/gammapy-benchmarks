"""Validation for Crab observations from LST-1 performance study."""

import logging
import time
from pathlib import Path

import astropy.units as u
import click
import requests
from astropy.coordinates import SkyCoord
from gammapy.data import DataStore
from gammapy.datasets import SpectrumDataset
from gammapy.estimators import FluxPointsEstimator
from gammapy.makers import (
    DatasetsMaker,
    ReflectedRegionsBackgroundMaker,
    SpectrumDatasetMaker,
    WobbleRegionsFinder,
)
from gammapy.maps import MapAxis, RegionGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import LogParabolaSpectralModel, SkyModel
from regions import PointSkyRegion

log = logging.getLogger(__name__)

LST1_WORKING_DIR = Path(__file__).parent
DL3_PATH = LST1_WORKING_DIR / "data"
PATH_RESULTS = LST1_WORKING_DIR / "results"
REFERENCE_RESULTS = LST1_WORKING_DIR / "reference"

ZENODO_URL_LST1_CRAB = '11445184'


@click.group()
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING"]))
def cli(log_level):
    """Command-line interface for the script.

    Parameters
    ----------
    log_level : str
        Logging level for the script. Options are "DEBUG", "INFO", or "WARNING".
    
    """
    logging.basicConfig(level=log_level)
    get_data_from_zenodo()


def dl3_to_dl4_reduction(dl3_path):
    """DL3 to DL4 reduction. Returns a Datasets object."""
    datastore = DataStore.from_dir(dl3_path)
    observations = datastore.get_observations(required_irf="point-like")
    target_position = SkyCoord(ra=83.6324, dec=22.0174, unit="deg", frame="icrs")
    on_region = PointSkyRegion(target_position)

    # True and estimated energy axes
    energy_axis = MapAxis.from_energy_bounds(
        0.01, 100, nbin=5, per_decade=True, unit="TeV", name="energy",
    )
    energy_axis_true = MapAxis.from_energy_bounds(
        0.005, 200, nbin=10, per_decade=True, unit="TeV", name="energy_true",
    )

    geom = RegionGeom.create(region=on_region, axes=[energy_axis])

    dataset_empty = SpectrumDataset.create(
        geom=geom, energy_axis_true=energy_axis_true,
    )
    spectrum_dataset_maker = SpectrumDatasetMaker(
        containment_correction=False,
        selection=["counts", "exposure", "edisp"],
    )
    region_finder = WobbleRegionsFinder(n_off_regions=1)
    bkg_maker = ReflectedRegionsBackgroundMaker(region_finder=region_finder)

    makers = [spectrum_dataset_maker, bkg_maker]

    datasets_maker = DatasetsMaker(
        makers,
        stack_datasets=False,
        n_jobs=8,
        parallel_backend="multiprocessing",
    )
    datasets = datasets_maker.run(dataset_empty, observations)
    
    return datasets


def spectral_model_fitting(datasets):
    """Perform spectral model fitting.
    
    The analysis settings are the same used in the LST-1 performance
    study, ApJ 956 80 (2023). It assumes a LogParabola spectral model,
    with reference energy of 400 GeV, in the energy range from 50 GeV to 30 TeV.
    The model is finally serialized as YAML file.
    """
    lp_spectral_model = LogParabolaSpectralModel(
        amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
        alpha=2,
        beta=0.1,
        reference=0.4 * u.TeV,
    )
    lp_spectral_model.parameters["alpha"].min = 0
    lp_spectral_model.parameters["beta"].min = 0

    lp_model = SkyModel(spectral_model=lp_spectral_model, name="crab")

    # Energy range for spectral model fitting
    for dataset in datasets:
        dataset.mask_fit = dataset.counts.geom.energy_mask(0.05 * u.TeV, 30 * u.TeV)

    datasets.models = lp_model

    fit = Fit()
    fit.run(datasets)
    datasets.models.write(PATH_RESULTS / "best_fit_model.yml", overwrite=True)


def get_flux_points(datasets):
    """Calculate flux points for a given datasets and write the results to an ECSV file."""
    # Use the same energy range as for DL3 to DL4 dataset production
    energy_fit_edges = MapAxis.from_energy_bounds(
        0.01, 100,
        nbin=5, per_decade=True,
        unit="TeV",
        name="energy",
    ).edges

    fpe = FluxPointsEstimator(
        energy_edges=energy_fit_edges,
        source="crab",
        selection_optional="all",
    )
    flux_points = fpe.run(datasets)

    log.info("Writing flux points file to %s", PATH_RESULTS)
    table = flux_points.to_table(sed_type="e2dnde", format="gadf-sed")
    table.write(PATH_RESULTS / "flux-points.ecsv", overwrite=True)


@cli.command("run-analysis", help="Run DL3 analysis validation")
def run_analysis():
    """1D analysis of LST-1 Crab observations with energy-dependent directional cuts.
    
    It performs the following steps:
    1. DL3 to DL4 data reduction.
    2. Spectral model fitting using a LogParabola model.
    3. Flux points estimation.
    """
    start_time = time.time()

    log.info("Running 1D spectral analysis for LST-1 Crab Nebula observations.")
    PATH_RESULTS.mkdir(exist_ok=True, parents=True)

    log.info("Running DL3 to DL4 reduction.")
    datasets = dl3_to_dl4_reduction(DL3_PATH)

    log.info("Running spectral model fitting.")
    spectral_model_fitting(datasets)

    log.info("Running flux points estimation.")
    get_flux_points(datasets)

    end_time = time.time()
    duration = end_time - start_time
    log.info(
        "Validation completed successfully: DL3 to DL4 reduction, "
        "spectral model fitting, and flux points estimation. "
        "Total time taken: %.0f s (%.1f min).", duration, duration / 60,
    )


def download_file(url, local_filename) -> None:
    """Download a file from a given URL to a local path."""
    if Path(local_filename).exists():
        log.debug("%s already exists, skipping download.", local_filename)
        return
    with requests.get(url, stream=True) as request:
        request.raise_for_status()
        with open(local_filename, 'wb') as file:
            file.write(request.content)


def get_data_from_zenodo(zenodo_record_id=ZENODO_URL_LST1_CRAB, dir_path=DL3_PATH) -> None:
    """Use Zenodo API to download files from a given Zenodo record ID.
    
    Parameters
    ----------
    zenodo_record_id : str
        The Zenodo record ID to download files from.
        Default is LST-1 Crab Zenodo record.
    dir_path : Path
        The directory path where files will be downloaded.

    TODO: add checksum validation 

    """
    dir_path.mkdir(exist_ok=True, parents=True)

    log.info("Downloading files from Zenodo.")

    zenodo_url = f'https://zenodo.org/api/records/{zenodo_record_id}'
    response = requests.get(zenodo_url)

    if response.ok:
        record_data = response.json()['files']
        file_urls = [entry['links']['self'] for entry in record_data]
        file_names = [entry['key'] for entry in record_data]

        for file_url, file_name in zip(file_urls, file_names):
            file_path = dir_path / file_name
            log.debug("Downloading %s from %s...", file_name, file_url)
            download_file(file_url, file_path)

    else:
        raise RuntimeError(
            "Failed to retrieve the Zenodo record. "
            f"Status code: {response.status_code}",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
