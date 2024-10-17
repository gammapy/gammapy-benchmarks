import logging
import time
from pathlib import Path
from astropy.coordinates import SkyCoord

import click
import astropy.units as u

from gammapy.data import DataStore
from gammapy.datasets import Datasets, SpectrumDataset
from gammapy.estimators import FluxPointsEstimator
from gammapy.makers import (
    SpectrumDatasetMaker,
    WobbleRegionsFinder,
    ReflectedRegionsBackgroundMaker,
)
from gammapy.maps import MapAxis, RegionGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    SkyModel,
    LogParabolaSpectralModel
)
from regions import PointSkyRegion

import requests


log = logging.getLogger(__name__)

DL3_PATH = Path("data")
PATH_RESULTS = Path("results")


@click.group()
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING"]))
def cli(log_level):
    logging.basicConfig(level=log_level)
    get_data_from_zenodo()


def dl3_to_dl4_reduction(dl3_path):
    """DL3 to DL4 reduction. Returns a stacked Dataset."""
    datastore = DataStore.from_dir(dl3_path)
    observations = datastore.get_observations(required_irf="point-like")
    target_position = SkyCoord.from_name("Crab Nebula", frame='icrs')
    on_region = PointSkyRegion(target_position)

    # True and estimated energy axes
    energy_axis = MapAxis.from_energy_bounds(
        0.01, 100, nbin=5, per_decade=True, unit="TeV", name="energy"
    )
    energy_axis_true = MapAxis.from_energy_bounds(
        0.005, 200, nbin=10, per_decade=True, unit="TeV", name="energy_true"
    )

    geom = RegionGeom.create(region=on_region, axes=[energy_axis])

    dataset_empty = SpectrumDataset.create(
        geom=geom, energy_axis_true=energy_axis_true
    )
    dataset_maker = SpectrumDatasetMaker(
        containment_correction=False,
        selection=["counts", "exposure", "edisp"]
    )
    region_finder = WobbleRegionsFinder(n_off_regions=1)
    bkg_maker = ReflectedRegionsBackgroundMaker(region_finder=region_finder)

    datasets = Datasets()

    for observation in observations:
        dataset = dataset_maker.run(
            dataset_empty.copy(name=str(observation.obs_id)), observation
        )
        dataset_on_off = bkg_maker.run(dataset, observation)
        datasets.append(dataset_on_off)
    
    return datasets.stack_reduce()


def spectral_model_fitting(dataset):
    """Perform spectral model fitting assuming a reference energy of 400 GeV.
    
    Then store the model serialized as YAML file.
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
    dataset.mask_fit = dataset.counts.geom.energy_mask(0.05 * u.TeV, 30 * u.TeV)

    dataset.models = lp_model

    fit = Fit()
    fit.run(dataset)
    dataset.models.write(PATH_RESULTS / "best_fit_model.yml", overwrite=True)


def get_flux_points(dataset):
    """Calculate flux points for a given datasets and write the results to an ECSV file."""
    # Use the same energy range as for DL3 to DL4 dataset production
    energy_fit_edges = MapAxis.from_energy_bounds(
        0.01, 100,
        nbin=5, per_decade=True,
        unit="TeV",
        name="energy"
    ).edges

    fpe = FluxPointsEstimator(
        energy_edges=energy_fit_edges,
        source="crab",
        selection_optional="all",
    )
    flux_points = fpe.run(dataset)

    log.info(f"Writing flux points file to {PATH_RESULTS}")
    table = flux_points.to_table(sed_type="e2dnde", format="gadf-sed")
    table.write(PATH_RESULTS / "flux-points.ecsv", overwrite=True)


@cli.command("run-analysis", help="Run DL3 analysis validation")
@click.option("--debug", is_flag=True)
def run_analysis(debug):
    """1D analysis of LST-1 Crab observations with energy-dependent directional cuts."""
    start_time = time.time()

    log.info("Running 1D analysis for Crab Nebula.")
    PATH_RESULTS.mkdir(exist_ok=True, parents=True)

    log.info("Running DL3 to DL4 reduction.")
    stacked_dataset = dl3_to_dl4_reduction(DL3_PATH)

    log.info("Running spectral model fitting.")
    spectral_model_fitting(stacked_dataset)

    log.info("Running flux points estimation.")
    get_flux_points(stacked_dataset)

    end_time = time.time()
    duration = end_time - start_time
    log.info(f"The time taken for the validation is: {duration:.0f} s ({duration/60:.1f} min).")


def download_file(url, local_filename):
    if Path(local_filename).exists():
        log.debug(f"{local_filename} already exists, skipping download.")
        return
    # Send request to Zenodo
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            # In chunks to prevent memory overload for large files
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive new chunks
                    f.write(chunk)
    return local_filename


def get_data_from_zenodo():
    zenodo_record_id = '11445184'
    zenodo_url = f'https://zenodo.org/api/records/{zenodo_record_id}'
    DL3_PATH.mkdir(exist_ok=True, parents=True)

    log.info("Downloading DL3 files from Zenodo.")

    response = requests.get(zenodo_url)
    if response.status_code == 200:
        record_data = response.json()

        files = record_data['files']
        
        # Download all files in the Zenodo entry
        for file in files:
            file_url = file['links']['self']
            file_name = file['key']
            file_path = DL3_PATH / file_name
            log.debug(f"Downloading {file_name}...")
            download_file(file_url, file_path)
    else:
        log.error("Failed to retrieve the Zenodo record.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
