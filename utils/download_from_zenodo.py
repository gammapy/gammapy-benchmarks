# Licensed under a 3-clause BSD style license - see LICENSE

"""Download files from Zenodo records using its API.

Usage:
    python download_from_zenodo.py <zenodo_record_id> <dir_path>

For example, for LST1 Crab data (assuming $GAMMAPY_DATA is set):
    python download_from_zenodo.py 11445184 $GAMMAPY_DATA/lst1_crab_data
"""

import logging
from pathlib import Path

import click
import requests
from tqdm import tqdm

__all__ = ["download_file"]

log = logging.getLogger(__name__)


def download_file(url, local_filename) -> None:
    """Download a file from a given URL to a specified local path.

    If the file already exists at the local path, the download is skipped.

    Parameters
    ----------
    url : str
        The URL from which to download the file.
    local_filename : str or Path
        The local file path where the downloaded file will be saved.

    Returns
    -------
    None

    Raises
    ------
    requests.HTTPError
        If the HTTP request returned an unsuccessful status code.

    """
    if Path(local_filename).exists():
        log.debug("%s already exists, skipping download.", local_filename)
        return
    with requests.get(url, stream=True) as request:
        request.raise_for_status()
        with open(local_filename, 'wb') as file:
            file.write(request.content)


@click.command()
@click.argument("zenodo_record_id", type=str)
@click.argument(
    "dir_path",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
)
def main(zenodo_record_id, dir_path) -> None:
    """Download files from a given Zenodo record ID using its API.
    
    Parameters
    ----------
    zenodo_record_id : str
        The Zenodo record ID to download files from.
    dir_path : Path
        The directory path where files will be downloaded.

    """
    # TODO: add checksum validation
    logging.basicConfig(level=logging.INFO)

    dir_path.mkdir(exist_ok=True, parents=True)

    zenodo_url = f'https://zenodo.org/api/records/{zenodo_record_id}'
    response = requests.get(zenodo_url)

    if not response.ok:
        raise RuntimeError(
            "Failed to retrieve the Zenodo record. "
            f"Status code: {response.status_code}",
        )
    
    else:
        title = response.json().get('metadata', {}).get('title', 'No title found')
        log.info("Downloading files from Zenodo record %s: %s", zenodo_record_id, title)

        record_data = response.json()['files']
        file_urls = [entry['links']['self'] for entry in record_data]
        file_names = [entry['key'] for entry in record_data]

        log.info("Files in the Zenodo record:")
        for file_name in file_names:
            log.info("  %s", file_name)

        for file_url, file_name in tqdm(
            list(zip(file_urls, file_names)), desc="Downloading files", unit="file",
        ):
            file_path = dir_path / file_name
            log.debug("Downloading %s from %s...", file_name, file_url)
            download_file(file_url, file_path)


if __name__ == "__main__":
    main()