from pathlib import Path

import numpy as np
import yaml
from astropy.table import Table
from numpy.testing import assert_allclose

AVAILABLE_DATA = ["hess", "magic", "veritas", "fact", "fermi", "joint"]
THIS_FOLDER = Path(__file__).resolve().parent


def test_fit():

    for instrument in AVAILABLE_DATA:
        filename = (
            THIS_FOLDER / ".." / ".." / "data" / "joint-crab" / "published" / "fit"
        )
        filename = filename / f"fit_{instrument}.yaml"
        with open(filename, "r") as file:
            paper_result = yaml.safe_load(file)

        filename = THIS_FOLDER / ".." / "results"
        filename = filename / f"fit_{instrument}.rst"
        validation_result = Table.read(filename, format="ascii")

        assert_allclose(
            paper_result["parameters"][0]["value"],
            validation_result["value"][0],
            rtol=5e-1,
        )
        assert_allclose(
            paper_result["parameters"][1]["value"],
            validation_result["value"][1],
            rtol=5e-1,
        )
        assert_allclose(
            paper_result["parameters"][2]["value"],
            validation_result["value"][2],
            rtol=5e-1,
        )
        assert_allclose(
            paper_result["parameters"][3]["value"],
            validation_result["value"][3] * np.log(10),
            rtol=5e-1,
        )
