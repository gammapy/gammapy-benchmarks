import json
from pathlib import Path

import pytest

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Widens the nominal white-noise expectation (mean=0, std=1) to absorb Monte
# Carlo noise, so the test only fails on genuine regressions.
BAND_SAFETY_FACTOR = 4


def _result_files():
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("isolated_source_detection_*crab_*h.json"))


# Optional: only collected once `make.py isd` has been run at least once for a
# given crab_fraction/livetime combination, since the underlying Monte Carlo
# simulation is too expensive to run as part of a regular test suite.
@pytest.mark.parametrize("filename", _result_files(), ids=lambda p: p.stem)
def test_isolated_source_detection(filename):
    """Check that the residual significance map after detect+fit is compatible with white noise."""
    with filename.open() as fh:
        summary = json.load(fh)

    n_valid = summary["n_valid"]
    assert n_valid > 0, (
        f"{filename.name}: no realization detected the source and converged the fit"
    )

    tol_mean = BAND_SAFETY_FACTOR * summary["std_of_means"] * n_valid ** -0.5
    assert abs(summary["mean_of_means"]) < tol_mean, (
        f"{filename.name}: mean_of_means = {summary['mean_of_means']:.3f} "
        f"outside 0 +/- {tol_mean:.3f}"
    )

    tol_std = BAND_SAFETY_FACTOR * summary["std_of_stds"] * n_valid ** -0.5
    assert abs(summary["mean_of_stds"] - 1.0) < tol_std, (
        f"{filename.name}: mean_of_stds = {summary['mean_of_stds']:.3f} "
        f"outside 1 +/- {tol_std:.3f}"
    )
