import json
from pathlib import Path

import pytest

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Widens the nominal standard-normal expectation (mean=0, std=1) to absorb
# Monte Carlo noise, so the test only fails on genuine regressions.
BAND_SAFETY_FACTOR = 4

COMPONENTS = ["big", "small"]
PULL_PARAMETERS = ["lon_0", "lat_0", "sigma", "index", "amplitude"]


def _result_files():
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("component_separation_big*crab_small*crab_*h.json"))


# Optional: only collected once `make.py cs` has been run at least once for a
# given crab_fraction/livetime combination, since the underlying Monte Carlo
# simulation is too expensive to run as part of a regular test suite.
@pytest.mark.parametrize("filename", _result_files(), ids=lambda p: p.stem)
def test_pull_distribution(filename):
    """Check that fitted-parameter pulls, (fitted - true) / fitted_error, are compatible with N(0, 1).

    Checked for both components ("big" and "small") -- the point of this use
    case is that both are recovered cleanly despite their blending, not just
    that one of them fits.
    """
    with filename.open() as fh:
        summary = json.load(fh)

    n_valid = summary["n_valid"]
    assert n_valid > 0, (
        f"{filename.name}: no realization detected both sources and produced a usable joint fit"
    )

    for component in COMPONENTS:
        for name in PULL_PARAMETERS:
            stats = summary["pulls"][component][name]

            tol_mean = BAND_SAFETY_FACTOR * stats["std"] * n_valid ** -0.5
            assert abs(stats["mean"]) < tol_mean, (
                f"{filename.name}: pull[{component}][{name}].mean = {stats['mean']:.3f} "
                f"outside 0 +/- {tol_mean:.3f}"
            )

            tol_std = BAND_SAFETY_FACTOR * stats["std"] * (2 * (n_valid - 1)) ** -0.5
            assert abs(stats["std"] - 1.0) < tol_std, (
                f"{filename.name}: pull[{component}][{name}].std = {stats['std']:.3f} "
                f"outside 1 +/- {tol_std:.3f}"
            )
