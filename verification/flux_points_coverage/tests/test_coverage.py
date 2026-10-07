import json
from pathlib import Path

import numpy as np
import pytest

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Widens the nominal coverage band (see fp_utils.create_coverage_figure) to
# absorb Monte Carlo noise, so the test only fails on genuine regressions.
BAND_SAFETY_FACTOR = 4


def _result_files():
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("flux_points_coverage_*crab_*h.json"))


# Optional: only collected once `make.py fp_coverage` has been run at least
# once for a given geometry/crab_fraction/livetime combination, since the
# underlying Monte Carlo simulation is too expensive to run as part of a
# regular test suite.
@pytest.mark.parametrize("filename", _result_files(), ids=lambda p: p.stem)
def test_flux_points_coverage(filename):
    """Check that simulated coverage fractions match their nominal value."""
    with filename.open() as fh:
        summary = json.load(fh)

    nsim = summary["nsim"]
    checks = [
        ("coverage_ci", summary["ref_ci"]),
        ("coverage_ci_covar", summary["ref_ci"]),
        ("coverage_ul", summary["ref_ul"]),
    ]
    for key, ref_val in checks:
        coverage = np.array(summary[key])
        tol = BAND_SAFETY_FACTOR * ref_val * nsim ** -0.5
        assert np.all(np.abs(coverage - ref_val) < tol), (
            f"{filename.name}: {key} = {coverage} outside {ref_val:.3f} +/- {tol:.3f}"
        )
