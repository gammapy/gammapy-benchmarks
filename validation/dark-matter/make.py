# Licensed under a 3-clause BSD style license - see LICENSE
"""Run the dark matter dSph validation: spectra, annihilation and decay limits.

python make.py                      # all steps
python make.py spectra decay        # some steps
python make.py --n-realizations 100 # median over realizations instead of Asimov
"""

import logging
import subprocess
import sys
from pathlib import Path

import click
import numpy as np
from astropy.table import Table

log = logging.getLogger(__name__)

FOLDER = Path(__file__).parent

STEPS = {
    "spectra": {"script": "make_spectra.py", "results": "spectra.ecsv"},
    "annihilation": {"script": "make_annihilation.py", "results": "limits_annihilation.ecsv"},
    "decay": {"script": "make_decay.py", "results": "limits_decay.ecsv"},
}


@click.command()
@click.argument("steps", nargs=-1, type=click.Choice([*STEPS, "all"]))
@click.option("--n-realizations", default=0, help="Limits: 0 means Asimov dataset")
@click.option("--binsz", default=None, type=float, help="Limits: spatial bin size in deg")
def main(steps, n_realizations, binsz):
    logging.basicConfig(level="INFO")
    steps = list(STEPS) if not steps or "all" in steps else list(steps)

    crashed = []
    for step in steps:
        cmd = [sys.executable, str(FOLDER / STEPS[step]["script"])]
        if step != "spectra":
            cmd += ["--n-realizations", str(n_realizations)]
            if binsz is not None:
                cmd += ["--binsz", str(binsz)]
        log.info(f"Running {step}: {' '.join(cmd[1:])}")
        if subprocess.run(cmd, cwd=FOLDER).returncode != 0:
            log.error(f"{step} did not finish")
            crashed.append(step)

    log.info("=== Summary ===")
    for step in steps:
        if step in crashed:
            log.info(f"{step:13s} CRASHED")
            continue
        table = Table.read(FOLDER / "results" / STEPS[step]["results"])
        n_passed = int(np.sum(table["passed"]))
        log.info(f"{step:13s} {n_passed}/{len(table)} checks passed")
        for row in table[~table["passed"]]:
            case = " ".join(str(row[c]) for c in ("source", "target", "channel") if c in row.colnames)
            if "mass" in row.colnames:
                case += f" {row['mass']:g} TeV"
            log.info(f"{'':13s} FAIL {case}")

    if crashed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()