# Licensed under a 3-clause BSD style license - see LICENSE
"""Validation of the PPPC4 and CosmiXs continuum spectra in Gammapy.

For every mass and x = E / m_DM listed in the tables, the dN/dE returned by
Gammapy (`ContinuumPrimaryFlux`) must equal the value expected from the table,
dN/dE = dN/dlog10(x) / (E ln 10). The tables are read with pandas,
independently of Gammapy. Check: relative difference < 1e-6.
"""

import logging
import warnings
from pathlib import Path

import click
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.table import Table

from gammapy.astro.darkmatter import ContinuumPrimaryFlux
from gammapy.utils.scripts import make_path

log = logging.getLogger(__name__)

PATH_RESULTS = Path(__file__).parent / "results"

CHANNELS = ["b", "tau", "mu", "W"]
TOLERANCE = 1e-6
REL_THRESHOLD = 1e-2 

TABLES = {
    "pppc4": {
        "filename": "$GAMMAPY_DATA/dark_matter_spectra/PPPC4DMID/AtProduction_gammas.dat",
        "mass": "mDM",
        "log10x": "Log[10,x]",
        "channels": {"b": "b", "tau": r"\[Tau]", "mu": r"\[Mu]", "W": "W"},
    },
    "cosmixs": {
        "filename": "$GAMMAPY_DATA/dark_matter_spectra/cosmixs/AtProduction-Gamma.dat",
        "mass": "#DM",
        "log10x": "Log10[x]",
        "channels": {
            "b": "dNdLog10x[b]",
            "tau": "dNdLog10x[tau]",
            "mu": "dNdLog10x[mu]",
            "W": "dNdLog10x[W]",
        },
    },
}


@click.command()
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING"]))
def main(log_level):
    logging.basicConfig(level=log_level)
    warnings.simplefilter("ignore")
    rows = []

    for source, cfg in TABLES.items():
        df = pd.read_csv(make_path(cfg["filename"]), sep=r"\s+")

        for channel in CHANNELS:
            primary_flux = ContinuumPrimaryFlux(1 * u.TeV, channel=channel, source=source)
            diffs = []

            for mass_gev, sub in df.groupby(cfg["mass"]):
                log10x = sub[cfg["log10x"]].values
                dndlog10x = sub[cfg["channels"][channel]].values

                # x < 1 and non-negligible values (e.g. no photons for W below threshold)
                sel = (log10x < 0) & (dndlog10x > REL_THRESHOLD * dndlog10x.max())
                if not np.any(sel):
                    continue

                mass = mass_gev * u.GeV
                energy = 10 ** log10x[sel] * mass
                expected = dndlog10x[sel] / (energy.to_value("GeV") * np.log(10))
                primary_flux.mDM = mass
                gammapy = primary_flux(energy).to_value("GeV-1")
                diffs.append(np.max(np.abs(gammapy / expected - 1)))

            row = {
                "source": source,
                "channel": channel,
                "n_masses": len(diffs),
                "max_rel_diff": np.max(diffs),
                "passed": bool(np.max(diffs) < TOLERANCE),
            }
            rows.append(row)
            log.info(
                f"{'PASS' if row['passed'] else 'FAIL'} {source:8s} {channel:4s} "
                f"{row['n_masses']} masses, max relative difference {row['max_rel_diff']:.1e}"
            )

    table = Table(rows=rows)
    table.meta["tolerance"] = TOLERANCE
    PATH_RESULTS.mkdir(exist_ok=True)
    table.write(PATH_RESULTS / "spectra.ecsv", overwrite=True)
    log.info(f"Wrote {PATH_RESULTS / 'spectra.ecsv'}")


if __name__ == "__main__":
    main()