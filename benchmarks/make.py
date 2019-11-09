#!/usr/bin/env python
"""Run gammapy benchmarks"""
import subprocess
import yaml
import logging
import datetime
import warnings
import getpass
import platform
import sys
import importlib
from pathlib import Path

import click
from psrecord.main import monitor

log = logging.getLogger(__name__)

THIS_REPO = Path(__file__).parent

AVAILABLE_BENCHMARKS = {
    "analysis_3d": "analysis_3d.py",
    "analysis_3d_joint": "analysis_3d_joint.py",
    "maps_3d": "maps_3d.py",
    "lightcurve_1d": "lightcurve_1d.py",
    "lightcurve_3d": "lightcurve_3d.py",
    "spectrum_1d": "spectrum_1d.py",
    "spectrum_1d_joint": "spectrum_1d_joint.py",
}

MONITOR_OPTIONS = {"duration": None, "interval": 0.5, "include_children": True}


def get_provenance():
    """Compute provenance info about software and data used."""
    data = {}

    data["env"] = {
        "user": getpass.getuser(),
        "machine": platform.machine(),
        "system": platform.system(),
    }

    data["software"] = {}
    data["software"]["python_executable"] = sys.executable
    data["software"]["python_version"] = platform.python_version()
    data["software"]["numpy"] = importlib.import_module("numpy").__version__
    data["software"]["scipy"] = importlib.import_module("scipy").__version__
    data["software"]["astropy"] = importlib.import_module("astropy").__version__
    data["software"]["gammapy"] = importlib.import_module("gammapy").__version__

    return data


@click.group()
@click.option(
    "--log-level",
    default="info",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
)
@click.option("--show-warnings", is_flag=True, help="Show warnings?")
def cli(log_level, show_warnings):
    """
    Run and manage Gammapy benchmarks.
    """
    levels = dict(
        debug=logging.DEBUG,
        info=logging.INFO,
        warning=logging.WARNING,
        error=logging.ERROR,
        critical=logging.CRITICAL,
    )
    logging.basicConfig(level=levels[log_level])
    log.setLevel(level=levels[log_level])

    if not show_warnings:
        warnings.simplefilter("ignore")


@cli.command("run-benchmark", help="Run Gammapy benchmarks")
@click.argument("benchmarks", type=click.Choice(list(AVAILABLE_BENCHMARKS) + ["all"]))
@click.option(
    "--tag",
    help="Assign a tag to the benchmark run, so results will"
         " be stored under this tag.",
)
def run_benchmarks(benchmarks, tag):
    info = get_provenance()

    if benchmarks == "all":
        benchmarks = list(AVAILABLE_BENCHMARKS)
    else:
        benchmarks = [benchmarks]

    if tag is None:
        now = datetime.datetime.now()
        tag = now.strftime("%Y-%m-%d")

    for benchmark in benchmarks:
        version = info["software"]["gammapy"]
        results_folder = THIS_REPO / f"results/{benchmark}/{version}/{tag}"
        results_folder.mkdir(exist_ok=True, parents=True)

        results_filename = results_folder / "results.txt"
        plot_filename = results_folder / "results.png"
        provenance_filename = results_folder / "provenance.yaml"

        run_single_benchmark(
            benchmark,
            logfile=str(results_filename),
            plot=str(plot_filename),
            **MONITOR_OPTIONS
        )

        log.info("Writing {}".format(results_filename))
        log.info("Writing {}".format(plot_filename))

        with provenance_filename.open("w") as fh:
            log.info("Writing {}".format(provenance_filename))
            yaml.dump(info, fh, default_flow_style=False)


def run_single_benchmark(benchmark, **kwargs):
    cmd = "python {}".format(AVAILABLE_BENCHMARKS[benchmark])
    log.info(f"Executing command: {cmd}")

    process = subprocess.Popen(cmd, shell=True)
    pid = process.pid

    monitor(pid, **kwargs)

    if process is not None:
        process.kill()


if __name__ == "__main__":
    cli()
