#!/usr/bin/env python
# Run gammapy benchmarks
import subprocess
import yaml
import logging
import datetime
import warnings
import getpass
import platform
import sys
from pathlib import Path

import click
import numpy as np
import astropy
import gammapy
from psrecord.main import monitor

log = logging.getLogger(__name__)

THIS_REPO = Path(__file__).parent

AVAILABLE_BENCHMARKS = {
    "3d_analysis": "benchmarks/3d_analysis.py",
    "benchmark_lightcurve": "benchmarks/benchmark_lightcurve.py",
    "benchmark_maps": "benchmarks/benchmark_maps.py",
    "spectral_analysis": "benchmarks/spectral_analysis.py",
}

MONITOR_OPTIONS = {"duration": None, "interval": 0.5, "include_children": True}


def get_provenance():
    """Compute provenance info about software and data used."""
    na = "not available"
    data = {}

    data["env"] = {}
    data["env"]["user"] = getpass.getuser()
    data["env"]["machine"] = platform.machine()
    data["env"]["system"] = platform.system()

    data["software"] = {}
    data["software"]["python_executable"] = sys.executable
    data["software"]["python_version"] = platform.python_version()
    data["software"]["numpy"] = np.__version__
    try:
        import scipy

        scipy_version = scipy.__version__
    except ImportError:
        scipy_version = na
    data["software"]["scipy"] = scipy_version
    data["software"]["astropy"] = astropy.__version__
    try:
        import sherpa

        sherpa_version = sherpa.__version__
    except ImportError:
        sherpa_version = na
    data["software"]["sherpa"] = sherpa_version
    data["software"]["gammapy"] = gammapy.__version__
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
        folder_ = dict(
            version=info["software"]["gammapy"], tag=tag, benchmark=benchmark
        )

        results_folder = THIS_REPO / "results/{benchmark}/{version}/{tag}".format(
            **folder_
        )
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
    command = "python {}".format(AVAILABLE_BENCHMARKS[benchmark])
    log.info("Executing command {}".format(command))

    sprocess = subprocess.Popen(command, shell=True)
    pid = sprocess.pid

    monitor(pid, **kwargs)

    if sprocess is not None:
        sprocess.kill()


if __name__ == "__main__":
    cli()
