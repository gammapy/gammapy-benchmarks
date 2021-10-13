#!/usr/bin/env python
"""Run gammapy benchmarks"""
import getpass
import importlib
import logging
import platform
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import click
import numpy as np
import yaml
from psrecord.main import monitor

log = logging.getLogger(__name__)

THIS_REPO = Path(__file__).parent

AVAILABLE_BENCHMARKS = {
    "io": "io.py",
    "analysis_3d": "analysis_3d.py",
    "analysis_3d_joint": "analysis_3d_joint.py",
    "lightcurve_1d": "lightcurve_1d.py",
    "lightcurve_3d": "lightcurve_3d.py",
    "spectrum_1d": "spectrum_1d.py",
    "spectrum_1d_joint": "spectrum_1d_joint.py",
    "tsmap_estimator": "tsmap_estimator.py",
    "ring_background_estimator": "ring_background_estimator.py",
    "npred": "npred.py",
}

MONITOR_OPTIONS = {"duration": None, "interval": 0.5, "include_children": True}


def get_provenance():
    """Compute provenance info about software and data used."""
    data = {
        "env": {
            "user": getpass.getuser(),
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "software": {},
    }

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
def run_benchmarks(benchmarks):
    info = get_provenance()

    if benchmarks == "all":
        benchmarks = list(AVAILABLE_BENCHMARKS)
    else:
        benchmarks = [benchmarks]

    result = {}  # condenses the results from each benchmark

    for benchmark in benchmarks:
        results_folder = THIS_REPO / f"results/{benchmark}"
        results_folder.mkdir(exist_ok=True, parents=True)

        results_filename = results_folder / "results.txt"
        plot_filename = results_folder / "results.png"
        provenance_filename = results_folder / "provenance.yaml"

        run_single_benchmark(
            benchmark,
            logfile=str(results_filename),
            plot=str(plot_filename),
            **MONITOR_OPTIONS,
        )

        log.info("Writing {}".format(results_filename))
        log.info("Writing {}".format(plot_filename))

        with provenance_filename.open("w") as fh:
            log.info("Writing {}".format(provenance_filename))
            yaml.dump(info, fh, default_flow_style=False)

        dict = {}
        t, cpu, _, memory = np.loadtxt(results_filename, unpack=True)
        dict["total_time"] = float(max(t))
        dict["CPU_max"] = float(max(cpu[2:]))
        dict["CPU_mean"] = float(np.mean(cpu[2:]))
        dict["memory_peak"] = float(np.max(memory))

        result[benchmark] = dict

    yaml_filename = THIS_REPO / "results/results.yaml"
    with yaml_filename.open("w") as fh:
        log.info("Writing {}".format(yaml_filename))
        yaml.dump(result, fh, default_flow_style=False)


def run_single_benchmark(benchmark, **kwargs):
    script_path = Path(AVAILABLE_BENCHMARKS[benchmark]).absolute()
    cmd = "python {}".format(script_path)
    log.info(f"Executing command: {cmd}")

    with tempfile.TemporaryDirectory() as path:
        process = subprocess.Popen(cmd, shell=True, cwd=str(path))
        pid = process.pid
        monitor(pid, **kwargs)
        shutil.copyfile(Path(path) / "bench.yaml", f"results/{benchmark}/bench.yaml")


if __name__ == "__main__":
    cli()
