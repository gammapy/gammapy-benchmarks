"""Run gammapy validations"""
import getpass
import importlib
import logging
import platform
import subprocess
import sys
import warnings
from pathlib import Path

import click
import yaml

log = logging.getLogger(__name__)

THIS_REPO = Path(__file__).parent

AVAILABLE_VALIDATIONS = {
    # "cta-1dc_cas_a": {"folder": "cta-1dc", "command": "make.py", "args": ["run-analyses", "cas_a"]},
    # "cta-1dc_hess_j1702": {"folder": "cta-1dc", "command": "make.py", "args": ["run-analyses", "hess_j1702"]},
    # cta-1dc validations disabled in Github actions because needs proprietary dataset
    #
    #
    # "catalog": {"folder": "catalog", "command": "catalog_checks.py", "args": []},
    "event-sampling": {
        "folder": "event-sampling",
        "command": "make.py",
        "args": ["all", "all-models"],
    },
    "fermi-3fhl": {"folder": "fermi-3fhl", "command": "make.py", "args": []},
    "hess-dl3-dr1": {
        "folder": "hess-dl3-dr1",
        "command": "make.py",
        "args": ["run-analyses", "all-targets", "all-methods"],
    },
    "hess-dl3-dr1-plot": {
        "folder": "hess-dl3-dr1",
        "command": "plot.py",
        "args": [],
    },
    "joint-crab-analyses": {
        "folder": "joint-crab",
        "command": "make.py",
        "args": ["run-analyses", "all"],
    },
    "joint-crab-fit": {
        "folder": "joint-crab",
        "command": "make.py",
        "args": ["run-fit", "all"],
    },
    "lightcurve": {
        "folder": "lightcurve",
        "command": "make.py",
        "args": ["run-analyses", "all"],
    },
}


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
    Run and manage Gammapy validations.
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


@cli.command("run-validation", help="Run Gammapy benchmarks")
@click.argument("validations", type=click.Choice(list(AVAILABLE_VALIDATIONS) + ["all"]))
def run_validations(validations):
    info = get_provenance()

    if validations == "all":
        # TODO: enable event sampling again
        validations = list(AVAILABLE_VALIDATIONS)[2:]
    else:
        validations = [validations]

    for validation in validations:
        cfg = AVAILABLE_VALIDATIONS[validation]
        results_folder = THIS_REPO / cfg["folder"] / "results"
        results_folder.mkdir(exist_ok=True, parents=True)

        run_single_validation(cfg)

        provenance_filename = results_folder / f"prov_{validation}.yaml"
        with provenance_filename.open("w") as fh:
            log.info("Writing {}".format(provenance_filename))
            yaml.dump(info, fh, default_flow_style=False)


def run_single_validation(cfg, **kwargs):
    command_path = (Path(cfg["folder"]) / Path(cfg["command"])).absolute()
    cmd = [sys.executable, str(command_path)]
    for arg in cfg["args"]:
        cmd.append(arg)
    log.info(f"Executing command: {cmd}")
    subprocess.run(cmd, cwd=cfg["folder"], check=True)


if __name__ == "__main__":
    cli()
