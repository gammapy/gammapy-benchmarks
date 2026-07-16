"""Run gammapy science use cases verifications."""
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

AVAILABLE_USE_CASES = {
    "flux-points-coverage": {
        "folder": "flux_points_coverage",
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


@cli.command("run-verifications", help="Run Gammapy science UC verifications")
@click.argument("use_cases", type=click.Choice(list(AVAILABLE_USE_CASES) + ["all"]))
def run_validations(use_cases):
    info = get_provenance()

    if uses_cases == "all":
        use_cases = list(AVAILABLE_USE_CASES)
    else:
        use_cases = [use_cases]

    for use_case in use_cases:
        cfg = AVAILABLE_USE_CASES[use_case]
        results_folder = THIS_REPO / cfg["folder"] / "results"
        results_folder.mkdir(exist_ok=True, parents=True)

        run_single_use_case(cfg)

        provenance_filename = results_folder / f"prov_{use_case}.yaml"
        with provenance_filename.open("w") as fh:
            log.info("Writing {}".format(provenance_filename))
            yaml.dump(info, fh, default_flow_style=False)


def run_single_use_case(cfg, **kwargs):
    command_path = (Path(cfg["folder"]) / Path(cfg["command"])).absolute()
    cmd = [sys.executable, str(command_path)]
    for arg in cfg["args"]:
        cmd.append(arg)
    log.info(f"Executing command: {cmd}")
    subprocess.run(cmd, cwd=cfg["folder"], check=True)


if __name__ == "__main__":
    cli()
