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
    "flux-points-coverage-1d": {
        "folder": "flux_points_coverage",
        "command": "make.py",
        "args": ["fp_coverage", "1d", "--livetime", "5h", "--n_samples", "100"],
    },
     "flux-points-coverage-3d": {
        "folder": "flux_points_coverage",
        "command": "make.py",
        "args": ["fp_coverage", "3d", "--livetime", "5h", "--n_samples", "100"],
    },
    "isolated-source-detection": {
        "folder": "isolated_source_detection",
        "command": "make.py",
        "args": ["isd", "--livetime", "5h", "--n_samples", "100"],
    },
    "transient-characterization": {
        "folder": "transient_characterization",
        "command": "make.py",
        "args": ["tc", "--livetime", "30min", "--n_samples", "20"],
    },
    "component-separation": {
        "folder": "component_separation",
        "command": "make.py",
        "args": ["cs", "--livetime", "5h", "--n_samples", "100"],
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

    if use_cases == "all":
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


@cli.command("run-tests", help="Run the pytest test suite for one or all use cases")
@click.argument("use_cases", type=click.Choice(list(AVAILABLE_USE_CASES) + ["all"]))
def run_tests(use_cases):
    """Run each use case's `tests/` folder with pytest.

    Tests are optional/self-skipping (see CLAUDE.md): a use case whose
    `make.py` hasn't been run yet simply reports 0 collected tests, not a
    failure.
    """
    if use_cases == "all":
        use_cases = list(AVAILABLE_USE_CASES)
    else:
        use_cases = [use_cases]

    # Some use cases share a folder (e.g. flux-points-coverage-1d/3d both
    # live in flux_points_coverage/) -- run each folder's tests only once.
    folders = sorted({AVAILABLE_USE_CASES[use_case]["folder"] for use_case in use_cases})

    failures = []
    for folder in folders:
        log.info(f"Running tests in {folder}/tests")
        cmd = [sys.executable, "-m", "pytest", "tests/"]
        result = subprocess.run(cmd, cwd=folder)
        if result.returncode != 0:
            failures.append(folder)

    if failures:
        raise click.ClickException(f"Tests failed in: {', '.join(failures)}")

    log.info("All tests passed.")


if __name__ == "__main__":
    cli()
