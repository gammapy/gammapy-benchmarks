import logging
from pathlib import Path
import yaml
import click
import warnings
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy import units as u
from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.modeling.models import Models

log = logging.getLogger(__name__)

AVAILABLE_TARGETS = ["cas_a", "hess_j1702", "gc"]


@click.group()
@click.option(
    "--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING"]),
)
@click.option("--show-warnings", is_flag=True, help="Show warnings?")
def cli(log_level, show_warnings):
    logging.basicConfig(level=log_level)
    log.setLevel(level=log_level)

    if not show_warnings:
        warnings.simplefilter("ignore")


@cli.command("run-analyses", help="Run Gammapy validation: CTA 1DC")
@click.argument("targets", type=click.Choice(list(AVAILABLE_TARGETS) + ["all"]))
def run_analyses(targets):
    targets = list(AVAILABLE_TARGETS) if targets == "all" else [targets]
    for target in targets:
        analysis_3d(target)


def analysis_3d_data_reduction(target):
    log.info(f"analysis_3d_data_reduction: {target}")

    opts = yaml.safe_load(open("targets.yaml"))[target]
    txt = Path("config_template.yaml").read_text()
    txt = txt.format_map(opts)

    config = AnalysisConfig.from_yaml(txt)
    config.flux_points.source = target
    config.datasets.safe_mask.parameters = {"offset_max": 5 * u.deg}

    analysis = Analysis(config)
    analysis.get_observations()
    log.info("Running data reduction")
    analysis.get_datasets()

    # TODO: write datasets and separate fitting to next function
    # Not implemented in Gammapy yet, coming very soon.
    log.info("Running fit ...")
    analysis.read_models(f"{target}/model_3d.yaml")
    logging.info(analysis.models)
    analysis.run_fit()
    logging.info(analysis.fit_result.parameters.to_table())
    path = f"{target}/{target}_3d_bestfit.rst"
    log.info(f"Writing {path}")
    analysis.fit_result.parameters.to_table().write(
        path, format="ascii.rst", overwrite=True
    )

    analysis.get_flux_points()
    path = f"{target}/{target}_3d_fluxpoints.ecsv"
    log.info(f"Writing {path}")
    keys = ["e_ref", "e_min", "e_max", "dnde", "dnde_errp", "dnde_errn", "is_ul"]
    analysis.flux_points.data.table_formatted[keys].write(
        path, format="ascii.ecsv", overwrite=True
    )

    return analysis  # will write to disk when possible


def analysis_3d_modeling(target):
    log.info(f"analysis_3d_modeling: {target}")


def analysis_3d_summary(analysis, target):
    log.info(f"analysis_3d_summary: {target}")
    # TODO:
    # - how to plot a SkyModels ?
    # - PowerLawSpectralModel hardcoded need to find auto way

    path = f"{target}/{target}_3d_bestfit.rst"
    tab = Table.read(path, format="ascii")
    tab.add_index("name")
    dt = "U30"
    comp_tab = Table(names=("Param", "DC1 Ref", "gammapy 3d"), dtype=[dt, dt, dt])

    ref_model = Models.read(f"{target}/reference/dc1_model_3d.yaml")
    pars = ref_model.parameters.names
    pars.remove("reference")  # need to find a better way to handle this

    for par in pars:
        ref = ref_model.parameters[par].value
        value = tab.loc[par]["value"]
        name = tab.loc[par]["name"]
        error = tab.loc[par]["error"]
        comp_tab.add_row([name, str(ref), f"{value}±{error}"])

    analysis.datasets["stacked"].counts.sum_over_axes().plot(add_cbar=True)
    plt.savefig(f"{target}/{target}_counts.png", bbox_inches="tight")
    plt.close()

    analysis.datasets["stacked"].plot_residuals(
        method="diff/sqrt(model)", vmin=-0.5, vmax=0.5
    )
    plt.savefig(f"{target}/{target}_residuals.png", bbox_inches="tight")
    plt.close()

    # Cannot specify flux_unit outputs in cm-2 s-1 TeV-1. Default to erg
    ax_sed, ax_residuals = analysis.flux_points.peek()

    pwl = ref_model[target].spectral_model
    ax_sed = pwl.plot(
        (0.1, 50) * u.TeV,
        ax=ax_sed,
        energy_power=2,
        flux_unit="cm-2 s-1 erg-1",
        label="Reference",
        ls="--",
    )
    ax_sed.legend()
    plt.savefig(f"{target}/{target}_fluxpoints.png", bbox_inches="tight")
    plt.close()

    # Generate README.md file with table and plots
    path = f"{target}/spectral_comparison_table.md"
    comp_tab.write(path, format="ascii.html", overwrite=True)

    txt = Path(f"{target}/spectral_comparison_table.md").read_text()
    im1 = f"\n ![Spectra]({target}_fluxpoints.png)"
    im2 = f"\n ![Excess map]({target}_counts.png)"
    im3 = f"\n ![Residual map]({target}_residuals.png)"

    out = txt + im1 + im2 + im3
    Path(f"{target}/README.md").write_text(out)


def analysis_3d(target):
    log.info(f"analysis_3d: {target}")
    analysis = analysis_3d_data_reduction(target)
    analysis_3d_modeling(target)
    analysis_3d_summary(analysis, target)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
