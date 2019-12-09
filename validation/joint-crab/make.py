#!/usr/bin/env python
"""Run Gammapy validation: CTA 1DC"""
import logging
from pathlib import Path
import yaml
import click
import warnings
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy import units as u
from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.modeling.models import SkyModel
from gammapy.modeling.models import LogParabolaSpectralModel
from gammapy.modeling import Fit, Datasets
from gammapy.utils.scripts import make_path
log = logging.getLogger(__name__)

AVAILABLE_DATA = ["hess","magic","veritas","fact"]

DATASETS = [
    {
        "name": "fermi",
        "label": "Fermi-LAT",
        "obs_ids": [0],
        "on_radius": "0.3 deg",
        "containment_correction": True,
        "energy_range": {"min": "0.03 TeV", "max": "2 TeV"},
        "color": "#21ABCD",
    },

    {
        "name": "joint",
        "label": "joint fit",
        "energy_range": {"min": "0.03 TeV", "max": "30 TeV"},
        "color": "crimson",
    },
]

instrument_opts = dict(
    hess = {'on_radius':'0.11 deg', 
            'stack':False, 
            'containment':True, 
            'emin':'0.66 TeV', 
            'emax':'30 TeV',
            'color': "#002E63",},
    magic = {'on_radius':'0.14 deg', 
             'stack':False, 
             'containment':False,
             'emin':'0.08 TeV', 
             'emax':'30 TeV',
             'color': "#FF9933",},
    veritas = {'on_radius':'0.1 deg', 
               'stack':False, 
               'containment':False,
               'emin':'0.15 TeV', 
               'emax':'30 TeV',
               'color': "#893F45",},
    fact = {'on_radius':'0.1732 deg', 
            'stack':True, 
            'containment':False,
            'emin':'0.4 TeV', 
            'emax':'30 TeV',
            'color': "#3EB489",},

)

@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING"]),
)
@click.option("--show-warnings", is_flag=True, help="Show warnings?")
def cli(log_level, show_warnings):
    logging.basicConfig(level=log_level)
    log.setLevel(level=log_level)

    if not show_warnings:
        warnings.simplefilter("ignore")


@cli.command("run-analyses", help="Run Gammapy validation: joint Crab")
@click.argument("instruments", type=click.Choice(list(AVAILABLE_DATA) + ["all"]))
@click.argument("npoints", required = False, type=int, default=30)
def run_analyses(instruments, npoints=40):    
    # Loop over instruments
    if instruments == "all":
        instruments = list(AVAILABLE_DATA)
    else:
        instruments = [instruments]

    joint = []
    for instrument in instruments:
        # First perform data reduction and save to disk
        data_reduction(instrument)
        data_fitting(instrument, npoints)
        make_summary(instrument)
        
def data_reduction(instrument):
    log.info(f"data_reduction: {instrument}")
    config = AnalysisConfig.read(f"config.yaml")
    config.observations.datastore = f"$JOINT_CRAB/data/{instrument}"
    config.datasets.stack = instrument_opts[instrument]['stack']
    config.datasets.containment_correction = instrument_opts[instrument]['containment']
    config.datasets.on_region.radius = instrument_opts[instrument]['on_radius']

    analysis = Analysis(config)
    analysis.get_observations()
    analysis.get_datasets() 
    analysis.datasets.write(instrument, overwrite=True)

    
def define_model():
    crab_spectrum = LogParabolaSpectralModel(amplitude=1e-11/u.cm**2/u.s/u.TeV,
                                         reference=1*u.TeV,
                                         alpha=2.3,
                                          beta=0.2)

    crab_model = SkyModel(spatial_model=None, spectral_model=crab_spectrum, name="crab")

    return crab_model

def make_contours(fit, result, npoints):
    log.info(f"Running contours with {npoints} points...")
    contours = dict()
    contour = fit.minos_contour(result.parameters['alpha'], 
                                 result.parameters['beta'], 
                                 numpoints=npoints)
    contours["contour_alpha_beta"] = {
        "alpha": contour["x"].tolist(),
        "beta": contour["y"].tolist(),
    }    
    contour = fit.minos_contour(result.parameters['amplitude'], 
                                 result.parameters['alpha'], 
                                 numpoints=npoints)
    contours["contour_amplitude_alpha"] = {
        "amplitude": contour["x"].tolist(),
        "alpha": contour["y"].tolist(),
    }     
    
    contour_amplitude_beta = fit.minos_contour(result.parameters['amplitude'], 
                                 result.parameters['beta'], 
                                 numpoints=npoints)
    contours["contour_amplitude_beta"] = {
        "amplitude": contour["x"].tolist(),
        "beta": contour["y"].tolist(),
    } 

    return contours


def data_fitting(instrument, npoints):
    log.info("Running fit ...")
    # First define model
    crab_model = define_model()
    
    # Read from disk
    datasets = Datasets.read(f"{instrument}/_datasets.yaml", 
                            f"{instrument}/_models.yaml")
    
    e_min = u.Quantity(instrument_opts[instrument]['emin'])
    e_max = u.Quantity(instrument_opts[instrument]['emax'])
    
    # Set model and fit range
    for ds in datasets:
        ds.models = crab_model
        ds.mask_fit = ds.counts.energy_mask(e_min, e_max)
    # Perform fit
    fit = Fit(datasets)
    result = fit.run()
    log.info(result.parameters.to_table())
    
    path = f"results/fit_{instrument}.rst"
    log.info(f"Writing {path}")
    result.parameters.to_table().write(
        path, format="ascii.rst", overwrite=True
    )
    
    contours = make_contours(fit, result, npoints)
    with open(f"results/contours_{instrument}.yaml", 'w') as file:
        yaml.dump(contours, file)
    

def plot_contour_line(ax, x, y, **kwargs):
    """Plot smooth curve from points.

    There is some noise in the contour points from MINUIT,
    which throws off most interpolation schemes.

    The LSQUnivariateSpline as used here gives good results.

    It could probably be simplified, or Bezier curve plotting via
    matplotlib could also be tried:
    https://matplotlib.org/gallery/api/quad_bezier.html
    """
    from scipy.interpolate import LSQUnivariateSpline

    x = np.hstack([x, x, x])
    y = np.hstack([y, y, y])

    t = np.linspace(-1, 2, len(x), endpoint=False)
    tp = np.linspace(0, 1, 50)

    t_knots = np.linspace(t[1], t[-2], 10)
    xs = LSQUnivariateSpline(t, x, t_knots, k=5)(tp)
    ys = LSQUnivariateSpline(t, y, t_knots, k=5)(tp)

    ax.plot(xs, ys, **kwargs)


def make_summary(instrument):
    log.info(f"Preparing summary: {instrument}")
    path = f"results/fit_{instrument}.rst"

    tab = Table.read(path, format="ascii")
    tab.add_index("name")
    dt = "U30"
    comp_tab = Table(names=("Param", "joint crab paper", "gammapy"), dtype=[dt, dt, dt])
        
    filename = make_path("$JOINT_CRAB/results/fit/")
    filename = filename / f"fit_{instrument}.yaml"
    with open(filename,'r') as file:
         paper_result = yaml.safe_load(file)
 
    for par in paper_result['parameters']:
        if par['name'] is not 'reference':
            name = par['name']
            ref = par['value']
            ref_error = par['error']
            value = tab.loc[name]["value"]*np.log(10)
            error = tab.loc[name]["error"]*np.log(10)
            comp_tab.add_row([name, f"{ref}±{ref_error}", f"{value}±{error}"])

    # Generate README.md file with table and plots
    path = f"{instrument}/spectral_comparison_table.md"
    comp_tab.write(path, format="ascii.html", overwrite=True)

    txt = Path(f"{instrument}/spectral_comparison_table.md").read_text()
    out = txt
    Path(f"{instrument}/README.md").write_text(out)


    

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

    ref_model = SkyModels.read(f"{target}/reference/dc1_model_3d.yaml")
    pars = ref_model.parameters.names
    pars.remove("reference")  # need to find a better way to handle this

    for par in pars:
        ref = ref_model.parameters[par].value
        value = tab.loc[par]["value"]
        name = tab.loc[par]["name"]
        error = tab.loc[par]["error"]
        comp_tab.add_row([name, ref, f"{value}±{error}"])


    analysis.datasets["stacked"].counts.sum_over_axes().plot(add_cbar=True)
    plt.savefig(f"{target}/{target}_counts.png", bbox_inches="tight")
    plt.close()

    analysis.datasets["stacked"].plot_residuals(
        method="diff/sqrt(model)", vmin=-0.5, vmax=0.5
    )
    plt.savefig(f"{target}/{target}_residuals.png", bbox_inches="tight")
    plt.close()

    ax_sed, ax_residuals = analysis.flux_points.peek() # Cannot specify flux_unit outputs in cm-2 s-1 TeV-1. Default to erg 
    ref_dict=ref_model.parameters.to_dict() 
    spec_comp_id={'cas_a':2, 'hess_j1702':5} #REally bad hack
    ref_dict_spectral = {'parameters': ref_dict['parameters'][spec_comp_id[target]:] } #keep only spectral parameters. Is there a better way ?

    pwl = PowerLawSpectralModel.from_dict( ref_dict_spectral ) #need to find a way to find spectral model auto
    ax_sed=pwl.plot((0.1,50)*u.TeV, ax=ax_sed, energy_power=2, flux_unit='cm-2 s-1 erg-1', label='Reference', ls='--')
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

    out = txt+im1+im2+im3
    Path(f"{target}/README.md").write_text(out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
