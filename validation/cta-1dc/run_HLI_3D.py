#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import sys, time
import numpy as np
from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord
from gammapy.analysis import Analysis, AnalysisConfig


import gammapy

print(gammapy.__version__)


def run_3d(name):
    """
    Run a full 3D analysis using the HLI producing:
    - fit
    - counts, residual maps
    - flux points
    
    Analysis is defined by external config files names
    config_{name}.yaml => where the run selection, ROI parameters are defined
    model_{name}.yaml => where the SkyModel is defined
    TODOs:
    - move the prints to a Log file
    - nicer output for the fitted spectral param
    """
    mode = "3d"
    config_file = f"config{mode}_{name}.yaml"
    model_file = f"model{mode}_{name}.yaml"

    config = AnalysisConfig.from_yaml(config_file)
    analysis = Analysis(config)
    print(config)
    analysis.get_observations()

    conf = config.settings["observations"]["filters"][0]
    nb, lon, lat, rad = (
        len(analysis.observations.ids),
        conf["lon"],
        conf["lat"],
        conf["radius"],
    )
    print(f"{nb} observations found in {rad} around {lon}, {lat} ")

    analysis.get_datasets()
    analysis.set_model(filename=model_file)
    print(analysis.model)
    analysis.run_fit()
    print(analysis.fit_result.parameters.to_table())
    analysis.fit_result.parameters.to_table().write(
        f"results/{name}_{mode}_bestfit.dat", format="ascii", overwrite=True
    )

    plt.figure(figsize=(5, 5))
    analysis.datasets["stacked"].counts.sum_over_axes().plot(add_cbar=True)
    plt.savefig(f"results/{name}_{mode}_counts.png", bbox_inches="tight")

    plt.figure(figsize=(5, 5))
    analysis.datasets["stacked"].plot_residuals(
        method="diff/sqrt(model)", vmin=-0.5, vmax=0.5
    )
    plt.savefig(f"results/{name}_{mode}_residuals.png", bbox_inches="tight")

    analysis.get_flux_points(source=f"{name}")
    analysis.flux_points.write(f"results/{name}_{mode}_fluxpoints.fits")

    plt.figure(figsize=(8, 5))
    ax_sed, ax_residuals = analysis.flux_points.peek()
    plt.savefig(f"results/{name}_{mode}_fluxpoints.png", bbox_inches="tight")


pattern = sys.argv[1]

targets = ["CasA", "J1702"]

if pattern == "all":
    for target in targets:
        run_3d(target)
else:
    if pattern in targets:
        run_3d(pattern)
    else:
        print("Target not yet implemented")
