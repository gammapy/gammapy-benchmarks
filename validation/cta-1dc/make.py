#!/usr/bin/env python
"""Run Gammapy validation: CTA 1DC"""
import logging
import yaml
import matplotlib.pyplot as plt
from gammapy.analysis import Analysis, AnalysisConfig

def target_config3d(config_file, target_config_file, tag):
    """Create analyis configuration for out source."""
    targets_config_ = yaml.safe_load(open(target_config_file))
    targets_config = {}
    for conf in targets_config_:  # define tag as key
        targets_config[conf["tag"]] = conf

    config = AnalysisConfig.from_yaml(config_file)
    config_dict = config.settings

    config_dict["observations"]["filters"][0]["frame"] = targets_config[tag]["frame"]
    config_dict["observations"]["filters"][0]["lon"] = targets_config[tag]["lon"]
    config_dict["observations"]["filters"][0]["lat"] = targets_config[tag]["lat"]
    config_dict["observations"]["filters"][0]["radius"] = targets_config[tag]["radius"]
    config_dict["observations"]["filters"][0]["border"] = targets_config[tag]["radius"]

    config_dict["datasets"]["geom"]["skydir"] = [
        float(targets_config[tag]["lon"].strip(" deg")),
        float(targets_config[tag]["lat"].strip(" deg")),
    ]
    config_dict["datasets"]["geom"]["axes"][0]["lo_bnd"] = targets_config[tag]["emin"]
    config_dict["datasets"]["geom"]["axes"][0]["hi_bnd"] = targets_config[tag]["emax"]
    config_dict["datasets"]["geom"]["axes"][0]["nbin"] = targets_config[tag]["nbin"]
    config_dict["datasets"]["geom"]["axes"][0]["nbin"] = targets_config[tag]["nbin"]

    config_dict["flux-points"]["fp_binning"]["lo_bnd"] = targets_config[tag]["emin"]
    config_dict["flux-points"]["fp_binning"]["hi_bnd"] = targets_config[tag]["emax"]
    config_dict["flux-points"]["fp_binning"]["nbin"] = targets_config[tag]["nbin"]

    config_dict["flux-points"]["fp_binning"]["lo_bnd"] = targets_config[tag]["emin"]
    config_dict["flux-points"]["fp_binning"]["hi_bnd"] = targets_config[tag]["emax"]
    config_dict["flux-points"]["fp_binning"]["nbin"] = targets_config[tag]["nbin"]

    config_dict["fit"]["fit_range"]["min"] = str(targets_config[tag]["emin"]) + " TeV"
    config_dict["fit"]["fit_range"]["max"] = str(targets_config[tag]["emax"]) + " TeV"

    config.update_settings(config=config_dict)

    return config


def run_3d(name):
    """Run 3D analysis for one source."""
    logging.info(f"run3d: {name}")
    mode = "3d"
    config_file = f"config{mode}.yaml"
    target_config_file = f"targets.yaml"
    model_file = f"model{mode}_{name}.yaml"

    outdir = f"results/{name}"

    config = target_config3d(config_file, target_config_file, name)
    analysis = Analysis(config)
    analysis.get_observations()

    conf = config.settings["observations"]["filters"][0]
    nb, lon, lat, rad = (
        len(analysis.observations.ids),
        conf["lon"],
        conf["lat"],
        conf["radius"],
    )
    logging.info(f"{nb} observations found in {rad} around {lon}, {lat} ")

    analysis.get_datasets()

    # test
    plt.figure(figsize=(5, 5))
    analysis.datasets["stacked"].counts.sum_over_axes().plot(add_cbar=True)
    plt.savefig(f"{outdir}/{name}_{mode}_counts.png", bbox_inches="tight")

    analysis.set_model(filename=model_file)
    logging.info(analysis.model)
    analysis.run_fit()
    logging.info(analysis.fit_result.parameters.to_table())
    analysis.fit_result.parameters.to_table().write(
        f"{outdir}/{name}_{mode}_bestfit.dat", format="ascii", overwrite=True
    )

    analysis.get_flux_points(source=f"{name}")
    analysis.flux_points.write(f"{outdir}/{name}_{mode}_fluxpoints.fits")

    plt.figure(figsize=(5, 5))
    analysis.datasets["stacked"].counts.sum_over_axes().plot(add_cbar=True)
    plt.savefig(f"{outdir}/{name}_{mode}_counts.png", bbox_inches="tight")

    plt.figure(figsize=(5, 5))
    analysis.datasets["stacked"].plot_residuals(
        method="diff/sqrt(model)", vmin=-0.5, vmax=0.5
    )
    plt.savefig(f"{outdir}/{name}_{mode}_residuals.png", bbox_inches="tight")

    plt.figure(figsize=(8, 5))
    ax_sed, ax_residuals = analysis.flux_points.peek()
    plt.savefig(f"{outdir}/{name}_{mode}_fluxpoints.png", bbox_inches="tight")


def main():
    targets = "all"
    if targets == "all":
        targets = ["cas_a", "hess_j1702"]
    else:
        targets = targets.split(",")

    for target in targets:
        run_3d(target)

if __name__ == "__main__":
    main()
