import logging
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from gammapy.estimators import FluxPoints
from gammapy.modeling.models import Model

log = logging.getLogger(__name__)


def main():
    # TODO: add the rxj1713 plotting
    targets = ["crab", "msh1552", "pks2155"]
    for target in targets:
        log.info(f"Processing source: {target}")
        path_ref = Path(str(target) + "/reference/")
        path_res = Path(str(target) + "/results/")
        path_plot = Path(str(target) + "/plots/")

        for ndim in [1, 3]:
            # Load the reference and best-fit spectral models
            with open(str(path_ref / f"reference-{ndim}d.yaml")) as file:
                reference_spectrum_file = yaml.safe_load(file)
                reference_spectrum = Model.create(
                    "PowerLawSpectralModel",
                    index=reference_spectrum_file["index"],
                    amplitude=f"{reference_spectrum_file['amplitude']} TeV-1 cm-2 s-1",
                    reference=f"{reference_spectrum_file['reference']} TeV",
                )
            reference_spectrum_errors = {
                "index_err": reference_spectrum_file["index_err"],
                "amplitude_err": reference_spectrum_file["amplitude_err"],
            }
            with open(str(path_res / f"result-{ndim}d.yaml")) as file:
                result_spectrum_file = yaml.safe_load(file)
                result_spectrum = Model.create(
                    "PowerLawSpectralModel",
                    index=result_spectrum_file["index"],
                    amplitude=f"{result_spectrum_file['amplitude']} TeV-1 cm-2 s-1",
                    reference=f"{result_spectrum_file['reference']} TeV",
                )
            result_spectrum_errors = {
                "index_err": result_spectrum_file["index_err"],
                "amplitude_err": result_spectrum_file["amplitude_err"],
            }

            # Load the reference and best-fit flux points
            reference_fpoints = (
                str(path_ref) + f"/gammapy_{target}_{ndim}d_spectral_points.ecsv"
            )
            reference_fpoints = FluxPoints.read(reference_fpoints)
            result_fpoints = str(path_res) + f"/flux-points-{ndim}d.ecsv"
            result_fpoints = FluxPoints.read(result_fpoints)

            # Plot
            fig = make_plots(
                reference_spectrum,
                reference_spectrum_errors,
                result_spectrum,
                result_spectrum_errors,
                reference_fpoints,
                result_fpoints,
            )
            log.info(f"Writing {path_plot}")
            fig.savefig(str(path_plot) + f"/flux-points-{ndim}d.png")


def make_plots(
    reference_spectrum,
    reference_spectrum_errors,
    result_spectrum,
    result_spectrum_errors,
    reference_fpoints,
    result_fpoints,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.rc("text", usetex=True)
    opts = {"energy_power": 2, "ax": ax}

    reference_spectrum.plot(
        **opts,
        energy_range=["0.5 TeV", "50 TeV"],
        color="red",
        alpha=0.8,
        label=f"Reference: \n    -$\Gamma$={reference_spectrum.index.value:5.2f} $\pm$ {reference_spectrum_errors['index_err']:5.2f},"
        f"\n    -$\Phi_0$: {reference_spectrum.amplitude.value:5.2e} $\pm$ {reference_spectrum_errors['amplitude_err']:5.2e} "
        f"TeV-1 cm-2 s-1",
    )
    result_spectrum.plot(
        **opts,
        energy_range=["0.5 TeV", "50 TeV"],
        color="blue",
        alpha=0.8,
        label=f"Validation: \n    -$\Gamma$={result_spectrum.index.value:5.2f} $\pm$ {result_spectrum_errors['index_err']:5.2f},"
        f"\n    -$\Phi_0$: {result_spectrum.amplitude.value:5.2e} $\pm$ {result_spectrum_errors['amplitude_err']:5.2e} "
        f"TeV-1 cm-2 s-1",
    )

    reference_fpoints.plot(**opts, color="red")
    result_fpoints.plot(**opts, color="blue")
    plt.legend()

    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
