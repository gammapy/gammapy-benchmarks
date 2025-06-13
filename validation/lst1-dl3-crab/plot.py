# Licensed under a 3-clause BSD style license - see LICENSE

"""Plot validation results for LST-1 Crab Nebula analysis.

Plot 1D spectral analysis results (best-fit model and flux points)
from Crab LST-1 DL3 release, comparing them with reference results
from ApJ 956 80 (2023).
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from gammapy.estimators import FluxPoints
from gammapy.modeling.models import Models

log = logging.getLogger(__name__)

LST1_WORKING_DIR = Path(__file__).parent

def main():
    """Validate analysis of LST-1 Crab observations."""
    log.info("Creating validation plots for LST-1 Crab 1D spectral analysis")
    path_ref = LST1_WORKING_DIR / "reference"
    path_res = LST1_WORKING_DIR / "results"
    path_plot = LST1_WORKING_DIR / "plots"
    path_plot.mkdir(parents=True, exist_ok=True)

    # Load the reference and best-fit spectral models
    reference_spectrum = Models.read(path_ref / "spectral_model.yml")[0].spectral_model
    result_spectrum = Models.read(path_res / "best_fit_model.yml")[0].spectral_model

    # Load the reference and best-fit flux points
    reference_fpoints = FluxPoints.read(
        path_ref / "flux_points_lst1.ecsv",
        sed_type="e2dnde",
        format="gadf-sed",
        reference_model=reference_spectrum,
    )
    result_fpoints = FluxPoints.read(
        path_res / "flux-points.ecsv",
        sed_type="e2dnde",
        format="gadf-sed",
        reference_model=result_spectrum,
    )

    # Plot
    fig = make_plots(
        reference_spectrum,
        result_spectrum,
        reference_fpoints,
        result_fpoints,
    )
    log.info("Writing plot to %s", path_plot)
    fig.savefig(path_plot / "flux-points.png")


def make_plots(
    reference_spectrum,
    result_spectrum,
    reference_fpoints,
    result_fpoints,
) -> plt.Figure:
    """Make plots comparing reference and validation results.

    Parameters
    ----------
    reference_spectrum : `~gammapy.modeling.models.SpectralModel`
        Reference spectral model from ApJ 956 80 (2023).
    result_spectrum : `~gammapy.modeling.models.SpectralModel`
        Best-fit spectral model.
    reference_fpoints : `~gammapy.estimators.FluxPoints`
        Reference flux points from ApJ 956 80 (2023).
    result_fpoints : `~gammapy.estimators.FluxPoints`
        Best-fit flux points.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        Figure with the plots.

    """
    fig, ax = plt.subplots(figsize=(7, 5))
    opts = {
        "sed_type":"e2dnde",
        "ax": ax,
    }
    energy_bounds = ["50 GeV", "30 TeV"]

    text_label_reference = (
        "Reference:"
        "\n"  
        rf"$\alpha$={reference_spectrum.alpha.value:.2f}"
        rf" $\pm$ {reference_spectrum.alpha.error:.2f},"
        "\n"
        rf"$\beta$={reference_spectrum.beta.value:.2f}"
        rf" $\pm$ {reference_spectrum.beta.error:.2f},"
        "\n"
        rf"$\Phi_0$={reference_spectrum.amplitude.value:.2e}" 
        rf" $\pm$ {reference_spectrum.amplitude.error:.2e} "
        f"{reference_spectrum.amplitude.unit}"
    )
    text_label_validation = (
        "Validation:"
        "\n"  
        rf"$\alpha$={result_spectrum.alpha.value:.2f}"
        rf" $\pm$ {result_spectrum.alpha.error:.2f},"
        "\n"
        rf"$\beta$={result_spectrum.beta.value:.2f}"
        rf" $\pm$ {result_spectrum.beta.error:.2f},"
        "\n"
        rf"$\Phi_0$={result_spectrum.amplitude.value:.2e}" 
        rf" $\pm$ {result_spectrum.amplitude.error:.2e} "
        f"{result_spectrum.amplitude.unit}"
    )

    reference_spectrum.plot(
        **opts,
        energy_bounds=energy_bounds,
        color="red",
        alpha=0.8,
        label=text_label_reference,
    )
    result_spectrum.plot(
        **opts,
        energy_bounds=energy_bounds,
        color="blue",
        alpha=0.8,
        label=text_label_validation,
    )

    reference_fpoints.plot(**opts, color="red")
    result_fpoints.plot(**opts, color="blue")
    plt.legend()

    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
