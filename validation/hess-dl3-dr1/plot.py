import logging
import matplotlib.pyplot as plt
from gammapy.spectrum import FluxPoints
from pathlib import Path

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
            reference = (
                str(path_ref) + f"/gammapy_{target}_{ndim}d_spectral_points.ecsv"
            )
            result = str(path_res) + f"/flux-points-{ndim}d.ecsv"
            fig = make_plots(reference, result)
            log.info(f"Writing {path_plot}")
            fig.savefig(str(path_plot) + f"/flux-points-{ndim}d.png")


def make_plots(reference, result):
    fpoints_ref = FluxPoints.read(reference)
    fpoints_res = FluxPoints.read(result)

    fig = plt.figure(figsize=(7, 5))
    opts = {"energy_power": 2}
    fpoints_ref.plot(**opts, label="reference")
    fpoints_res.plot(**opts)
    plt.legend()

    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
