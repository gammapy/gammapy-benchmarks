# simulate bright sources
from astropy import units as u
from scipy.stats import norm
from astropy.table import Table
from numpy.testing import assert_allclose
from pathlib import Path

from gammapy.modeling.models import Models

# path config
THIS_FOLDER = Path(__file__).resolve().parent


def test_sims():
    model_name = "point-pwltest"
    LIVETIME = 1 * u.hr

    name = f"fit-results-all_{LIVETIME.value:.0f}{LIVETIME.unit}"

    filename = THIS_FOLDER / f"../results/models/{model_name}/{name}.fits.gz"
    results = Table.read(str(filename))

    filename_ref = THIS_FOLDER / f"../models/{model_name}.yaml"
    model_ref = Models.read(filename_ref)[0]
    names = [name for name in results.colnames if "err" not in name]

    dico = {
        'lon_0': [-0.07436674674257586, 0.8972187989963998],
        'lat_0': [0.09308440750573033, 0.9446169745134495],
        'index': [0.0026888056711527543, 1.044217363117438],
        'amplitude': [-0.1931904180829501, 1.0703295695736814],
    }

    for name in names:
        values = results[name]
        values_err = results[name + "_err"]
        par = model_ref.parameters[name]

        if par.frozen:
            continue

        pull = (values - par.value) / values_err

        mu, std = norm.fit(pull)

        assert_allclose(mu, dico[name][0], rtol=5e-1)
        assert_allclose(std, dico[name][1], rtol=5e-1)
