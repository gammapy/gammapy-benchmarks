import pytest
from astropy import units as u
from scipy.stats import norm
from astropy.table import Table
from numpy.testing import assert_allclose
from pathlib import Path
import logging

from gammapy.modeling.models import Models

# config
THIS_FOLDER = Path(__file__).resolve().parent
LIVETIME = 1 * u.hr

AVAILABLE_MODELS = ["point-pwl", "point-ecpl", "point-log-parabola",
                    "point-pwl2", "point-ecpl-3fgl", "point-ecpl-4fgl",
                    "point-template", "diffuse-cube",
                    "disk-pwl", "gauss-pwl"]


def dict_model(model):
    if model == "point-pwl":
        dico = {
            'index': [1.9371396104401883, 0.06361042814531495],
            'amplitude': [9.446665630440995e-13, 9.495928455508491e-14],
            'lon_0': [-0.00016134594628571756, 0.0035583626437786283],
            'lat_0': [0.003347677824804991, 0.0035084950906510977],
        }

    if model == "point-ecpl":
        dico = {
            'index': [2.03293474932192, 0.0936744022552162],
                'amplitude': [8.063750165568713e-13, 8.894287365426223e-14],
                'lambda_': [0.04367859866784394, 0.01300420813421953],
                'alpha': [4.55122490355222, 6.230981036156794],
                'lon_0': [0.0023958800746886086, 0.004322228369704309],
                'lat_0': [0.0020927243057559685, 0.004686464372388325],
        }

    if model == "point-log-parabola":
        dico = {
            'amplitude': [1.0995786896017883e-12, 1.375746931652359e-13],
                'alpha': [1.8503567846850004, 0.11227219424431928],
                'beta': [0.2136267277722347, 0.06643245808931664],
                'lon_0': [-0.005270016025567908, 0.0037816511278345264],
                'lat_0': [0.000645980766132007, 0.004013094037026454],
        }

    if model == "point-pwl2":
        dico = {
            'amplitude': [9.490568771387954e-13, 8.7353467667155e-14],
                'index': [1.9722827963615606, 0.06326355235753542],
                'lon_0': [-0.0009589927476716934, 0.003178629105505736],
                'lat_0': [-0.0019229980036613449, 0.0033846110629347265],
        }

    if model == "point-ecpl-3fgl":
        dico = {
            'index': [1.8322522465645152, 0.12061624064170963],
                'amplitude': [9.337809982184247e-13, 1.0425335585538515e-13],
                'ecut': [12.375312760465096, 5.522504051736185],
                'lon_0': [0.000649732261371735, 0.003930879015647395],
                'lat_0': [0.0016820870606485696, 0.004174771640757175],
        }

    if model == "point-ecpl-4fgl":
        dico = {
            'amplitude': [7.785965793859072e-13, 2.910364357259499e-13],
                'expfactor': [0.5856199475359893, 1.2561479379236957],
                'index_1': [1.4464423590062163, 1.3735844221037117],
                'index_2': [2.312099016111144, 1.6211806961380666],
                'lon_0': [0.0005886708286564173, 0.006996212314673001],
                'lat_0': [0.007484735718804748, 0.007062140770150318],
        }

    if model == "point-template":
        dico = {
            'norm': [0.9608531643373919, 0.0850648080182836],
                'lon_0': [0.0016638517028239289, 0.0030635134823544935],
                'lat_0': [0.0017497707211191482, 0.0030312710009298646],
        }

    if model == "diffuse-cube":
        dico = {
            'norm': [1.0155626141535683, 0.028705621059615206],
        }

    if model == "disk-pwl":
        dico = {
            'index': [1.8806138128156011, 0.10145988377628408],
            'amplitude': [7.507014490091267e-13, 1.4782813238520706e-13],
            'r_0': [0.3078902265977048, 0.006832840776347008],
        }

    if model == "gauss-pwl":
        dico = {
            'index': [1.829486481664308, 0.16220756621739896],
                'amplitude': [6.804590935158721e-13, 3.0840680953665e-13],
                'lon_0': [0.10243620707244663, 0.08206675748344971],
                'lat_0': [0.20709511516651594, 0.09668326099763286],
                'sigma': [0.330589298365092, 0.07588413369108643],
                'e': [0.0, 0.4121044259520015],
                'phi': [0.0, 1.4142135623730951],
        }

    return dico


def param_sim_model(model):
    dico = dict_model(model)

    filename_ref = THIS_FOLDER / f"../results/models/{model}/fit_{int(LIVETIME.value)}h/best-fit-model_0000.yaml"
    model_ref = Models.read(filename_ref)[1]
    names = model_ref.parameters.free_parameters.names

    for name in names:
        values = model_ref.parameters[name].value
        values_err = model_ref.parameters[name].error

        assert_allclose(values, dico[name][0], rtol=5e-3)
        assert_allclose(values_err, dico[name][1], rtol=5e-3)


def test_model_results():
    for model in AVAILABLE_MODELS:
        param_sim_model(model)


