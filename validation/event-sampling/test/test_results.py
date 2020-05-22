# simulate bright sources
import astropy.units as u
import numpy as np
from scipy.stats import norm
from astropy.table import Table
import numpy as np
from numpy.testing import assert_allclose
from pathlib import Path

from gammapy.modeling.models import Models

#
#
#from pathlib import Path
#import logging
#import warnings
#import click
#import multiprocessing
#from itertools import repeat

#from scipy.stats import norm
#import matplotlib.pyplot as plt
#import astropy.units as u
#from astropy.convolution import Tophat2DKernel
#from astropy.coordinates import SkyCoord
#from astropy.table import Table
#from gammapy.data import GTI, Observation, EventList
#from gammapy.datasets import MapDataset, MapDatasetEventSampler
##from gammapy.estimators import LiMaMapEstimator as lima
#from gammapy.estimators import TSMapEstimator as ts
#from gammapy.maps import MapAxis, WcsGeom, Map
#from gammapy.irf import EnergyDispersion2D, load_cta_irfs
#from gammapy.makers import MapDatasetMaker
#from gammapy.modeling import Fit
#from gammapy.modeling.models import Models
#from gammapy.utils.table import table_from_row_data
#from regions import CircleSkyRegion
#
#log = logging.getLogger(__name__)

# path config
BASE_PATH = Path("make.py").parent

def test_sims():
    model_name = "point-pwltest"
    LIVETIME = 1 * u.hr
    
    name = f"fit-results-all_{LIVETIME.value:.0f}{LIVETIME.unit}"

    filename = BASE_PATH / f"results/models/{model_name}/{name}.fits.gz"
    results = Table.read(str(filename))

    filename_ref = BASE_PATH / f"models/{model_name}.yaml"
    model_ref = Models.read(filename_ref)[0]
    names = [name for name in results.colnames if "err" not in name]

    dict = {}
    dict['lon_0'] = [-0.07436674674257586, 0.8972187989963998]
    dict['lat_0'] = [0.09308440750573033, 0.9446169745134495]
    dict['index'] = [0.0026888056711527543, 1.044217363117438]
    dict['amplitude'] = [-0.1931904180829501, 1.0703295695736814]

         
    for name in names:
        values = results[name]
        values_err = results[name + "_err"]
        par = model_ref.parameters[name]
        
        if par.frozen:
            continue

        pull = (values - par.value) / values_err

        mu, std = norm.fit(pull)

        assert_allclose(mu, dict[name][0], rtol=5e-1)
        assert_allclose(std, dict[name][1], rtol=5e-1)



