# simulate bright sources
from pathlib import Path
import logging

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from scipy.stats import norm

from gammapy.cube import MapDataset, MapDatasetEventSampler, MapDatasetMaker, SafeMaskMaker
#from gammapy.cube.tests.test_fit import get_map_dataset
from gammapy.data import GTI, Observation, EventList
from gammapy.maps import MapAxis, WcsGeom, WcsNDMap
from gammapy.irf import load_cta_irfs
from gammapy.modeling import Fit
from gammapy.modeling.models import (
                                     PointSpatialModel,
                                     GaussianSpatialModel,
                                     PowerLawSpectralModel,
                                     SkyModel,
                                     SkyModels,
                                     )

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

##############
exp = 1.0
src_morph = 'point'
src_spec = 'pwl'

BASE_PATH = Path(__file__).parent


path = "$GAMMAPY_VALIDATION/gammapy-benchmarks/validation/event-sampling/"

model_path = "/Users/fabio/LAVORO/CTA/GAMMAPY/GIT/gammapy-benchmarks/validation/event-sampling/models/"+src_morph+"-"+src_spec+"/"+src_morph+"-"+src_spec+".yaml"
model_fit_path = "/Users/fabio/LAVORO/CTA/GAMMAPY/GIT/gammapy-benchmarks/validation/event-sampling/results/models/"+src_morph+"-"+src_spec+"/"+src_morph+"-"+src_spec+".yaml"


events_path = path+"/models/"+src_morph+"-"+src_spec+"/events_"+str(int(exp))+"hr.fits.gz"


##############

ENERGY_AXIS = MapAxis.from_bounds(0.1, 300, nbin=30, unit="TeV", name="energy", interp="log")
ENERGY_AXIS_TRUE = MapAxis.from_bounds(0.1, 300, nbin=30, unit="TeV", name="energy", interp="log")

POINTING = SkyCoord(0.0, 0.0, frame="galactic", unit="deg")
WCS_GEOM = WcsGeom.create(skydir=POINTING, width=(6, 6), binsz=0.02, coordsys="GAL", axes=[ENERGY_AXIS])


LIVETIME = 1 * u.hr
GTI_TABLE = GTI.create(start=0 * u.s, stop=LIVETIME.to(u.s))

filename = "data/dataset_{value:.0f}_{unit}.fits.gz".format(value=LIVETIME.value, unit=LIVETIME.unit)
DATASET_PATH = BASE_PATH / filename


def prepare_dataset():
    # read irfs create observation with a single pointing
    # choose some geom, rather fine energy binnning at least 10 bins / per decade
    # computed reduced dataset see e.g. https://docs.gammapy.org/0.15/notebooks/simulate_3d.html#Simulation
    # write dataset to data/dataset-{livetime}.fits.gz

    irfs = load_cta_irfs(
                     "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
                     )

    observation = Observation.create(obs_id=1001, pointing=POINTING, livetime=LIVETIME, irfs=irfs)

    empty = MapDataset.create(WCS_GEOM)
    maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
    maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=4.0 * u.deg)
    dataset = maker.run(empty, observation)
    dataset = maker_safe_mask.run(dataset, observation)

    log.info(f"Writing {DATASET_PATH}")
    dataset.write(DATASET_PATH, overwrite=True)
    return observation


def simulate_events(filename_model, observation):
    """Simulate events for a given model and dataset."""
    # read dataset using MapDataset.read()
    # read model from model.yaml using SkyModels.read()
    # set the model on the dataset write
    # simulate events and write them to data/models/your-model/events-1.fits
    # optionally : bin events here and write counts map to data/models/your-model/counts-1.fits
    
    dataset = MapDataset.read(DATASET_PATH)

    models = SkyModels.read(filename_model)
    dataset.models = models

    events = MapDatasetEventSampler(random_state=0)
    events = events.run(dataset, observation)

    model_str = filename_model.name.replace(filename_model.suffix, "")
    filename = f"data/models/{model_str}/events.fits.gz"
    path = BASE_PATH / filename
    log.info(f"Writing {path}")
    events.table.writeto(str(path), overwrite=True)


def fit_model(dataset, events, model):
    # read dataset using MapDataset.read()
    # read events using EventList.read()
    # bin events into datasets using WcsNDMap.fill_events(events)
    # read reference model and set it on the dataset
    # fit and write best-fit model
    
    dataset = MapDataset.read(dataset)
    event = EventList.read(events)
    model_simu = SkyModels.read(model)
    model_fit = SkyModels.read(model)

#    model_fit = model_simu[0].copy
    dataset.models = model_fit
    dataset.fake()
    
    background_model = dataset.background_model
    background_model.parameters["norm"].value = 1.0
    background_model.parameters["norm"].frozen = True
    background_model.parameters["tilt"].frozen = True

    fit = Fit([dataset])
    result = fit.run(optimize_opts={"print_level": 1})

    # TODO: use log.info() instead
    print("True model: \n", model_simu, "\n\n Fitted model: \n", model_fit)
    result.parameters.to_table()

    covar = result.parameters.get_subcovariance(model_fit[0].spectral_model.parameters)
    
    model_fit.write(path+"/results/models/"+src_morph+"-"+src_spec+"/"+src_morph+"-"+src_spec+".yaml", overwrite=True)

    return covar
#    pass

def plot_results(dataset, model, best_fit_model, covar_matrix):
    # read model and best-fit model
    # write to results folder
    # compare the spectra
    # plot summed residuals
    # plot residual significance distribution and check for normal distribution
    # compare best fit values by writting to model.yaml

    model = SkyModels.read(model)
    best_fit_model = SkyModels.read(best_fit_model)
    best_fit_model[0].spectral_model.parameters.covariance = covar_matrix

    # plot spectral models
    ax1 = model[0].spectral_model.plot(energy_range=(0.1,300)*u.TeV, label='Sim. model')
    ax2 = best_fit_model[0].spectral_model.plot(energy_range=(0.1,300)*u.TeV, label='Best-fit model')
    ax3 = best_fit_model[0].spectral_model.plot_error(energy_range=(0.1,300)*u.TeV)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.savefig("/Users/fabio/LAVORO/CTA/GAMMAPY/GIT/gammapy-benchmarks/validation/event-sampling/results/models/"+src_morph+"-"+src_spec+"/"+src_morph+"-"+src_spec+"_"+str(int(exp))+"hr.eps", format='eps', dpi=1000)
    plt.gcf().clear()
    plt.close

    # plot residuals
    dataset = MapDataset.read(dataset)
    dataset.models = best_fit_model
    dataset.fake()
    dataset.plot_residuals(method="diff/sqrt(model)", vmin=-0.5, vmax=0.5)
    plt.savefig("/Users/fabio/LAVORO/CTA/GAMMAPY/GIT/gammapy-benchmarks/validation/event-sampling/results/models/"+src_morph+"-"+src_spec+"/"+src_morph+"-"+src_spec+"_"+str(int(exp))+"hr_residuals.eps", format='eps', dpi=100)
    plt.gcf().clear()
    plt.close
    
    # plot residual significance distribution
    resid = dataset.residuals()
    sig_resid = resid.data[np.isfinite(resid.data)]
    
    plt.hist(
             sig_resid,
             density=True,
             alpha=0.5,
             color="red",
             bins=100,
             )
             
    mu, std = norm.fit(sig_resid)
    # replace with log.info()
    print("Fit results: mu = {:.2f}, std = {:.2f}".format(mu, std))
    x = np.linspace(-8, 8, 50)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, lw=2, color="black")
    plt.legend()
    plt.xlabel("Significance")
    plt.yscale("log")
    plt.ylim(1e-5, 1)
    xmin, xmax = np.min(sig_resid), np.max(sig_resid)
    plt.xlim(xmin, xmax)
    plt.savefig("/Users/fabio/LAVORO/CTA/GAMMAPY/GIT/gammapy-benchmarks/validation/event-sampling/results/models/"+src_morph+"-"+src_spec+"/"+src_morph+"-"+src_spec+"_"+str(int(exp))+"hr_resid_distrib.eps", format='eps', dpi=100)
    plt.gcf().clear()
    plt.close
    
    pass


if __name__ == "__main__":
    observation = prepare_dataset()

    for filename_model in (BASE_PATH / "models").glob("*.yaml"):
        simulate_events(filename_model, observation)
    #covar = fit_model(dataset_path, events_path, model_path)
    #plot_results(dataset_path, model_path, model_fit_path, covar)
