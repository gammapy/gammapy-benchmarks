# simulate bright sources
from gammapy.maps import MapAxis, WcsGeom

ENERGY_AXIS = MapAxis.from_bounds()
ENERGY_AXIS_TRUE = MapAxis.from_bounds()
WCS_GEOM = WcsGeom.create(width=(8, 8), binsz=0.02, axes=[ENERGY_AXES])


def prepare_dataset():
    # read irfs create observation with a single pointing
    # choose some geom, rather fine energy binnning at least 10 bins / per decade
    # computed reduced dataset see e.g. https://docs.gammapy.org/0.15/notebooks/simulate_3d.html#Simulation
    # write dataset to data/dataset-{livetime}.fits.gz
    pass

def simulate_events(model):
    # read dataset using MapDataset.read()
    # read model from model.yaml using SkyModels.read()
    # set the model on the dataset write
    # simulate events and write them to data/models/your-model/events-1.fits
    # optionally : bin events here and write counts map to data/models/your-model/counts-1.fits
    pass

def fit_model(dataset, model):
    # read dataset using MapDataset.read()
    # read events using EventList.read()
    # bin events into datasets using WcsNDMap.fill_events(events)
    # read reference model and set it on the dataset
    # fit and write best-fit model
    pass

def plot_results(model, best_fit_model):
    # read model and best-fit model
    # write to results folder
    # compare the spectra
    # plot summed residuals
    # plot residual signifiance distribution and check for normal distribution 
    # compare best fit values by writting to model.yaml 
    pass


if __name__ == "__main__":
    prepare_dataset()
    simulate_events()
    fit_model()
    plot_results()	
