import logging
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from pathlib import Path
from scipy.stats import norm
from astropy.coordinates import SkyCoord
from astropy.convolution import Tophat2DKernel
from regions import CircleSkyRegion

from gammapy.detect import compute_lima_on_off_image
from gammapy.data import DataStore
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.cube import (
    MapDatasetMaker,
    SafeMaskMaker,
    MapDataset,
    MapDatasetOnOff,
    RingBackgroundMaker,
)

PATH_RESULTS = Path("./results/")
OBS_IDS = [23592, 23523, 23526, 23559]
EMIN, EMAX = [0.1, 10] * u.TeV
NBINS = 5 
TARGET_POS = SkyCoord(83.63, 22.01, unit='deg', frame='icrs')
OFFSET_MAX = 2 * u.deg
EXCLUSION_REGION_RAD = 0.5 * u.deg
R_IN = 0.5 * u.deg
WIDTH = 0.2 * u.deg
CORR_RADIUS = 0.1 * u.deg

#  Observation selection
data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
observations = data_store.get_observations(OBS_IDS)

# Run ring background estimation
energy_axis = MapAxis.from_edges(
    np.logspace(np.log10(EMIN.value), np.log10(EMAX.value), NBINS), unit="TeV", name="energy", interp="log"
    )
geom = WcsGeom.create(
    skydir=TARGET_POS,
    binsz=0.02,
    width=(7, 7),
    coordsys="GAL",
    proj="CAR",
    axes=[energy_axis],
)

maker = MapDatasetMaker(geom=geom, offset_max=OFFSET_MAX)

# TODO: As soon as it is implemented, use the edisp bias method as in the reference paper.
# For now, the SafeMaskMaker uses the default which is the 10% aeff method
maker_safe_mask = SafeMaskMaker()

regions = CircleSkyRegion(center=TARGET_POS, radius=EXCLUSION_REGION_RAD)
geom_image = geom.to_image().to_cube([energy_axis.squash()])
exclusion_mask = Map.from_geom(geom_image)
exclusion_mask.data = geom_image.region_mask([regions], inside=False)
ring_maker = RingBackgroundMaker(
    r_in=R_IN, width=WIDTH, exclusion_mask=exclusion_mask
)

stacked_on_off = MapDatasetOnOff.create(geom=geom_image)
for obs in observations:
    dataset = maker.run(obs)
    dataset = maker_safe_mask.run(dataset, obs)

    dataset_image = dataset.to_image()
    dataset_on_off = ring_maker.run(dataset_image)


    stacked_on_off.stack(dataset_on_off)

# Compute excess and Li&Ma significance images:
scale = geom.pixel_scales[0].to("deg")

theta = CORR_RADIUS / scale
tophat = Tophat2DKernel(theta)
tophat.normalize("peak")

lima_maps = compute_lima_on_off_image(
    stacked_on_off.counts,
    stacked_on_off.counts_off,
    stacked_on_off.acceptance,
    stacked_on_off.acceptance_off,
    tophat,
)

significance_map = lima_maps["significance"]
excess_map = lima_maps["excess"]

plt.figure(figsize=(5, 5))
excess_map.get_image_by_idx((0,)).plot(add_cbar=True)
plt.savefig(str(PATH_RESULTS / "ring_excess_map.png"))

# Signficance distribution outside the exclusion region:
significance_map_off = significance_map * exclusion_mask
significance_all = significance_map.data[np.isfinite(significance_map.data)]
significance_off = significance_map_off.data[
    np.isfinite(significance_map_off.data)
]
plt.figure(figsize=(7, 5))
plt.hist(
    significance_all,
    density=True,
    alpha=0.5,
    color="red",
    label="all bins",
    bins=21
)

plt.hist(
    significance_off,
    density=True,
    alpha=0.5,
    color="blue",
    label="off bins",
    bins=21
)

mu, std = norm.fit(significance_off)
x = np.linspace(-8, 8, 50)
p = norm.pdf(x, mu, std)
plt.plot(x, p, lw=2, color="black")
plt.legend()
plt.xlabel("Significance")
plt.yscale("log")
plt.ylim(1e-5, 1)
xmin, xmax = np.min(significance_all), np.max(significance_all)
plt.xlim(xmin, xmax)
plt.savefig(str(PATH_RESULTS / "ring_significance_histogram.png"))
