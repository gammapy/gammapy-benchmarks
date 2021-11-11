from regions import CircleSkyRegion
import time
from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.makers import RingBackgroundMaker
from gammapy.estimators import ExcessMapEstimator, ExcessProfileEstimator
from gammapy.maps import Map
from gammapy.datasets import MapDatasetOnOff
import yaml
from astropy import units as u
from astropy.coordinates import SkyCoord
from gammapy.utils.regions import make_orthogonal_rectangle_sky_regions
from pathlib import Path

def data_prep():
    # source_pos = SkyCoord.from_name("MSH 15-52")
    source_pos = SkyCoord(228.32, -59.08, unit="deg")
    config = AnalysisConfig()
    # Select observations - 2.5 degrees from the source position
    config.observations.datastore = "$GAMMAPY_DATA/hess-dl3-dr1/"
    config.observations.obs_cone = {
        "frame": "icrs",
        "lon": source_pos.ra,
        "lat": source_pos.dec,
        "radius": 2.5 * u.deg,
    }
    config.datasets.type = "3d"
    config.datasets.geom.wcs.skydir = {
        "lon": source_pos.ra,
        "lat": source_pos.dec,
        "frame": "icrs",
    }
    # The WCS geometry - centered on MSH 15-52
    config.datasets.geom.wcs.width = {"width": "3 deg", "height": "3 deg"}

    # The FoV radius to use for cutouts
    config.datasets.geom.wcs.binsize = "0.02 deg"
    config.datasets.geom.selection.offset_max = 3.5 * u.deg

    # We now fix the energy axis for the counts map - (the reconstructed
    # energy binning)
    config.datasets.geom.axes.energy.min = "0.5 TeV"
    config.datasets.geom.axes.energy.max = "5 TeV"
    config.datasets.geom.axes.energy.nbins = 10

    # We need to extract the ring for each observation separately, hence, no
    # stacking at this stage
    config.datasets.stack = False

    # create the config
    analysis = Analysis(config)

    # for this specific case,w e do not need fine bins in true energy
    analysis.config.datasets.geom.axes.energy_true = (
        analysis.config.datasets.geom.axes.energy
    )

    # `First get the required observations
    analysis.get_observations()

    # Analysis extraction
    analysis.get_datasets()
    return analysis


def create_stacked_dataset(analysis):
    # source_pos = SkyCoord.from_name("MSH 15-52")
    source_pos = SkyCoord(228.32, -59.08, unit="deg")

    # get the geom that we use
    geom = analysis.datasets[0].counts.geom
    energy_axis = analysis.datasets[0].counts.geom.axes["energy"]
    geom_image = geom.to_image().to_cube([energy_axis.squash()])

    # Make the exclusion mask
    regions = CircleSkyRegion(center=source_pos, radius=0.3 * u.deg)

    exclusion_mask = geom_image.region_mask([regions], inside=False)

    ring_maker = RingBackgroundMaker(
        r_in="0.5 deg", width="0.3 deg", exclusion_mask=exclusion_mask
    )

    # Creation of the MapDatasetOnOff
    energy_axis_true = analysis.datasets[0].exposure.geom.axes["energy_true"]
    stacked_on_off = MapDatasetOnOff.create(
        geom=geom_image, energy_axis_true=energy_axis_true, name="stacked"
    )

    for dataset in analysis.datasets:
        # Ring extracting makes sense only for 2D analysis
        dataset_on_off = ring_maker.run(dataset.to_image())
        stacked_on_off.stack(dataset_on_off)
    return stacked_on_off


def compute_correlations(stacked_on_off):
    # Using a convolution radius of 0.1 degrees
    estimator = ExcessMapEstimator(0.1 * u.deg)
    lima_maps = estimator.run(stacked_on_off)

    significance_map = lima_maps["sqrt_ts"]
    excess_map = lima_maps.npred_excess
    return significance_map, excess_map

def compute_profile(stacked_on_off):
    wcs = stacked_on_off.counts.geom.wcs
    start_line = SkyCoord(227.3, -58.08, unit="deg", frame="icrs")
    end_line = SkyCoord(229.3, -60.08, unit="deg", frame="icrs")
    boxes = make_orthogonal_rectangle_sky_regions(
        start_line, end_line, wcs, 0.1 * u.deg, 20
    )
    prof_maker = ExcessProfileEstimator(boxes, energy_edges=[0.5, 1, 5] * u.TeV)
    imp_prof = prof_maker.run(stacked_on_off)
    return imp_prof

def run_benchmark():
    info = {}

    t = time.time()
    analysis = data_prep()
    info["data_preparation"] = time.time() - t

    t = time.time()
    stacked_on_off = create_stacked_dataset(analysis)
    info["run_asmooth"] = time.time() - t

    t = time.time()
    compute_correlations(stacked_on_off)
    info["compute_correlations"] = time.time() - t

    t = time.time()
    compute_correlations(stacked_on_off)
    info["compute_ExcessProfileEstimator"] = time.time() - t

    Path("bench.yaml").write_text(yaml.dump(info, sort_keys=False, indent=4))


if __name__ == "__main__":
    run_benchmark()
