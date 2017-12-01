import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
from gammapy.image import IACTBasicImageEstimator, SkyImage
from gammapy.data import DataStore
from gammapy.background import RingBackgroundEstimator

N_OBS = 100
OBS_ID = 1320

def run_benchmark():
    # Set up data store and select N_OBS times the observation OBS_ID
    data_store = DataStore.from_dir('$GAMMAPY_EXTRA/test_datasets/cta_1dc/')
    obs_ids = OBS_ID * np.ones(N_OBS)
    obs_list = data_store.obs_list(obs_id=obs_ids)

    target_position = SkyCoord(0, 0, unit='deg', frame='galactic')
    on_radius = 0.2 * u.deg
    on_region = CircleSkyRegion(center=target_position, radius=on_radius)

    bkg_estimator = RingBackgroundEstimator(
        r_in=0.5 * u.deg,
        width=0.2 * u.deg,
    )

    # Define reference image centered on the target
    xref = target_position.galactic.l.value
    yref = target_position.galactic.b.value

    ref_image = SkyImage.empty(
        nxpix=800, nypix=600, binsz=0.02,
        xref=xref, yref=yref,
        proj='TAN', coordsys='GAL',
    )

    exclusion_mask = ref_image.region_mask(on_region)
    exclusion_mask.data = 1 - exclusion_mask.data

    image_estimator = IACTBasicImageEstimator(
        reference=ref_image,
        emin=100 * u.GeV,
        emax=100 * u.TeV,
        offset_max=3 * u.deg,
        background_estimator=bkg_estimator,
        exclusion_mask=exclusion_mask,
    )
    result = image_estimator.run(obs_list)

if __name__ == "__main__":
    run_benchmark()