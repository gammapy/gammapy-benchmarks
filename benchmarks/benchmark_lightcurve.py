import numpy as np
from gammapy.data import DataStore
from gammapy.maps import MapAxis, WcsGeom
from gammapy.cube import MapMaker
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from gammapy.background import ReflectedRegionsBackgroundEstimator
from gammapy.utils.energy import EnergyBounds
from gammapy.spectrum import SpectrumExtraction
from gammapy.spectrum.models import PowerLaw
from gammapy.time import LightCurveEstimator

N_OBS = 100
OBS_ID = 23523


def run_benchmark():

    data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")

    obs_ids = OBS_ID * np.ones(N_OBS)
    observations = data_store.get_observations(obs_ids)

    target_position = SkyCoord(ra=83.63308, dec=22.01450, unit="deg")
    on_region_radius = Angle("0.2 deg")
    on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)

    bkg_estimator = ReflectedRegionsBackgroundEstimator(
        on_region=on_region, observations=observations
    )
    bkg_estimator.run()

    ebounds = EnergyBounds.equal_log_spacing(0.7, 100, 50, unit="TeV")
    extraction = SpectrumExtraction(
        observations=observations,
        bkg_estimate=bkg_estimator.result,
        containment_correction=False,
        e_reco=ebounds,
        e_true=ebounds,
    )
    extraction.run()
    spectrum_observations = extraction.spectrum_observations

    time_intervals = [(obs.tstart, obs.tstop) for obs in observations]

    spectral_model = PowerLaw(
        index=2, amplitude=2.0e-11 * u.Unit("1 / (cm2 s TeV)"), reference=1 * u.TeV
    )

    energy_range = [1, 100] * u.TeV

    lc_estimator = LightCurveEstimator(extraction)
    lc = lc_estimator.light_curve(
        time_intervals=time_intervals,
        spectral_model=spectral_model,
        energy_range=energy_range,
    )


if __name__ == "__main__":
    run_benchmark()
