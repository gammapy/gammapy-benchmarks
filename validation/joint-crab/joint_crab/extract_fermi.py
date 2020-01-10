"""Extract 1D spectrum information for Fermi-LAT"""
import logging
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
from gammapy.irf import EffectiveAreaTable, EDispKernel
from gammapy.spectrum import CountsSpectrum, SpectrumDataset, SpectrumDatasetOnOff
from gammapy.cube import MapDataset, PSFKernel
from regions import CircleSkyRegion, CircleAnnulusSkyRegion
from gammapy.data import EventList, DataStore, GTI
from gammapy.maps import HpxNDMap, MapAxis, Map, WcsGeom
from gammapy.irf import EnergyDependentTablePSF
from gammapy.utils.time import time_ref_from_dict

log = logging.getLogger(__name__)   
    

class FermiAnalysis1D:
    """Simple class to perform aperture photometry and produce SpectrumDatasetOnOff"""
    def __init__(self,
                 geom,
                 evt_file="$JOINT_CRAB/data/fermi/events.fits.gz",
                 exp_file="$JOINT_CRAB/data/fermi/exposure_cube.fits.gz",
                 psf_file="$JOINT_CRAB/data/fermi/psf.fits.gz",
                 max_radius='0.5 deg'
                ):
        log.info("Extracting 1d spectra for Fermi-LAT")
        # Read data
        self.events = EventList.read(evt_file)
        self.exposure = HpxNDMap.read(exp_file)
        self.exposure.unit = u.Unit('cm2s')   # no unit stored on map...
        self.psf = EnergyDependentTablePSF.read(psf_file)
        self.geom = geom
        self.dataset = self._make_mapdataset(max_radius)
        
    def _make_gti(self):
        # Tentative extraction of the GTI
        tstart = self.events.table.meta['TSTART']*u.s
        tstop = self.events.table.meta['TSTOP']*u.s
        time_ref = time_ref_from_dict(self.events.table.meta)
        return GTI.create(tstart, tstop, time_ref)
    
    def _fill_psfmap(self, psf, dataset):
        # Fill each bin of the PSFMap with the psf table.
        coord = dataset.psf.psf_map.geom.get_coord()
        values = psf.evaluate(
                            energy=coord['energy'],
                            rad=coord['theta'],
                            method='linear'
        )
        dataset.psf.psf_map.quantity = values
        
    def _make_mapdataset(self, max_radius):
        """Create and fill the map dataset"""
        dataset = MapDataset.create(self.geom, binsz_irf=1.0)
        dataset.counts.fill_events(self.events)

        dataset.gti = self._make_gti()
       
        self._fill_psfmap(self.psf, dataset)
        
        # recompute exposure on geom
        coords = self.geom.get_coord()
        values = self.exposure.interp_by_coord(coords)
        dataset.exposure = Map.from_geom(geom, data=values, unit=self.exposure.unit)

        # Not the real Fermi-LAT EDISP: Use 5% energy resolution as approximation
        energy = self.geom.axes[0]
        edisp = EDispKernel.from_gauss(
            e_true=energy.edges, e_reco=energy.edges, sigma=0.05, bias=0
        )
        dataset.edisp = edisp
 
        return dataset
    
    def run(target_position, on_radius, off_region):
        """Perform the spectral extraction at target_position for a circular region."""

        # create dummy WcsGeom for event selection. Center it on the Crab
        geom = WcsGeom.create(skydir=target_position,width='10 deg', binsize='0.01 deg')
    
        # Select ON and OFF events
        on_events = events.select_region(on_region, geom.wcs)
    
        if 
        off_events = events.select_region(off_region, geom.wcs)

        
    
    
def extract_spectra_fermi(target_position, on_radius):
    """Extract 1d spectra for Fermi-LAT"""
    log.info("Extracting 1d spectra for Fermi-LAT")
    events = EventList.read()
    exposure = HpxNDMap.read()
    psf = EnergyDependentTablePSF.read("$JOINT_CRAB/data/fermi/psf.fits.gz")

#    valid_range = (config.energy_bins >= 30 * u.GeV) * (config.energy_bins <= 2 * u.TeV)
    crab_pos = SkyCoord(ra=83.63, dec=22.01, unit='deg', frame='icrs')
    on_radius = Angle('0.3 deg')
    on_region = CircleSkyRegion(center=crab_pos, radius=on_radius)
    
    # create dummy WcsGeom for event selection. Center it on the Crab
    geom = WcsGeom.create(skydir=crab_pos,width='10 deg', binsize='0.01 deg')
    
    # Select ON and OFF events
    on_events = events.select_region(on_region, geom.wcs)
    off_region = CircleAnnulusSkyRegion(crab_pos, 1 * u.deg, 2 * u.deg)
    off_events = events.select_region(off_region, geom.wcs)

    # Create dataset
    e_reco = MapAxis.from_bounds(0.01, 100, 80, unit='TeV', interp='log', name='energy')
    dataset = SpectrumDatasetOnOff.create(e_reco, e_reco, on_region)

    # Fill counts and counts_off
    dataset.counts.fill_events(on_events)
    dataset.counts_off.fill_events(events_off)
    
    # Add edisp
    # Not the real Fermi-LAT EDISP: Use 5% energy resolution as approximation
    edisp = EDispKernel.from_gauss(
            e_true=e_reco.edges, e_reco=e_reco.edges, sigma=0.05, bias=0
        )
    dataset.edisp = edisp
    
    # Add aeff
    aeff = extract_aeff(exposure, crab_pos, e_reco.edges)

    containment_factor = extract_psf_containment(psf, on_radius, e_reco.edges)
    aeff.data.data *= containment_factor

    dataset.aeff = aeff
    # Here we cheat. aeff is an exposure (in cm2.s) which is stored as an EffectiveArea.
    # We have to put a livetime value. Here 1 sec.
    dataset.livetime = 1.0 * u.s

    path = "results/spectra/fermi"
    log.info(f"Writing to {path}")
    dataset.write(path, use_sherpa=True, overwrite=True)

def extract_aeff(exposure, target_position, energy):
    energy_log_ctr = np.sqrt(energy[1:] * energy[:-1])
    lon = target_position.galactic.l
    lat = target_position.galactic.b
    expo_values = exposure.get_by_coord((lon.value, lat.value, energy_log_ctr.value))

    table = Table(
        [energy[:-1], energy[1:], expo_values],
        names=("ENERG_LO", "ENERG_HI", "SPECRESP"),
        dtype=("float64", "float64", "float64"),
        meta={"name": "Fermi-LAT exposure"},
    )

    table["ENERG_LO"].unit = str(energy.unit)
    table["ENERG_HI"].unit = str(energy.unit)
    table["SPECRESP"].unit = "cm2"

    table.meta["EXTNAME"] = "SPECRESP"
    table.meta["TELESCOP"] = "Fermi"
    table.meta["INSTRUME"] = "LAT"
    table.meta["EXPOSURE"] = "1"
    table.meta["FILTER"] = ""
    table.meta["HDUCLASS"] = "OGIP"
    table.meta["HDUCLAS1"] = "RESPONSE"
    table.meta["HDUCLAS2"] = "SPECRESP"
    table.meta["HDUVERS"] = "1.1.0"

    return EffectiveAreaTable.from_table(table)

def extract_psf_containment(psf, on_radius, energy):
    energy_log_ctr = np.sqrt(energy[:-1] * energy[1:])
    containment_factor = np.asarray(
        [psf.integral(_, rad_min="0 deg", rad_max=on_radius) for _ in energy_log_ctr]
    )
    return containment_factor


class SpectrumExtractionFermi1D(object):
    def __init__(
        self, events, exposure, psf, bkg_estimate, target_position, on_radius, energy
    ):
        self.events = events
        self.exposure = exposure
        self.psf = psf
        self.bkg_estimate = bkg_estimate
        self.target_position = target_position
        self.on_radius = on_radius
        self.energy = energy.to("MeV")

    def make_empty_vectors(self, bkg_estimate):
        self._on_vector = PHACountsSpectrum(
            energy_lo=self.energy[:-1],
            energy_hi=self.energy[1:],
            backscal=bkg_estimate.a_on,
            obs_id=0,
        )

        self._off_vector = self._on_vector.copy()
        self._off_vector.is_bkg = True
        self._off_vector.backscal = bkg_estimate.a_off
        # here we set the livetime of 1s, because we are actually storing an
        # exposure rather than an effective area
        self._on_vector.livetime = 1.0 * u.s
        self._off_vector.livetime = 1.0 * u.s

    def extract_counts(self, bkg_estimate):
        self._on_vector.fill(bkg_estimate.on_events)
        self._off_vector.fill(bkg_estimate.off_events)

    def run(self):
        
        self.make_empty_vectors(self.bkg_estimate)
        self.extract_counts(self.bkg_estimate)

        aeff = extract_aeff(self.exposure, self.target_position, self.energy)

        containment_factor = extract_psf_containment(
            self.psf, self.on_radius, self.energy
        )
        aeff.data.data *= containment_factor

        # Not the real Fermi-LAT EDISP
        # Use 5% energy resolution as approximation
        edisp = EnergyDispersion.from_gauss(
            e_true=self.energy, e_reco=self.energy, sigma=0.05, bias=0
        )

        return SpectrumObservation(
            on_vector=self._on_vector,
            aeff=aeff,
            off_vector=self._off_vector,
            edisp=edisp,
        )




