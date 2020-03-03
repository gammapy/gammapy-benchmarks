"""Extract 1D spectrum information for Fermi-LAT"""
import logging
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
from gammapy.irf import EffectiveAreaTable, EDispKernel
from gammapy.maps import CountsSpectrum
from gammapy.datasets import SpectrumDataset, SpectrumDatasetOnOff, MapDataset
from gammapy.irf import PSFKernel
from gammapy.data import EventList, DataStore, GTI
from gammapy.maps import HpxNDMap, MapAxis, Map, WcsGeom, MapCoord
from gammapy.irf import EnergyDependentTablePSF
from gammapy.utils.time import time_ref_from_dict

log = logging.getLogger(__name__)   
    

class FermiDatasetMaker:
    """Simple class to perform aperture photometry and produce SpectrumDatasetOnOff"""
    def __init__(self,
                 evt_file="$JOINT_CRAB/data/fermi/events.fits.gz",
                 exp_file="$JOINT_CRAB/data/fermi/exposure_cube.fits.gz",
                 psf_file="$JOINT_CRAB/data/fermi/psf.fits.gz",
                 max_psf_radius='0.5 deg'
                ):
        # Read data
        self.events = EventList.read(evt_file)
        self.exposure = HpxNDMap.read(exp_file)
        self.exposure.unit = u.Unit('cm2s')   # no unit stored on map...
        self.psf = EnergyDependentTablePSF.read(psf_file)
                
    def _make_gti(self):
        # Tentative extraction of the GTI
        tstart = self.events.table.meta['TSTART']*u.s
        tstop = self.events.table.meta['TSTOP']*u.s
        time_ref = time_ref_from_dict(self.events.table.meta)
        return GTI.create(tstart, tstop, time_ref)
    
    def _fill_psfmap(self, psf, dataset):
        # Fill each bin of the PSFMap with the psf table.
        energy = dataset.psf.psf_map.geom.get_axis_by_name('energy_true')
        theta = dataset.psf.psf_map.geom.get_axis_by_name('theta')

        values = psf.evaluate(
                            energy=energy.center,
                            rad=theta.center,
                            method='linear'
        )
        
        dataset.psf.psf_map.quantity *= 0
        dataset.psf.psf_map.quantity += values[:,:,np.newaxis,np.newaxis]
        
    def run(self, geom):
        """Create and fill the map dataset"""
        dataset = MapDataset.create(geom, binsz_irf=1.0)
        dataset.counts.fill_events(self.events)

        dataset.gti = self._make_gti()
       
        self._fill_psfmap(self.psf, dataset)
        
        # recompute exposure on geom
        coords = geom.get_coord()
        # this is to change the axis name. Can we avoid this?
        coords = MapCoord.create(dict(skycoord=coords.skycoord, energy_true=coords['energy']))
        values = self.exposure.interp_by_coord(coords)
        dataset.exposure = Map.from_geom(geom, data=values, unit=self.exposure.unit)

        # Not the real Fermi-LAT EDISP: Use 5% energy resolution as approximation
        energy = geom.axes[0]
        edisp = EDispKernel.from_gauss(
            e_true=energy.edges, e_reco=energy.edges, sigma=0.05, bias=0
        )
        dataset.edisp = edisp
 
        return dataset
    
def extract_spectrum_fermi(on_region, off_region, energy, containment_correction):
    """Perform the spectral extraction at target_position for a circular region."""
    
    geom = WcsGeom.create(skydir=on_region.center,width='5 deg', binsz=0.01, axes=[energy])

    ds = FermiDatasetMaker().run(geom)
    
    spec_dataset = ds.to_spectrum_dataset(
        on_region,
        containment_correction=containment_correction
    )
    on_mask=ds.counts.geom.region_mask([on_region])
    on_solid_angle = np.sum(ds.counts.geom.solid_angle()*on_mask)
    
    off_dataset = ds.to_spectrum_dataset(
        off_region,
        containment_correction=False
    )
    off_mask=ds.counts.geom.region_mask([off_region])
    off_solid_angle = np.sum(ds.counts.geom.solid_angle()*off_mask)

    
    return SpectrumDatasetOnOff(
        counts=spec_dataset.counts,
        counts_off=off_dataset.counts,
        gti=spec_dataset.gti,
        aeff=spec_dataset.aeff,
        edisp=spec_dataset.edisp,
        livetime=spec_dataset.livetime,
        acceptance=1,
        acceptance_off=(off_solid_angle/on_solid_angle).to_value("")
        )
