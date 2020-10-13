"""Extract 1D spectrum information for Fermi-LAT"""
import logging

import astropy.units as u
import numpy as np

from gammapy.data import GTI, EventList
from gammapy.datasets import MapDataset, SpectrumDatasetOnOff
from gammapy.irf import EDispKernelMap, EnergyDependentTablePSF
from gammapy.maps import HpxNDMap, Map, MapCoord, WcsGeom
from gammapy.utils.time import time_ref_from_dict

log = logging.getLogger(__name__)


class FermiDatasetMaker:
    """Simple class to perform aperture photometry and produce SpectrumDatasetOnOff"""

    def __init__(
        self,
        evt_file="../data/joint-crab/fermi/events.fits.gz",
        exp_file="../data/joint-crab/fermi/exposure_cube.fits.gz",
        psf_file="../data/joint-crab/fermi/psf.fits.gz",
        max_psf_radius="0.5 deg",
    ):
        # Read data
        self.events = EventList.read(evt_file)
        self.exposure = HpxNDMap.read(exp_file)
        self.exposure.unit = u.Unit("cm2s")  # no unit stored on map...
        self.psf = EnergyDependentTablePSF.read(psf_file)

    def _make_gti(self):
        # Tentative extraction of the GTI
        tstart = self.events.table.meta["TSTART"] * u.s
        tstop = self.events.table.meta["TSTOP"] * u.s
        time_ref = time_ref_from_dict(self.events.table.meta)
        return GTI.create(tstart, tstop, time_ref)

    def _fill_psfmap(self, psf, dataset):
        # Fill each bin of the PSFMap with the psf table.
        energy = dataset.psf.psf_map.geom.axes["energy_true"]
        theta = dataset.psf.psf_map.geom.axes["rad"]

        values = psf.evaluate(energy=energy.center, rad=theta.center, method="linear")

        dataset.psf.psf_map.quantity *= 0
        dataset.psf.psf_map.quantity += values[:, :, np.newaxis, np.newaxis]

    def run(self, geom):
        """Create and fill the map dataset"""
        energy = geom.axes[0]
        energy_true = energy.copy(name="energy_true")
        geom_true = geom.to_image().to_cube([energy_true])

        dataset = MapDataset.create(geom, energy_axis_true=energy_true, binsz_irf=1.0)
        dataset.counts.fill_events(self.events)

        dataset.gti = self._make_gti()

        self._fill_psfmap(self.psf, dataset)

        # recompute exposure on geom
        coords = geom_true.get_coord()
        data = self.exposure.interp_by_coord(coords)

        dataset.exposure = Map.from_geom(geom_true, data=data, unit=self.exposure.unit)

        # Not the real Fermi-LAT EDISP: Use 5% energy resolution as approximation

        edisp = EDispKernelMap.from_gauss(
            energy_axis=energy, energy_axis_true=energy_true, sigma=0.05, bias=0
        )
        dataset.edisp = edisp

        return dataset


def extract_spectrum_fermi(on_region, off_region, energy, containment_correction):
    """Perform the spectral extraction at target_position for a circular region."""

    geom = WcsGeom.create(
        skydir=on_region.center, width="5 deg", binsz=0.01, axes=[energy]
    )

    ds = FermiDatasetMaker().run(geom)

    spec_dataset = ds.to_spectrum_dataset(
        on_region, containment_correction=containment_correction, name="fermi"
    )
    on_mask = ds.counts.geom.region_mask([on_region])
    on_solid_angle = np.sum(ds.counts.geom.solid_angle() * on_mask)

    off_dataset = ds.to_spectrum_dataset(off_region, containment_correction=False)
    off_mask = ds.counts.geom.region_mask([off_region])
    off_solid_angle = np.sum(ds.counts.geom.solid_angle() * off_mask)

    return SpectrumDatasetOnOff(
        counts=spec_dataset.counts,
        counts_off=off_dataset.counts,
        gti=spec_dataset.gti,
        exposure=spec_dataset.exposure,
        edisp=spec_dataset.edisp,
        acceptance=1,
        acceptance_off=(off_solid_angle / on_solid_angle).to_value(""),
    )
