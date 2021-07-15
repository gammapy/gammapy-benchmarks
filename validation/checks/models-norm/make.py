# Licensed under a 3-clause BSD style license - see LICENSE.rst
from gammapy.modeling.models import SPATIAL_MODEL_REGISTRY
from gammapy.maps import WcsGeom
import numpy as np
import time
from astropy import units as u
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
start_time = time.time()


def norm_correction(self, geom=None):
    """Norm correction to be multiplied with spectral model amplitude"""
    if geom is None:
        if self.evaluation_radius == 0 * u.deg:
            radius = 0.1 * u.deg
        elif self.evaluation_radius is None:
            radius = 180 * u.deg
        else:
            radius = self.evaluation_radius
        oversampling_factor = 100.0
        evaluation_binsz = self.evaluation_radius / oversampling_factor
        geom = WcsGeom.create(
            skydir=self.position,
            width=(2 * radius),
            binsz=evaluation_binsz,
            frame=self.frame,
        )
    return np.sum(self.evaluate_geom(geom).to_value("sr-1") * geom.solid_angle()).value


# grid setup
nr = 40
ne = 11
values = dict(
    r_0=np.logspace(-1.6, 1.4, nr),
    sigma=np.logspace(-1.6, 1.4, nr),
    eta=np.linspace(0.05, 1, ne),
    e=np.linspace(0, 0.99, ne),
)
log_dr = np.log10(values["r_0"][1]) - np.log10(values["r_0"][0])
de = values["e"][1] - values["e"][0]

extent = [
    np.log10(values["r_0"][0]) - log_dr / 2,
    np.log10(values["r_0"][-1]) + log_dr / 2,
    values["e"][0] - de / 2,
    values["e"][-1] + de / 2,
]

# model tags and parameter names
tags = ["disk", "gauss-general", "gauss-general", "gauss", "shell2"]
par_x = ["r_0", "r_0", "r_0", "sigma", "r_0"]
par_y = ["e", "e", "e", "e", "eta"]
eta_val = [None, 0.1, 0.9, None, None]

# norm correction array
for k, tag in enumerate(tags):
    class_ = SPATIAL_MODEL_REGISTRY.get_cls(tag)
    class_.norm_correction = norm_correction
    m = class_()
    ngrid = np.zeros((nr, ne))
    if eta_val[k] is not None:
        m.eta.value = eta_val[k]
        tag += f"_eta{eta_val[k]}"
    for kr in range(nr):
        for ke in range(ne):
            m.parameters[par_x[k]].value = values[par_x[k]][kr]
            m.parameters[par_y[k]].value = values[par_y[k]][ke]
            ngrid[kr, ke] = m.norm_correction()
    plt.figure(figsize=(12, 4), dpi=100)
    plt.imshow(
        ngrid.T, origin="lower", extent=extent, cmap="coolwarm", interpolation=None
    )
    plt.title(tag)
    plt.colorbar()
    plt.clim([0.9, 1.1])
    plt.xlabel(f"log({par_x[k]})")
    plt.ylabel(f"{par_y[k]}")
    plt.tight_layout()
    plt.savefig(f"./norm_correction_{tag}.png", dpi=110)

exec_time = time.time() - start_time
print("Execution time in seconds: ", exec_time)
