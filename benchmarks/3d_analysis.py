from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.data import DataStore
from gammapy.irf import EnergyDispersion, make_mean_psf, make_mean_edisp
from gammapy.maps import WcsGeom, MapAxis, Map, WcsNDMap
from gammapy.cube import MapMaker, PSFKernel, MapDataset
from gammapy.cube.models import SkyModel, SkyDiffuseCube, BackgroundModel
from gammapy.spectrum.models import PowerLaw, ExponentialCutoffPowerLaw
from gammapy.spectrum import FluxPointsEstimator
from gammapy.image.models import SkyPointSource
from gammapy.utils.fitting import Fit


N_OBS = 100
OBS_ID = 110380


def run_benchmark():
    data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")
    obs_ids = OBS_ID * np.ones(N_OBS)

    observations = data_store.get_observations(obs_ids)

    energy_axis = MapAxis.from_edges(
        np.logspace(-1.0, 1.0, 10), unit="TeV", name="energy", interp="log"
    )
    geom = WcsGeom.create(
        skydir=(0, 0),
        binsz=0.02,
        width=(10, 8),
        coordsys="GAL",
        proj="CAR",
        axes=[energy_axis],
    )

    maker = MapMaker(geom, offset_max=4.0 * u.deg)
    maps = maker.run(observations)

    counts = maps["counts"].sum_over_axes()
    background = maps["background"].sum_over_axes()
    exposure = maps["exposure"].sum_over_axes()

    diffuse_gal = Map.read("$GAMMAPY_DATA/fermi-3fhl-gc/gll_iem_v06_gc.fits.gz")
    coord = maps["counts"].geom.get_coord()
    data = diffuse_gal.interp_by_coord(
        {
            "skycoord": coord.skycoord,
            "energy": coord["energy"]
            * maps["counts"].geom.get_axis_by_name("energy").unit,
        },
        interp=3,
    )
    diffuse_galactic = WcsNDMap(maps["counts"].geom, data)
    diffuse = diffuse_galactic.sum_over_axes()
    combination = diffuse * exposure
    combination.unit = ""

    src_pos = SkyCoord(0, 0, unit="deg", frame="galactic")
    table_psf = make_mean_psf(observations, src_pos)

    psf_kernel = PSFKernel.from_table_psf(table_psf, geom, max_radius="0.3 deg")

    energy = energy_axis.edges
    edisp = make_mean_edisp(
        observations, position=src_pos, e_true=energy, e_reco=energy
    )

    path = Path("analysis_3d")
    path.mkdir(exist_ok=True)

    maps["counts"].write(str(path / "counts.fits"), overwrite=True)
    maps["background"].write(str(path / "background.fits"), overwrite=True)
    maps["exposure"].write(str(path / "exposure.fits"), overwrite=True)

    psf_kernel.write(str(path / "psf.fits"), overwrite=True)
    edisp.write(str(path / "edisp.fits"), overwrite=True)

    maps = {
        "counts": Map.read(str(path / "counts.fits")),
        "background": Map.read(str(path / "background.fits")),
        "exposure": Map.read(str(path / "exposure.fits")),
    }

    psf_kernel = PSFKernel.read(str(path / "psf.fits"))
    edisp = EnergyDispersion.read(str(path / "edisp.fits"))

    coords = maps["counts"].geom.get_coord()
    mask = coords["energy"] > 0.3

    spatial_model = SkyPointSource(lon_0="0.01 deg", lat_0="0.01 deg")
    spectral_model = PowerLaw(
        index=2.2, amplitude="3e-12 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)

    background_model = BackgroundModel(maps["background"], norm=1.1, tilt=0.0)
    background_model.parameters["norm"].frozen = False
    background_model.parameters["tilt"].frozen = True

    dataset = MapDataset(
        model=model,
        counts=maps["counts"],
        exposure=maps["exposure"],
        background_model=background_model,
        mask_fit=mask,
        psf=psf_kernel,
        edisp=edisp,
    )

    fit = Fit(dataset)
    result = fit.run(optimize_opts={"print_level": 1})

    spec = model.spectral_model

    covariance = result.parameters.covariance
    spec.parameters.covariance = covariance[2:5, 2:5]

    diffuse_model = SkyDiffuseCube.read(
        "$GAMMAPY_DATA/fermi-3fhl-gc/gll_iem_v06_gc.fits.gz"
    )

    background_diffuse = BackgroundModel.from_skymodel(
        diffuse_model, exposure=maps["exposure"], psf=psf_kernel
    )

    background_irf = BackgroundModel(maps["background"], norm=1.0, tilt=0.0)
    background_total = background_irf + background_diffuse

    spatial_model = SkyPointSource(lon_0="-0.05 deg", lat_0="-0.05 deg")
    spectral_model = ExponentialCutoffPowerLaw(
        index=2 * u.Unit(""),
        amplitude=3e-12 * u.Unit("cm-2 s-1 TeV-1"),
        reference=1.0 * u.TeV,
        lambda_=0.1 / u.TeV,
    )

    model_ecpl = SkyModel(
        spatial_model=spatial_model, spectral_model=spectral_model, name="gc-source"
    )

    dataset_combined = MapDataset(
        model=model_ecpl,
        counts=maps["counts"],
        exposure=maps["exposure"],
        background_model=background_total,
        psf=psf_kernel,
        edisp=edisp,
    )

    fit_combined = Fit(dataset_combined)
    fit_combined.run()

    e_edges = [0.3, 1, 3, 10] * u.TeV
    fpe = FluxPointsEstimator(
        datasets=[dataset_combined], e_edges=e_edges, source="gc-source"
    )

    fpe.run()


if __name__ == "__main__":
    run_benchmark()
