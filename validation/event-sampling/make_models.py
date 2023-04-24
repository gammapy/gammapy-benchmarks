import click
import logging
import multiprocessing
import warnings

from itertools import repeat
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.table import Table
from astropy.time import Time
import numpy as np
from regions import CircleSkyRegion

from gammapy.maps import Map
from gammapy.modeling.models import (
    Model,
    Models,
    SkyModel,
    ConstantSpectralModel,
    PowerLawSpectralModel,
    PointSpatialModel,
    DiskSpatialModel,
    GaussianSpatialModel,
    TemplateSpatialModel,
    ExpCutoffPowerLawSpectralModel,
    LogParabolaSpectralModel,
    PowerLaw2SpectralModel,
    ExpCutoffPowerLaw3FGLSpectralModel,
    SuperExpCutoffPowerLaw4FGLSpectralModel,
    PowerLawNormSpectralModel,
    FoVBackgroundModel,
    TemplateSpectralModel,
    ExpDecayTemporalModel,
    GaussianTemporalModel,
    LightCurveTemplateTemporalModel,
)

log = logging.getLogger(__name__)

# path config
BASE_PATH = Path(__file__).parent

AVAILABLE_MODELS = [
    "point-pwl",
    "point-ecpl",
    "point-log-parabola",
    "point-pwl2",
    "point-ecpl-3fgl",
    "point-ecpl-4fgl",
    "point-template",
    "diffuse-cube",
    "disk-pwl",
    "gauss-pwl",
    "point-pwl-expdecay",
    "point-pwl-gausstemp",
    "point-pwl-lightemplate",
    "point-enedip_template",
]


@click.group()
@click.option(
    "--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING"])
)
@click.option("--show-warnings", is_flag=True, help="Show warnings?")
def cli(log_level, show_warnings):
    logging.basicConfig(level=log_level)
    if not show_warnings:
        warnings.simplefilter("ignore")


@cli.command("all", help="Build all models")
def run():
    for model in AVAILABLE_MODELS:
        make_model(model)
    


@cli.command("single_model", help="Build the given model")
@click.argument("model", type=click.Choice(list(AVAILABLE_MODELS) + ["all-models"]))
def run(model):
    make_model(model)


def make_model(model):
    log.info(f"Making model {model}")
    bkg_model = FoVBackgroundModel(dataset_name="my-dataset")

    spatial_model = PointSpatialModel(lon_0="0.0 deg",
                                      lat_0="0.0 deg",
                                      frame="galactic")
    temporal_model = None

    if model=="point-pwl":
        spectral_model = PowerLawSpectralModel(
                            index=2.0,
                            amplitude="1e-12 TeV-1 cm-2 s-1",
                            reference="1 TeV"
                            )

    if model=="point-ecpl":
        spectral_model = ExpCutoffPowerLawSpectralModel(
                            index=2.0,
                            amplitude="1e-12 TeV-1 cm-2 s-1",
                            lambda_ = "0.05 TeV-1",
                            reference="7 TeV"
                            )

    if model=="point-log-parabola":
        spectral_model = LogParabolaSpectralModel(
                            index=2.0,
                            amplitude="1e-12 TeV-1 cm-2 s-1",
                            alpha = 2.0,
                            beta = 0.12,
                            reference="1 TeV"
                            )

    if model=="point-pwl2":
        spectral_model = PowerLaw2SpectralModel(
                            index=2.0,
                            amplitude="1e-12 cm-2 s-1",
                            emin = 1.0 * u.TeV,
                            emax = 10.0 * u.TeV,
                            reference="1 TeV"
                            )

    if model=="point-ecpl-3fgl":
        spectral_model = ExpCutoffPowerLaw3FGLSpectralModel(
                            index=2.0,
                            amplitude="1e-12 TeV-1 cm-2 s-1",
                            ecut = 20.0 * u.TeV,
                            reference="1 TeV"
                            )

    if model=="point-ecpl-4fgl":
        spectral_model = SuperExpCutoffPowerLaw4FGLSpectralModel(
                            index_1=1.4,
                            index_2=2.0,
                            amplitude="1e-12 TeV-1 cm-2 s-1",
                            expfactor = 0.5,
                            reference="1 TeV"
                            )

    if model=="point-template":
        energy = np.array([0.10000000000000002, 0.12589254117941673, 0.15848931924611137, 0.199526231496888, 0.25118864315095807, 0.316227766016838, 0.39810717055349737, 0.5011872336272724, 0.6309573444801934, 0.7943282347242816, 1.0, 1.258925411794167, 1.584893192461114, 1.9952623149688802, 2.5118864315095806, 3.1622776601683795, 3.9810717055349727, 5.011872336272723, 6.309573444801932, 7.943282347242814, 9.999999999999998, 12.58925411794167, 15.84893192461113, 19.952623149688787, 25.11886431509581, 31.622776601683803, 39.810717055349734, 50.11872336272724, 63.095734448019364, 79.43282347242813, 100.00000000000004]) * u.TeV
        values = np.array([6.309573444801931e-11, 4.168693834703354e-11, 2.7542287033381657e-11, 1.819700858609983e-11, 1.2022644346174126e-11, 7.943282347242812e-12, 5.248074602497723e-12, 3.4673685045253156e-12, 2.2908676527677723e-12, 1.5135612484362078e-12, 1.0e-12, 6.606934480075961e-13, 4.3651583224016565e-13, 2.8840315031266045e-13, 1.9054607179632465e-13, 1.258925411794167e-13, 8.317637711026709e-14, 5.495408738576245e-14, 3.630780547701014e-14, 2.398832919019491e-14, 1.5848931924611138e-14, 1.0471285480508999e-14, 6.918309709189368e-15, 4.570881896148754e-15, 3.019951720402014e-15, 1.995262314968878e-15, 1.3182567385564065e-15, 8.709635899560802e-16, 5.754399373371562e-16, 3.801893963205613e-16, 2.511886431509578e-16]) * 1 / (u.cm**2 * u.s * u.TeV)

        spectral_model = TemplateSpectralModel(energy=energy,
                                               values=values)

    if model=="diffuse-cube":
        spectral_model = PowerLawNormSpectralModel(norm=1.0,
                                                   tilt=0.0,
                                                   reference=1 * u.TeV)
                       
        filename = "$GAMMAPY_DATA/fermi-3fhl-gc/gll_iem_v06_gc.fits.gz"
        m = Map.read(filename)
#        m = m.copy(unit="sr^-1")
        spatial_model = TemplateSpatialModel(m, filename=filename, normalize=False)

    if model=="disk-pwl":
        spectral_model = PowerLawSpectralModel(
                            index=2.0,
                            amplitude="1e-12 TeV-1 cm-2 s-1",
                            reference="1 TeV"
                            )

        spatial_model = DiskSpatialModel(lon_0 = 0.0 * u.deg,
                                         lat_0 = 0.0 * u.deg,
                                         r_0 = 0.3 * u.deg,
                                         frame="galactic")

    if model=="gauss-pwl":
        spectral_model = PowerLawSpectralModel(
                            index=2.0,
                            amplitude="1e-12 TeV-1 cm-2 s-1",
                            reference="1 TeV"
                            )

        spatial_model = GaussianSpatialModel(lon_0 = 0.0 * u.deg,
                                             lat_0 = 0.0 * u.deg,
                                             sigma = 0.3 * u.deg,
                                             frame="galactic")

    if model=="point-pwl-expdecay":
        spectral_model = PowerLawSpectralModel(
                            index=2.0,
                            amplitude="5e-11 TeV-1 cm-2 s-1",
                            reference="1 TeV"
                            )
                            
        t0 = 0.2 * u.hr
        t_ref = Time(51544.00074287037, format="mjd", scale="tt").mjd * u.d
        temporal_model = ExpDecayTemporalModel(t_ref=t_ref, t0=t0)
        temporal_model.t_ref.frozen=True

    if model=="point-pwl-gausstemp":
        spectral_model = PowerLawSpectralModel(
                            index=2.0,
                            amplitude="5e-11 TeV-1 cm-2 s-1",
                            reference="1 TeV"
                            )
                            
        sigma = 0.1 * u.hr
        t_ref = Time(51544.03074287037, format="mjd", scale="tt").mjd * u.d
        temporal_model = GaussianTemporalModel(t_ref=t_ref, sigma=sigma)
        temporal_model.t_ref.frozen=True

    if model=="point-pwl-lightemplate":
        livetime = 1 * u.hr
        sigma = 0.1 * u.h
        t_ref = Time(51544.00074287037, format="mjd", scale="tt")
        times = t_ref.mjd * u.d + livetime * np.linspace(0, 1, 1000)

        flare_model = GaussianTemporalModel(t_ref=times[500], sigma=sigma)

        # create the astropy table
        from gammapy.utils.time import time_ref_to_dict
        meta = time_ref_to_dict(t_ref)
        lc = Table()
        lc.meta = meta
        lc.meta["TIMEUNIT"] = 's'

        t = Time(times, format="mjd", scale="tt")
        lc["TIME"] = (times - times[0]).to("s")
        lc["NORM"] = flare_model(t)

        lc.write("./models/lc.fits", overwrite=True)
    
        spectral_model = PowerLawSpectralModel(
                            index=2.0,
                            amplitude="5e-11 TeV-1 cm-2 s-1",
                            reference="1 TeV"
                            )
                            
        temporal_model = LightCurveTemplateTemporalModel.read("./models/lc.fits")

    if model=="point-enedip_template":
        spectral_model = ConstantSpectralModel(const="1 cm-2 s-1 TeV-1")

        filename = "$GAMMAPY_DATA/gravitational_waves/GW_example_DC_map_file.fits.gz"
        temporal_model = LightCurveTemplateTemporalModel.read(filename, format="map")
        temporal_model.t_ref.value = 51544.00074287037

    sky_model = SkyModel(spectral_model = spectral_model,
                         spatial_model = spatial_model,
                         temporal_model = temporal_model,
                         name=model)
        
    save_model(sky_model, bkg_model, f"{model}.yaml")


def save_model(sky_model, bkg_model, filename):
    models = Models([sky_model, bkg_model])
    file_model = f"./models/{filename}"
    models.write(file_model, overwrite=True)


if __name__ == "__main__":
    cli()
