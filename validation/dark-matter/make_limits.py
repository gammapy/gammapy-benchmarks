# Licensed under a 3-clause BSD style license - see LICENSE
"""Validation of DM continuum limits (annihilation and decay) against the CTAO dSph paper.

Reference: CTAO Consortium (2025), MNRAS 544, 2946, arXiv:2508.19120, and its
supplementary material https://zenodo.org/records/17079907 (see data/README.md).

Median expected limit for single dSphs, 100 h, Einasto, without the J/D factor as
nuisance ("wosj" in the paper), compared with the median of the paper:
upper limit on <sigma v> for annihilation, lower limit on the lifetime for decay.
By default the median is computed on an Asimov dataset; with --n-realizations it
is the median over Poisson realizations.

python make_limits.py --mode annihilation
python make_limits.py --mode decay
"""

import logging
import time
import warnings
from pathlib import Path

import click
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from gammapy.astro.darkmatter import DarkMatterSpectralModel
from gammapy.data import FixedPointingInfo, Observation
from gammapy.datasets import MapDataset
from gammapy.estimators import ParameterEstimator
from gammapy.irf import load_irf_dict_from_file
from gammapy.makers import MapDatasetMaker, SafeMaskMaker
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling.models import (
    FoVBackgroundModel,
    Models,
    SkyModel,
    TemplateSpatialModel,
)

log = logging.getLogger(__name__)

VALIDATION_DIR = Path(__file__).parent
PATH_DATA = VALIDATION_DIR / "data"
PATH_RESULTS = VALIDATION_DIR / "results"

TARGETS = ["DraI", "Scl"]
TARGETS_FILE = PATH_DATA / "inputs_combined_analysis.csv"

MASSES = [1, 10, 50] * u.TeV
CHANNELS = ["b", "tau"]  # soft and hard continuum
PROFILE = "Einasto"
LIVETIME = 100 * u.h
SPECTRA = "pppc4"
DELTA_TS = 4  # Delta TS defining the limit, as in the paper pipeline
TOLERANCE = 0.15  # check: Gammapy / paper median in [1 / (1 + tol), 1 + tol]

ENERGY_AXIS = MapAxis.from_energy_bounds(0.03, 100, nbin=30, unit="TeV")
ENERGY_AXIS_TRUE = MapAxis.from_energy_bounds(
    0.02, 200, nbin=40, unit="TeV", name="energy_true"
)

MODES = {
    "annihilation": {
        "factor_unit": "GeV2 cm-5",
        "limit_unit": "cm3 s-1",
        "map_file": (
            "clumpy/Jfactors/2Dmaps/{profile}/"
            "annihil_{target}2D_FOVdiameter10.0deg_nside1024-JFACTOR-Jtot-image.fits"
        ),
        "reference_file": (
            "limits_wosj/annihilation/single/"
            "{target}_{profile}_annihilation_t100h_wosj_quantiles.csv"
        ),
    },
    "decay": {
        "factor_unit": "GeV cm-2",
        "limit_unit": "s",
        "map_file": (
            "clumpy/Dfactors/2Dmaps/{profile}/"
            "decay_{target}2D_FOVdiameter10.0deg_nside1024-DFACTOR-Dtot-image.fits"
        ),
        "reference_file": (
            "limits_wosj/decay/single/{target}_{profile}_decay_t100h_wosj_quantiles.csv"
        ),
    },
}


def read_targets(targets, mode):
    """IRF, integration radius (tidal radius) and J or D factor of each target.

    As in the paper pipeline, the J/D factor is integrated up to the tidal
    radius, which is also the maximum offset of the safe mask.
    """
    df = pd.read_csv(TARGETS_FILE).set_index("src")
    missing = set(targets) - set(df.index)
    if missing:
        raise click.ClickException(f"{sorted(missing)} not in {TARGETS_FILE.name}")
    column = "jfactor" if mode == "annihilation" else "dfactor"
    unit = u.Unit(MODES[mode]["factor_unit"])
    return {
        t: {
            "irf": df.loc[t, "irf"],
            "alpha_int": float(df.loc[t, "r_tidal"]),
            "factor": float(df.loc[t, column]) * unit,
        }
        for t in targets
    }


def read_spatial_model(mode, target):
    """Normalized spatial template from the CLUMPY 2D map and target position."""
    filename = PATH_DATA / MODES[mode]["map_file"].format(profile=PROFILE, target=target)
    header = fits.getheader(filename)
    position = SkyCoord(header["PSI_0"], header["THETA_0"], unit="deg", frame="galactic")
    return TemplateSpatialModel.read(filename, normalize=True), position


def read_paper_median(mode, target, channel):
    """Paper median at MASSES (log-log interpolation in mass) and the paper J/D factor."""
    filename = PATH_DATA / MODES[mode]["reference_file"].format(
        target=target, profile=PROFILE.lower()
    )
    df = pd.read_csv(filename)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]  # row index written by pandas

    if "profile" in df.columns:
        df = df[df["profile"].astype(str).str.lower() == PROFILE.lower()]

    ch = df["ch"].astype(str).str.lower().str.strip()
    df = df[ch.isin([channel.lower(), 2 * channel.lower()])]

    quantile = pd.to_numeric(df["quantile"], errors="coerce")
    if quantile.notna().all():
        is_median = np.isclose(quantile / (100 if quantile.max() > 1 else 1), 0.5)
    else:
        is_median = df["quantile"].astype(str).str.lower().str.strip().isin(["median", "50%"])
    df = df[is_median]

    value_columns = [c for c in df.columns
                    if c not in ["name", "jfactor", "profile", "ch", "mass", "quantile"]]
    
    median = df.groupby("mass")[value_columns[0]].median()

    mass = (median.index.values * u.GeV).to_value("TeV")
    log.info(
        f"{filename.name} ({channel}): {len(mass)} masses, {mass.min():g}-{mass.max():g} TeV, "
        f"column '{value_columns[0]}'"
    )
    if MASSES.to_value("TeV").min() < mass.min() or MASSES.to_value("TeV").max() > mass.max():
        log.warning(f"{MASSES} outside the mass range of {filename.name}")

    log_median = np.interp(
        np.log10(MASSES.to_value("TeV")), np.log10(mass), np.log10(median.values)
    )
    return 10**log_median

def build_dataset(target, irf, alpha_int, position):
    irf_file = PATH_DATA / "IRFs/cta/prod5-v0.1/bcf" / irf / "irf_file.fits.gz"
    obs = Observation.create(
        pointing=FixedPointingInfo(fixed_icrs=position.icrs),
        livetime=LIVETIME,
        irfs=load_irf_dict_from_file(irf_file),
    )
    geom = WcsGeom.create(
        skydir=position, binsz=0.02*u.deg, width=2 * u.deg, frame="galactic", axes=[ENERGY_AXIS]
    )
    empty = MapDataset.create(geom, energy_axis_true=ENERGY_AXIS_TRUE, name=target)
    maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
    safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=alpha_int * u.deg)
    return safe_mask.run(maker.run(empty, obs), obs)


def set_models(mode, dataset, mass, channel, factor, spatial_model, unit=1.0):
    """DM + background models."""
    spectral = DarkMatterSpectralModel(
        mDM=mass,
        channel=channel,
        factor=factor * unit,
        scale=0,
        source=SPECTRA,
        annihilation=(mode == "annihilation"),
    )
    spectral.scale.min = 0

    spectral.scale.scale_method = None
    dm = SkyModel(spatial_model=spatial_model.copy(), spectral_model=spectral, name="dm")

    bkg = FoVBackgroundModel(dataset_name=dataset.name)
    bkg.spectral_model.tilt.frozen = False
    dataset.models = Models([dm, bkg])
    return dataset


def scale_upper_limit(dataset):
    scale = dataset.models["dm"].spectral_model.scale
    scale.value, scale.error = 1.0, 0.1
    estimator = ParameterEstimator(
        n_sigma_ul=np.sqrt(DELTA_TS), null_value=0, selection_optional=["ul"]
    )
    return float(np.squeeze(estimator.run([dataset], parameter=scale)["scale_ul"]))


def to_physical(mode, scale_ul):
    """<sigma v> upper limit (annihilation) or lifetime lower limit (decay)."""
    if mode == "annihilation":
        limit = scale_ul * DarkMatterSpectralModel.THERMAL_RELIC_CROSS_SECTION
    else:  # scale = age of the Universe / lifetime
        limit = DarkMatterSpectralModel.LIFETIME_AGE_OF_UNIVERSE / scale_ul
    return limit.to_value(MODES[mode]["limit_unit"])


def median_limit(mode, dataset, mass, channel, factor, spatial_model, n_realizations):
    """Median limit on <sigma v> (annihilation) or the lifetime (decay)."""
    model_args = (mass, channel, factor, spatial_model)

    ds = set_models(mode, dataset.copy(name=dataset.name), *model_args)
    ds.models["dm"].spectral_model.scale.value = 1
    signal = ds.npred_signal().data[ds.mask].sum()
    unit = 2 * np.sqrt(ds.npred_background().data[ds.mask].sum()) / signal
    limits = []

    for _ in range(3):
        asimov = set_models(mode, dataset.copy(name=dataset.name), *model_args, unit)
        asimov.counts = asimov.npred()
        scale_ul = scale_upper_limit(asimov)
        if 0.3 < scale_ul < 3:
            break
        unit *= scale_ul
    limits = [scale_ul * unit]

    if n_realizations > 0:
        limits = []
        for seed in range(n_realizations):
            toy = set_models(mode, dataset.copy(name=dataset.name), *model_args, unit)
            toy.fake(random_state=seed)
            limits.append(scale_upper_limit(toy) * unit)

    return to_physical(mode, np.nanmedian(limits))


@click.command()
@click.option("--mode", required=True, type=click.Choice(list(MODES)))
@click.option("--targets", default=",".join(TARGETS), help="Comma-separated subset")
@click.option("--n-realizations", default=0, help="0: Asimov dataset")
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING"]))
def main(mode, targets, n_realizations, log_level):
    logging.basicConfig(level=log_level)
    warnings.simplefilter("ignore")
    start = time.time()
    rows = []

    for target, cfg in read_targets(targets.split(","), mode).items():
        factor = cfg["factor"]
        spatial_model, position = read_spatial_model(mode, target)
        dataset = build_dataset(target, cfg["irf"], cfg["alpha_int"], position)
        log.info(f"{target} ({cfg['irf']}): factor({cfg['alpha_int']} deg) = {factor:.3e}")

        for channel in CHANNELS:
            paper = read_paper_median(mode, target, channel)

            for mass, paper_median in zip(MASSES, paper):
                gammapy_median = median_limit(mode, dataset, mass, channel, factor,
                                              spatial_model, n_realizations)
                ratio = gammapy_median / paper_median
                passed = bool(abs(np.log(ratio)) < np.log(1 + TOLERANCE))
                rows.append({
                    "target": target, "channel": channel,
                    "mass": mass.to_value("TeV"), "gammapy_median": gammapy_median,
                    "paper_median": paper_median, "ratio": ratio, "passed": passed,
                })
                log.info(
                    f"{'PASS' if passed else 'FAIL'} {target:6s} {channel:4s} {mass:5.0f}: "
                    f"gammapy {gammapy_median:.3e}  paper {paper_median:.3e}  ratio {ratio:.2f}"
                )

    table = Table(rows=rows)
    table["mass"].unit = "TeV"
    table["gammapy_median"].unit = table["paper_median"].unit = MODES[mode]["limit_unit"]
    table.meta.update({
        "mode": mode, "profile": PROFILE, "livetime": str(LIVETIME), "spectra": SPECTRA,
        "delta_ts": DELTA_TS, "n_realizations": n_realizations,
        "tolerance": TOLERANCE,
    })
    PATH_RESULTS.mkdir(exist_ok=True)
    suffix = "" if targets == ",".join(TARGETS) else "_" + targets.replace(",", "_")
    filename = PATH_RESULTS / f"limits_{mode}{suffix}.ecsv"
    table.write(filename, overwrite=True)
    log.info(f"{np.sum(table['passed'])}/{len(table)} passed, wrote {filename.name} "
             f"in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()