import logging

log = logging.getLogger(__name__)

import numpy as np
import astropy.units as u
import yaml
from pathlib import Path
from astropy.coordinates import SkyCoord, Angle
from astropy.io import ascii
from regions import CircleSkyRegion
from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.maps import Map
from gammapy.modeling import Fit, Datasets
from gammapy.data import DataStore
from gammapy.modeling.models import PowerLawSpectralModel, create_crab_spectral_model
from gammapy.cube import SafeMaskMaker
from gammapy.spectrum import (
    SpectrumDatasetMaker,
    SpectrumDatasetOnOff,
    FluxPointsEstimator,
    FluxPointsDataset,
    ReflectedRegionsBackgroundMaker,
)

with open("targets.yaml", "r") as stream:
    targets = yaml.safe_load(stream)

# Run 1D analysis
for target in targets:
    TAG = target["tag"]
    NAME = target["name"]
    RA = target["ra"]
    DEC = target["dec"]
    ON_SIZE = target["on_size"]
    DECORRELATION_ENERGY = target["decorrelation_energy"]
    EXCLUSION_RA = target["exclusion_ra"]
    EXCLUSION_DEC = target["exclusion_dec"]
    EXCLUSION_SIZE = target["exclusion_size"]

    PATH_RESULTS = Path(TAG + "/results/")
    TARGET_POS = SkyCoord(RA, DEC, unit="deg", frame="icrs")
    E_RECO = np.logspace(-1, np.log10(70), 20) * u.TeV
    E_TRUE = np.logspace(np.log10(0.11), 2, 200) * u.TeV
    ON_RADIUS = Angle(ON_SIZE * u.deg)
    CONTAINMENT_CORR = True
    EMIN, EMAX = [0.7, 60] * u.TeV  # Used for flux points estimation
    N_FLUX_POINTS = 15

    # Observations selection
    data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
    mask = data_store.obs_table["TARGET_NAME"] == NAME
    obs_table = data_store.obs_table[mask]
    observations = data_store.get_observations(obs_table["OBS_ID"])

    # Exclusion mask
    regions = []
    for idx in range(len(EXCLUSION_RA)):
        regions.append(
            CircleSkyRegion(
                center=SkyCoord(
                    EXCLUSION_RA[idx], EXCLUSION_DEC[idx], unit="deg", frame="icrs"
                ),
                radius=EXCLUSION_SIZE[idx] * u.deg,
            )
        )
    exclusion_mask = Map.create(
        npix=(150, 150), binsz=0.05, skydir=TARGET_POS, proj="TAN", coordsys="CEL"
    )
    mask = exclusion_mask.geom.region_mask(regions, inside=False)
    exclusion_mask.data = mask

    # Reflected regions background estimation
    on_region = CircleSkyRegion(center=TARGET_POS, radius=ON_RADIUS)
    dataset_maker = SpectrumDatasetMaker(
        region=on_region,
        e_reco=E_RECO,
        e_true=E_TRUE,
        containment_correction=CONTAINMENT_CORR,
    )
    bkg_maker = ReflectedRegionsBackgroundMaker(exclusion_mask=exclusion_mask)
    safe_mask_masker = SafeMaskMaker(methods=["edisp-bias"], bias_percent=10)

    datasets = []

    for observation in observations:
        dataset = dataset_maker.run(observation, selection=["counts", "aeff", "edisp"])
        dataset_on_off = bkg_maker.run(dataset, observation)
        dataset_on_off = safe_mask_masker.run(dataset_on_off, observation)
        datasets.append(dataset_on_off)

    # Fit spectrum
    model = PowerLawSpectralModel(
        index=2,
        amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"),
        reference=DECORRELATION_ENERGY * u.TeV,
    )

    for dataset in datasets:
        dataset.model = model

    fit_joint = Fit(datasets)
    result_joint = fit_joint.run()

    # Store fit results
    fit_results_dict = {}
    parameters = model.parameters
    parameters.covariance = result_joint.parameters.covariance
    for parameter in parameters:
        value = parameter.value
        error = parameters.error(parameter)
        unit = parameter.unit
        name = parameter.name
        string = "{0:.2e} +- {1:.2e} {2}".format(value, error, unit)
        fit_results_dict.update({name: string})
    with open(str(PATH_RESULTS / "results-summary-fit-1d.yaml"), "w") as f:
        yaml.dump(fit_results_dict, f)

    # Flux points
    e_edges = (
        np.logspace(np.log10(EMIN.value), np.log10(EMAX.value), N_FLUX_POINTS + 1)
        * u.TeV
    )
    fpe = FluxPointsEstimator(datasets=datasets, e_edges=e_edges)
    flux_points = fpe.run()
    keys = ["e_ref", "e_min", "e_max", "dnde", "dnde_errp", "dnde_errn"]
    ascii.write(
        flux_points.table_formatted[keys], str(PATH_RESULTS / "flux-points-1d.dat")
    )

# Run 3D analysis
for target in targets:
    TAG = target["tag"]

    PATH_CONFIG = Path(TAG + "/config/")
    PATH_RESULTS = Path(TAG + "/results/")
    EMIN, EMAX = [0.7, 60] * u.TeV  # Used for flux points estimation
    N_FLUX_POINTS = 15

    filename = str(PATH_CONFIG / "config_3d.yaml")
    config = yaml.load(open(filename))
    config = AnalysisConfig(config)

    #  Observation selection
    analysis = Analysis(config)
    analysis.get_observations()

    # Data reduction
    analysis.get_datasets()

    # Set runwise energy threshold. See reference paper, section 5.1.1.
    for dataset in analysis.datasets:
        # E_thr_bias (= 10% edisp)
        geom = dataset.counts.geom
        e_reco = geom.get_axis_by_name("energy").edges
        E_thr_bias = dataset.edisp.get_bias_energy(0.1)

        # E_thr_bkg (= background peak energy)
        background_model = dataset.background_model
        bkg_spectrum = background_model.map.get_spectrum()
        peak = bkg_spectrum.data.max()
        idx = list(bkg_spectrum.data).index(peak)
        E_thr_bkg = bkg_spectrum.energy.center[idx]

        esafe = max(E_thr_bias, E_thr_bkg)
        dataset.mask_fit = geom.energy_mask(emin=esafe)

    # Model fitting
    model_config = str(PATH_CONFIG / "config_model.yaml")
    analysis.set_model(filename=model_config)

    for dataset in analysis.datasets:
        dataset.background_model.norm.frozen = False
        dataset.background_model.tilt.frozen = False

    result = analysis.run_fit()
    assert analysis.fit_result.success == True

    # Store fit results
    fit_results_dict = {}
    parameters = analysis.model.parameters
    parameters.covariance = analysis.fit_result.parameters.covariance[0:5, 0:5]
    for parameter in parameters:
        value = parameter.value
        error = parameters.error(parameter)
        unit = parameter.unit
        name = parameter.name
        string = "{0:.2e} +- {1:.2e} {2}".format(value, error, unit)
        fit_results_dict.update({name: string})
    with open(str(PATH_RESULTS / "results-summary-fit-3d.yaml"), "w") as f:
        yaml.dump(fit_results_dict, f)

    # Flux points
    # TODO: This is a workaround to reoptimize the bkg in each energy bin. Add this option to the Analysis class
    datasets = analysis.datasets.copy()
    for dataset in datasets:
        for par in dataset.parameters:
            if par is not dataset.background_model.norm:
                par.frozen = True

    e_edges = (
        np.logspace(np.log10(EMIN.value), np.log10(EMAX.value), N_FLUX_POINTS + 1)
        * u.TeV
    )
    fpe = FluxPointsEstimator(
        datasets=datasets, e_edges=e_edges, source=TAG, reoptimize=True
    )

    flux_points = fpe.run()
    keys = ["e_ref", "e_min", "e_max", "dnde", "dnde_errp", "dnde_errn"]
    ascii.write(
        flux_points.table_formatted[keys], str(PATH_RESULTS / "flux-points-3d.dat")
    )
