import logging
import yaml
from pathlib import Path
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.maps import MapAxis
from gammapy.modeling import Fit
from gammapy.data import DataStore
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.cube import SafeMaskMaker
from gammapy.spectrum import (
    SpectrumDatasetMaker,
    FluxPointsEstimator,
    ReflectedRegionsBackgroundMaker,
)

log = logging.getLogger(__name__)

with open("targets.yaml", "r") as stream:
    targets = yaml.safe_load(stream)

# If DEBUG is True, analyzes only 1 run, complutes only 1 flux point and does
# not re-optimize the bkg during flux points computation
DEBUG = True

E_RECO = MapAxis.from_bounds(
    0.1, 100, nbin=72, unit="TeV", name="energy", interp="log"
).edges
NBIN = 24 if DEBUG is False else 1
FLUXP_EDGES = MapAxis.from_bounds(
    0.1, 100, nbin=NBIN, unit="TeV", name="energy", interp="log"
).edges


def main(analyse1d=True, analyse3d=True, plot_fluxp=False):
    # TODO: add the rxj1713 validation
    sources = ["crab", "msh1552", "pks2155"]
    for source in sources:
        # read config for the target
        target_filter = filter(lambda _: _["tag"] == source, targets)
        target_dict = list(target_filter)[0]

        if analyse1d:
            run_analysis_1d(target_dict)
        if analyse3d:
            run_analysis_3d(target_dict)
        if plot_fluxp:
            plot_flux_points()


def write_fit_summary(parameters, outfile):
    """Store fit results with uncertainties"""
    fit_results_dict = {}
    for parameter in parameters:
        value = parameter.value
        error = parameters.error(parameter)
        unit = parameter.unit
        name = parameter.name
        string = "{0:.2e} +- {1:.2e} {2}".format(value, error, unit)
        fit_results_dict.update({name: string})
    with open(str(outfile), "w") as f:
        yaml.dump(fit_results_dict, f)


def plot_flux_points():
    raise NotImplementedError

def run_analysis_1d(target_dict):
    """Run spectral analysis for the selected target"""
    tag = target_dict["tag"]
    name = target_dict["name"]

    log.info(f"running 1d analysis, {tag}")
    path_res = Path(tag + "/results/")

    ra = target_dict["ra"]
    dec = target_dict["dec"]
    on_size = target_dict["on_size"]
    e_decorr = target_dict["e_decorr"]

    target_pos = SkyCoord(ra, dec, unit="deg", frame="icrs")
    on_radius = Angle(on_size * u.deg)
    containment_corr = True

    # Observations selection
    data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
    mask = data_store.obs_table["TARGET_NAME"] == name
    obs_table = data_store.obs_table[mask]
    observations = data_store.get_observations(obs_table["OBS_ID"])

    if DEBUG is True:
        observations = [observations[0]]

    # Reflected regions background estimation
    on_region = CircleSkyRegion(center=target_pos, radius=on_radius)
    dataset_maker = SpectrumDatasetMaker(
        region=on_region,
        e_reco=E_RECO,
        e_true=E_RECO,
        containment_correction=containment_corr,
    )
    bkg_maker = ReflectedRegionsBackgroundMaker()
    safe_mask_masker = SafeMaskMaker(methods=["edisp-bias"], bias_percent=10)

    datasets = []

    for observation in observations:
        dataset = dataset_maker.run(observation, selection=["counts", "aeff", "edisp"])
        dataset_on_off = bkg_maker.run(dataset, observation)
        dataset_on_off = safe_mask_masker.run(dataset_on_off, observation)
        datasets.append(dataset_on_off)

    # Fit spectrum
    model = PowerLawSpectralModel(
        index=2, amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=e_decorr * u.TeV
    )

    for dataset in datasets:
        dataset.model = model

    fit_joint = Fit(datasets)
    result_joint = fit_joint.run()

    parameters = model.parameters
    parameters.covariance = result_joint.parameters.covariance
    write_fit_summary(parameters, str(path_res / "results-summary-fit-1d.yaml"))

    # Flux points
    fpe = FluxPointsEstimator(datasets=datasets, e_edges=FLUXP_EDGES)
    flux_points = fpe.run()
    flux_points.table["is_ul"] = flux_points.table["ts"] < 4
    keys = ["e_ref", "e_min", "e_max", "dnde", "dnde_errp", "dnde_errn", "is_ul"]
    flux_points.table_formatted[keys].write(
        path_res / "flux-points-1d.ecsv", format="ascii.ecsv"
    )


def run_analysis_3d(target_dict):
    """Run 3D analysis for the selected target"""
    tag = target_dict["tag"]
    name = target_dict["name"]
    log.info(f"running 3d analysis, {tag}")

    path_res = Path(tag + "/results/")

    ra = target_dict["ra"]
    dec = target_dict["dec"]
    e_decorr = target_dict["e_decorr"]

    config_str = f"""
    general:
        logging:
            level: INFO
        outdir: .

    observations:
        datastore: $GAMMAPY_DATA/hess-dl3-dr1/
        filters:
            - filter_type: par_value
              value_param: {name}
              variable: TARGET_NAME

    datasets:
        dataset-type: MapDataset
        stack-datasets: true
        offset-max: 2.5 deg
        geom:
            skydir: [{ra}, {dec}]
            width: [5, 5]
            binsz: 0.02
            coordsys: CEL
            proj: TAN
            axes:
              - name: energy
                hi_bnd: 100
                lo_bnd: 0.1
                nbin: 24
                interp: log
                node_type: edges
                unit: TeV
        energy-axis-true:
            name: energy
            hi_bnd: 100
            lo_bnd: 0.1
            nbin: 72
            interp: log
            node_type: edges
            unit: TeV
    """
    print(config_str)
    config = AnalysisConfig(config_str)

    #  Observation selection
    analysis = Analysis(config)
    analysis.get_observations()

    if DEBUG is True:
        analysis.observations.list = [analysis.observations.list[0]]

    # Data reduction
    analysis.get_datasets()

    # Set runwise energy threshold. See reference paper, section 5.1.1.
    for dataset in analysis.datasets:
        # energy threshold given by the 10% edisp criterium
        e_thr_bias = dataset.edisp.get_bias_energy(0.1)

        # energy at which the background peaks
        background_model = dataset.background_model
        bkg_spectrum = background_model.map.get_spectrum()
        peak = bkg_spectrum.data.max()
        idx = list(bkg_spectrum.data).index(peak)
        e_thr_bkg = bkg_spectrum.energy.center[idx]

        esafe = max(e_thr_bias, e_thr_bkg)
        dataset.mask_fit = dataset.counts.geom.energy_mask(emin=esafe)

    # Model fitting
    spatial_model = target_dict["spatial_model"]
    model_config = f"""
    components:
        - name: {tag}
          type: SkyModel
          spatial:
            type: {spatial_model}
            frame: icrs
            parameters:
            - name: lon_0
              value: {ra}
              unit: deg
            - name: lat_0 
              value: {dec}    
              unit: deg
          spectral:
            type: PowerLawSpectralModel
            parameters:
            - name: amplitude      
              value: 1.0e-12
              unit: cm-2 s-1 TeV-1
            - name: index
              value: 2.0
              unit: ''
            - name: reference
              value: {e_decorr}
              unit: TeV
              frozen: true
    """
    model_npars = 5
    if spatial_model == "DiskSpatialModel":
        model_config = yaml.load(model_config)
        parameters = model_config["components"][0]["spatial"]["parameters"]
        parameters.append(
            {
                "name": "r_0",
                "value": 0.2,
                "unit": "deg",
                "frozen": False
            }
        )
        parameters.append(
            {
                "name": "e",
                "value": 0.8,
                "unit": "",
                "frozen": False
            }
        )
        parameters.append(
            {
                "name": "phi",
                "value": 150,
                "unit": "deg",
                "frozen": False
            }
        )
        parameters.append(
            {
                "name": "edge",
                "value": 0.01,
                "unit": "deg",
                "frozen": True
            }
        )
        model_npars += 4
    analysis.set_model(model=model_config)

    for dataset in analysis.datasets:
        dataset.background_model.norm.frozen = False

    analysis.run_fit()

    parameters = analysis.model.parameters
    parameters.covariance = analysis.fit_result.parameters.covariance[0:model_npars, 0:model_npars]
    write_fit_summary(parameters, str(path_res / "results-summary-fit-3d.yaml"))

    # Flux points
    # TODO: This is a workaround to re-optimize the bkg in each energy bin. Add has to be added to the Analysis class
    datasets = analysis.datasets.copy()
    for dataset in datasets:
        for par in dataset.parameters:
            if par is not dataset.background_model.norm:
                par.frozen = True

    reoptimize = True if DEBUG is False else False
    fpe = FluxPointsEstimator(
        datasets=datasets, e_edges=FLUXP_EDGES, source=tag, reoptimize=reoptimize
    )

    flux_points = fpe.run()
    flux_points.table["is_ul"] = flux_points.table["ts"] < 4
    keys = ["e_ref", "e_min", "e_max", "dnde", "dnde_errp", "dnde_errn"]
    flux_points.table_formatted[keys].write(
        path_res / "flux-points-3d.ecsv", format="ascii.ecsv"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
