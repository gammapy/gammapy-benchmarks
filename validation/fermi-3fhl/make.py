import logging
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path
from time import time

import click
import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from matplotlib import cm
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d

from gammapy.catalog import SourceCatalog3FHL
from gammapy.data import EventList
from gammapy.datasets import Datasets, MapDataset
from gammapy.datasets.map import MapEvaluator
from gammapy.estimators import FluxPoints, FluxPointsEstimator
from gammapy.irf import PSFMap, EDispKernelMap
from gammapy.maps import Map, MapAxis, WcsGeom, WcsNDMap
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    FoVBackgroundModel,
    LogParabolaSpectralModel,
    Models,
    SkyModel,
    PowerLawSpectralModel,
    TemplateSpatialModel,
    PowerLawNormSpectralModel,
    create_fermi_isotropic_diffuse_model,
)
from gammapy.utils.scripts import make_path

log = logging.getLogger(__name__)
BASE_PATH = Path(__file__).parent


def iscompatible(x, y, dx, dy):
    x, y, dx, dy = np.array(x), np.array(y), np.array(dx), np.array(dy)
    return abs(y - x) < dx + dy


def relative_error(x, y):
    x, y = np.array(x), np.array(y)
    return (y - x) / x


class Validation_3FHL:
    """Run Gammapy 3FHL validation.

    Parameters
    ----------
    selection : {"long", "short", "debug"}
        What analyses to run
    savefig : bool
        Save figures? TODO: why not always save figures?
    """

    def __init__(self, selection="short", savefig=True):
        log.info("Executing __init__()")
        self.resdir = BASE_PATH / "results"
        self.savefig = savefig

        # event list
        self.events = EventList.read(
            "$GAMMAPY_DATA/fermi_3fhl/fermi_3fhl_events_selected.fits.gz"
        )

        # energies
        self.El_flux = [10.0, 20.0, 50.0, 150.0, 500.0, 2000.0] * u.GeV
        El_fit = 10 ** np.arange(1, 3.31, 0.1) * u.GeV
        self.energy_axis = MapAxis.from_edges(
            El_fit, name="energy", unit="GeV", interp="log"
        )

        # psf margin for mask
        psf = PSFMap.read(
            "$GAMMAPY_DATA/fermi_3fhl/fermi_3fhl_psf_gc.fits.gz", format="gtpsf"
        )
        psf_r99max = np.max(psf.containment_radius(fraction=0.99, energy_true=El_fit))
        self.psf_margin = np.ceil(psf_r99max.value * 10) / 10.0

        # iso norm=0.92 see paper appendix A
        self.model_iso = create_fermi_isotropic_diffuse_model(
            filename="data/iso_P8R2_SOURCE_V6_v06_extrapolated.txt",
            interp_kwargs={"fill_value": None},
        )
        self.model_iso.spectral_model.model2.norm.value = 0.92
        # regions selection
        file3fhl = "$GAMMAPY_DATA/catalogs/fermi/gll_psch_v13.fit.gz"
        self.FHL3 = SourceCatalog3FHL(file3fhl)
        hdulist = fits.open(make_path(file3fhl))
        self.ROIs = hdulist["ROIs"].data
        Scat = hdulist[1].data
        order = np.argsort(Scat.Signif_Avg)[::-1]
        ROIs_ord = Scat.ROI_num[order]

        if selection == "short":
            self.ROIs_sel = [430, 135, 118, 212, 277, 42, 272, 495]
            # Crab, Vela, high-lat, +some fast regions
        elif selection == "long":
            # get small regions with few sources among the most significant
            indexes = np.unique(ROIs_ord, return_index=True)[1]
            ROIs_ord = [ROIs_ord[index] for index in sorted(indexes)]
            self.ROIs_sel = [
                kr
                for kr in ROIs_ord
                if sum(Scat.ROI_num == kr) <= 4 and self.ROIs.RADIUS[kr] < 6
            ][:100]
        elif selection == "debug":
            self.ROIs_sel = [135]  # Vela region
        else:
            raise ValueError(f"Invalid selection: {selection!r}")

        # fit options
        self.fit_opts = {
            "backend": "minuit",
            "optimize_opts": {"tol": 10.0, "strategy": 2},
        }

        # calculate flux points only for sources significant above this threshold
        self.sig_cut = 8.0

        # diagnostics stored to produce plots and outputs
        self.diags = {
            "message": [],
            "stat": [],
            "params": {},
            "errel": {},
            "compatibility": {},
            "cat_fp_sel": [],
        }
        self.diags["errel"]["flux_points"] = []
        keys = [
            "PL_tags",
            "PL_index",
            "PL_amplitude",
            "LP_tags",
            "LP_alpha",
            "LP_beta",
            "LP_amplitude",
        ]
        for key in keys:
            self.diags["params"][key] = []

    def run_all(self, run_fit=True, get_diags=True, processes=4):
        log.info("Executing run_all()")
        if run_fit:
            self.parallel_regions(processes)

        orig_stdout = sys.stdout
        f = open(self.resdir / "results.md", "w")
        sys.stdout = f
        self.read_regions()
        if get_diags:
            self.save_diags()
        sys.stdout = orig_stdout
        f.close()

    def parallel_regions(self, processes):
        log.info("Executing parallel_regions()")
        if processes >1:
            with mp.Pool(processes=processes) as pool:
                args = [
                    (
                        kr,
                        self.ROIs.GLON[kr - 1],
                        self.ROIs.GLAT[kr - 1],
                        self.ROIs.RADIUS[kr - 1],
                    )
                    for kr in self.ROIs_sel
                ]
                pool.starmap(self.run_region, args)
        else:
            for kr in self.ROIs_sel:
                self.run_region(kr,
                        self.ROIs.GLON[kr - 1],
                        self.ROIs.GLAT[kr - 1],
                        self.ROIs.RADIUS[kr - 1],
                        )
    
    def run_region(self, kr, lon, lat, radius):
        #    TODO: for now we have to read/create the allsky maps each in each job
        #    because we can't pickle <functools._lru_cache_wrapper object
        #    send this back to init when fixed

        log.info(f"ROI {kr}: loading data")

        # exposure
        exposure_hpx = Map.read(
            "$GAMMAPY_DATA/fermi_3fhl/fermi_3fhl_exposure_cube_hpx.fits.gz"
        )
        exposure_hpx.unit = "cm2 s"

        # psf
        psf_map = PSFMap.read(
            "$GAMMAPY_DATA/fermi_3fhl/fermi_3fhl_psf_gc.fits.gz", format="gtpsf"
        )
        # reduce size of the PSF
        axis = psf_map.psf_map.geom.axes["rad"].center.to_value(u.deg)
        indmax = np.argmin(np.abs(self.psf_margin - axis))
        psf_map = psf_map.slice_by_idx(slices={"rad": slice(0, indmax)})

        # iem
        iem_filepath = BASE_PATH / "data" / "gll_iem_v06_extrapolated.fits"
        iem_fermi_extra = Map.read(iem_filepath)
        # norm=1.1, tilt=0.03 see paper appendix A
        model_iem = SkyModel(
            PowerLawNormSpectralModel(norm=1.1, tilt=0.03),
            TemplateSpatialModel(iem_fermi_extra, normalize=False),
            name="iem_extrapolated",
        )

        # ROI
        roi_time = time()
        ROI_pos = SkyCoord(lon, lat, frame="galactic", unit="deg")
        width = 2 * (radius + self.psf_margin)

        # Counts
        counts = Map.create(
            skydir=ROI_pos,
            width=width,
            proj="CAR",
            frame="galactic",
            binsz=1 / 8.0,
            axes=[self.energy_axis],
            dtype=float,
        )
        counts.fill_by_coord(
            {"skycoord": self.events.radec, "energy": self.events.energy}
        )

        axis = MapAxis.from_nodes(
            counts.geom.axes[0].center, name="energy_true", unit="GeV", interp="log"
        )
        wcs = counts.geom.wcs
        geom = WcsGeom(wcs=wcs, npix=counts.geom.npix, axes=[axis])
        coords = geom.get_coord()
        # expo
        data = exposure_hpx.interp_by_coord(coords)
        exposure = WcsNDMap(geom, data, unit=exposure_hpx.unit, dtype=float)

        # Energy Dispersion
        edisp = EDispKernelMap.from_diagonal_response(
            energy_axis_true=axis, energy_axis=self.energy_axis
        )

        # fit mask
        if coords["lon"].min() < 90 * u.deg and coords["lon"].max() > 270 * u.deg:
            coords["lon"][coords["lon"].value > 180] -= 360 * u.deg
        mask = (
            (coords["lon"] >= coords["lon"].min() + self.psf_margin * u.deg)
            & (coords["lon"] <= coords["lon"].max() - self.psf_margin * u.deg)
            & (coords["lat"] >= coords["lat"].min() + self.psf_margin * u.deg)
            & (coords["lat"] <= coords["lat"].max() - self.psf_margin * u.deg)
        )
        mask_fermi = WcsNDMap(counts.geom, mask)
        mask_safe_fermi = WcsNDMap(counts.geom, np.ones(mask.shape, dtype=bool))

        log.info(f"ROI {kr}: pre-computing diffuse")

        # IEM
        eval_iem = MapEvaluator(
            model=model_iem,
            exposure=exposure,
            psf=psf_map.get_psf_kernel(geom),
            edisp=edisp.get_edisp_kernel(),
        )
        bkg_iem = eval_iem.compute_npred()

        # ISO
        eval_iso = MapEvaluator(
            model=self.model_iso, exposure=exposure, edisp=edisp.get_edisp_kernel()
        )
        bkg_iso = eval_iso.compute_npred()

        # merge iem and iso, only one local normalization is fitted
        dataset_name = "3FHL_ROI_num" + str(kr)
        background_total = bkg_iem + bkg_iso

        # Dataset
        dataset = MapDataset(
            counts=counts,
            exposure=exposure,
            background=background_total,
            psf=psf_map,
            edisp=edisp,
            mask_fit=mask_fermi,
            mask_safe=mask_safe_fermi,
            name=dataset_name,
        )

        background_model = FoVBackgroundModel(dataset_name=dataset_name)
        background_model.parameters["norm"].min = 0.0

        # Sources model
        in_roi = self.FHL3.positions.galactic.contained_by(wcs)
        FHL3_roi = []
        for ks in range(len(self.FHL3.table)):
            if in_roi[ks] == True:
                model = self.FHL3[ks].sky_model()
                model.spatial_model.parameters.freeze_all()  # freeze spatial
                model.spectral_model.parameters["amplitude"].min = 0.0
                if isinstance(model.spectral_model, PowerLawSpectralModel):
                    model.spectral_model.parameters["index"].min = 0.1
                    model.spectral_model.parameters["index"].max = 10.0
                else:
                    model.spectral_model.parameters["alpha"].min = 0.1
                    model.spectral_model.parameters["alpha"].max = 10.0

                FHL3_roi.append(model)
        model_total = Models(FHL3_roi + [background_model])
        dataset.models = model_total

        cat_stat = dataset.stat_sum()
        datasets = Datasets([dataset])

        log.info(f"ROI {kr}: running fit")
        fit = Fit(**self.fit_opts)
        results = fit.run(datasets=datasets)
        print("ROI_num", str(kr), "\n", results)
        fit_stat = datasets.stat_sum()

        if results['optimize_result'].message != "Optimization failed.":
            filedata = Path(self.resdir) / f"3FHL_ROI_num{kr}_datasets.yaml"
            filemodel = Path(self.resdir) / f"3FHL_ROI_num{kr}_models.yaml"
            datasets.write(filedata, filemodel, overwrite=True)
            np.savez(
                self.resdir / f"3FHL_ROI_num{kr}_fit_infos.npz",
                message=results['optimize_result'].message,
                stat=[cat_stat, fit_stat],
            )

            exec_time = time() - roi_time
            print("ROI", kr, " time (s): ", exec_time)

            log.info(f"ROI {kr}: running flux points")
            for model in FHL3_roi:
                if (
                    self.FHL3[model.name].data["ROI_num"] == kr
                    and self.FHL3[model.name].data["Signif_Avg"] >= self.sig_cut
                ):
                    print(model.name)
                    flux_points = FluxPointsEstimator(
                        energy_edges=self.El_flux,
                        source=model.name,
                        n_sigma_ul=2,
                        selection_optional=["ul"],
                    ).run(datasets=datasets)
                    flux_points.meta["sqrt_ts_threshold"] = 1

                    filename = self.resdir / f"{model.name}_flux_points.fits"
                    flux_points.write(filename, overwrite=True)
    
            exec_time = time() - roi_time - exec_time
            print("ROI", kr, " Flux points time (s): ", exec_time)

    def read_regions(self):
        for kr in self.ROIs_sel:
            filedata = self.resdir / f"3FHL_ROI_num{kr}_datasets.yaml"
            filemodel = self.resdir / f"3FHL_ROI_num{kr}_models.yaml"
            try:
                dataset = list(Datasets.read(filedata, filemodel, lazy=False))[0]
            except (FileNotFoundError, IOError):
                continue

            infos = np.load(self.resdir / f"3FHL_ROI_num{kr}_fit_infos.npz")
            self.diags["message"].append(infos["message"])
            self.diags["stat"].append(infos["stat"])

            if self.savefig:
                self.plot_maps(dataset)

            for model in dataset.models:
                if (
                    isinstance(model, FoVBackgroundModel) is False
                    and self.FHL3[model.name].data["ROI_num"] == kr
                    and self.FHL3[model.name].data["Signif_Avg"] >= self.sig_cut
                ):
                    res_spec = model.spectral_model
                    cat_spec = self.FHL3[model.name].spectral_model()

                    res_fp = FluxPoints.read(
                        self.resdir / f"{model.name}_flux_points.fits",
                        reference_model = cat_spec
                    )
                    cat_fp = self.FHL3[model.name].flux_points
                    self.update_spec_diags(
                        dataset, model, cat_spec, res_spec, cat_fp, res_fp
                    )
                    if self.savefig:
                        self.plot_spec(kr, model, cat_spec, res_spec, cat_fp, res_fp)

    def plot_maps(self, dataset):
        plt.figure(figsize=(6, 6), dpi=150)
        dataset.plot_residuals_spatial(
            method="diff/sqrt(model)", smooth_kernel="gauss", smooth_radius=0.1 * u.deg,
        )
        plt.title("Residuals: (Nobs-Npred)/sqrt(Npred)")
        plt.savefig(self.resdir / f"resi_{dataset.name}.png", dpi=150)

        plt.figure(figsize=(6, 6), dpi=150)
        npred = dataset.npred().sum_over_axes()
        fig, ax, cb = npred.plot(cmap=cm.nipy_spectral, add_cbar=True, norm=LogNorm())
        plt.title("Npred")
        cl = ax.get_images()[0].get_clim()
        plt.savefig(self.resdir / f"npred_{dataset.name}.png", dpi=plt.gcf().dpi)

        plt.figure(figsize=(6, 6), dpi=150)
        nobs = dataset.counts.sum_over_axes()
        fig, ax, cb = nobs.plot(cmap=cm.nipy_spectral, add_cbar=True, norm=LogNorm())
        plt.title("Nobs")
        ax.get_images()[0].set_clim(cl)
        plt.savefig(self.resdir / f"counts_{dataset.name}.png", dpi=plt.gcf().dpi)
        plt.close("all")

    def plot_spec(self, kr, model, cat_spec, res_spec, cat_fp, res_fp):
        energy_bounds = [0.01, 2] * u.TeV
        plt.figure(figsize=(6, 6), dpi=150)
        ax = cat_spec.plot(
            energy_bounds=energy_bounds, energy_power=2, label="3FHL Catalogue", color="k"
        )
        cat_spec.plot_error(ax=ax, energy_bounds=energy_bounds, energy_power=2)
        res_spec.plot(
            ax=ax,
            energy_bounds=energy_bounds,
            energy_power=2,
            label="Gammapy fit",
            color="b",
        )
        res_spec.plot_error(
            ax=ax, energy_bounds=energy_bounds, energy_power=2, facecolor="c"
        )

        cat_fp.plot(ax=ax, energy_power=2, color="k")
        res_fp.plot(ax=ax, energy_power=2, color="b")
        plt.legend()
        tag = model.name.replace(" ", "_")
        plt.savefig(self.resdir / f"spec_{tag}_ROI_num{kr}.png", dpi=plt.gcf().dpi)
        plt.close("all")

    def update_spec_diags(self, dataset, model, cat_spec, res_spec, cat_fp, res_fp):
        ind = ~(res_fp.is_ul.data) & ~(cat_fp.is_ul.data)
        self.diags["errel"]["flux_points"] += list(
            relative_error(cat_fp.dnde.data[ind], res_fp.dnde.data[ind])
        )
        self.diags["cat_fp_sel"] += list(cat_fp.dnde.data[ind])

        if isinstance(res_spec, PowerLawSpectralModel):
            self.diags["params"]["PL_index"].append(
                [
                    cat_spec.parameters["index"].value,
                    res_spec.parameters["index"].value,
                    cat_spec.parameters["index"].error,
                    res_spec.parameters["index"].error,
                ]
            )
            self.diags["params"]["PL_amplitude"].append(
                [
                    cat_spec.parameters["amplitude"].value,
                    res_spec.parameters["amplitude"].value,
                    cat_spec.parameters["amplitude"].error,
                    res_spec.parameters["amplitude"].error,
                ]
            )
            self.diags["params"]["PL_tags"].append([dataset.name, model.name])

        if isinstance(res_spec, LogParabolaSpectralModel):
            self.diags["params"]["LP_alpha"].append(
                [
                    cat_spec.parameters["alpha"].value,
                    res_spec.parameters["alpha"].value,
                    cat_spec.parameters["alpha"].error,
                    res_spec.parameters["alpha"].error,
                ]
            )
            self.diags["params"]["LP_beta"].append(
                [
                    cat_spec.parameters["beta"].value,
                    res_spec.parameters["beta"].value,
                    cat_spec.parameters["beta"].error,
                    res_spec.parameters["beta"].error,
                ]
            )
            self.diags["params"]["LP_amplitude"].append(
                [
                    cat_spec.parameters["amplitude"].value,
                    res_spec.parameters["amplitude"].value,
                    cat_spec.parameters["amplitude"].error,
                    res_spec.parameters["amplitude"].error,
                ]
            )
            self.diags["params"]["LP_tags"].append([dataset.name, model.name])

    def save_diags(self):
        print("\n Vela region \n")
        filenames = [
            "/resi_3FHL_ROI_num135.png",
            "/npred_3FHL_ROI_num135.png",
            "/counts_3FHL_ROI_num135.png",
            "/spec_3FHL_J0833.1-4511e_ROI_num135.png",
            "/spec_3FHL_J0835.3-4510_ROI_num135.png",
            "/spec_3FHL_J0851.9-4620e_ROI_num135.png",
        ]
        for filename in filenames:
            print(f"\n ![]({self.resdir} {filename})")

        print("\n All following values are given in percent \n")

        # likelihood
        message = np.array(self.diags["message"])

        print("\n-", message, "-\n")
        msgs = [
            "Optimization terminated successfully.",
            "Optimization failed.",
            "Optimization failed. Estimated distance to minimum too large.",
        ]

        for msg in msgs:
            print(message)
            print(msg, 100 * sum(message == msg) / len(message), "  ")

        plt.figure(figsize=(6, 6), dpi=120)
        stat = np.array(self.diags["stat"])
        plt.plot(stat[:, 0], stat[:, 0] - stat[:, 1], "+ ")
        xl = plt.xlim()
        plt.ylim([-100, 100])
        plt.plot(xl, [0, 0], "-k")
        plt.xlabel("Cash Catalog")
        plt.ylabel("dCash Catalog - Fit")
        filename = self.resdir / "Cash_stat_corr.png"
        plt.savefig(filename, dpi=plt.gcf().dpi)
        print(f"\n ![]({filename})")

        # flux points
        key = "flux_points"
        errel = np.array(self.diags["errel"][key])
        print("\n" + key, "  ")
        print("Rel. err. <10%:", 100 * sum(abs(errel) < 0.1) / len(errel), "  ")
        print("Rel. err. <30%:", 100 * sum(abs(errel) < 0.3) / len(errel), "  ")
        print("Rel. err. mean:", 100 * np.nanmean(errel), "  ")

        plt.figure(figsize=(6, 6), dpi=150)
        plt.semilogx(self.diags["cat_fp_sel"], errel, " .")
        xl = plt.xlim()
        plt.plot(xl, [0, 0], "-k")
        plt.ylim([-1, 1])
        plt.xlabel(key + " Catalog")
        plt.ylabel("Relative error: (Gammapy-Catalog)/Catalog")
        if "amplitude" in key:
            ax = plt.gca()
            ax.set_xscale("log")
        filename = self.resdir / f"{key}_errel.png"
        plt.savefig(filename, dpi=plt.gcf().dpi)
        print(f"\n ![]({filename})")

        # Parameters diagnostics
        for key, diag in self.diags["params"].items():
            diag = np.array(diag)
            if "tags" not in key and len(diag) != 0:
                # fit vs. catalogue parameters comparisons
                print("\n" + key, "  ")
                ind = diag[:, 3] / diag[:, 1] < 0.1
                print("dx/x <10% : ", 100 * sum(ind) / len(diag[:, 1]), "  ")
                ind = diag[:, 3] / diag[:, 1] < 0.3
                print("dx/x <30% : ", 100 * sum(ind) / len(diag[:, 1]), "  ")

                plt.figure(figsize=(6, 6), dpi=150)
                plt.errorbar(
                    diag[ind, 0],
                    diag[ind, 1],
                    xerr=diag[ind, 2],
                    yerr=diag[ind, 3],
                    ls=" ",
                )
                if "amplitude" in key:
                    ax = plt.gca()
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                xl = plt.xlim()
                plt.plot(xl, xl, "-k")
                plt.xlabel(key + " Catalog")
                plt.ylabel(key + " Gammapy")
                filename = self.resdir / f"{key}_corr.png"
                plt.savefig(filename, dpi=plt.gcf().dpi)
                print(f"\n ![]({filename})")

                # relative error and compatibility
                comp = iscompatible(diag[:, 0], diag[:, 1], diag[:, 2], diag[:, 3])
                self.diags["compatibility"][key] = comp
                errel = relative_error(diag[:, 0], diag[:, 1])
                self.diags["errel"][key] = errel
                print("Rel. err. <10%:", 100 * sum(abs(errel) < 0.1) / len(errel), "  ")
                print("Rel. err. <30%:", 100 * sum(abs(errel) < 0.3) / len(errel), "  ")
                print("Rel. err. mean:", 100 * np.nanmean(errel), "  ")
                print("compatibility:", 100 * sum(comp) / len(comp), "  ")
                self.plot_errel(diag[:, 0], errel, key)

                key += "_error"
                errel = relative_error(diag[:, 2], diag[:, 3])
                self.diags["errel"][key] = errel
                self.plot_errel(diag[:, 2], errel, key)

                plt.close("all")

    def plot_errel(self, value, errel, key):
        plt.figure(figsize=(6, 6), dpi=150)
        plt.plot(value, errel, " .")
        xl = plt.xlim()
        plt.plot(xl, [0, 0], "-k")
        plt.ylim([-0.3, 0.3])
        plt.xlabel(key + " Catalog")
        plt.ylabel("Relative error: (Gammapy-Catalog)/Catalog")
        if "amplitude" in key:
            ax = plt.gca()
            ax.set_xscale("log")
        filename = self.resdir / f"{key}_errel.png"
        plt.savefig(filename, dpi=plt.gcf().dpi)
        print(f"\n ![]({filename})")


def get_data():
    """Get input data files for this validation."""
    items = """
    - path: data/gll_iem_v06.fits
      url: https://fermi.gsfc.nasa.gov/ssc/data/analysis/software/aux/gll_iem_v06.fits
    - path: data/iso_P8R2_SOURCE_V6_v06.txt
      url: https://raw.githubusercontent.com/gammapy/gammapy-extra/master/datasets/fermi_3fhl/iso_P8R2_SOURCE_V6_v06.txt
    """
    for item in yaml.safe_load(items):
        path = Path(item["path"])
        if path.exists():
            log.info(f"Skipping download. File exists: {path}")
        else:
            cmd = "wget {url} -O {path}".format_map(item)
            log.info(f"Execution: {cmd}")
            subprocess.call(cmd, shell=True)


def extrapolate_iso_model(logEc_extra):
    """Get ISO emission model with high-energy extrapolation."""
    infile = "data/iso_P8R2_SOURCE_V6_v06.txt"
    outfile = "data/iso_P8R2_SOURCE_V6_v06_extrapolated.txt"

    if Path(outfile).exists():
        return

    tmp = np.loadtxt(infile, delimiter=" ")
    Ecbd = tmp[:, 0]
    qiso = tmp[:, 1]
    finterp = interp1d(
        np.log10(Ecbd), np.log10(qiso), kind="linear", fill_value="extrapolate"
    )
    qiso_extra = 10 ** finterp(logEc_extra)

    log.info(f"Writing {outfile}")
    np.savetxt(outfile, np.c_[10 ** logEc_extra, qiso_extra], delimiter=" ")


def extrapolate_iem_model(logEc_extra):
    """Get IEM emission model with high-energy extrapolation."""
    infile = BASE_PATH / "data" / "gll_iem_v06.fits"
    outfile = BASE_PATH / "data" / "gll_iem_v06_extrapolated.fits"

    if Path(outfile).exists():
        return

    iem_fermi = Map.read(infile)
    Ec = iem_fermi.geom.axes[0].center.value
    finterp = interp1d(
        np.log10(Ec),
        np.log10(iem_fermi.data),
        axis=0,
        kind="linear",
        fill_value="extrapolate",
    )
    iem_extra = 10 ** finterp(logEc_extra)
    Ec_ax = MapAxis.from_nodes(
        10 ** logEc_extra, unit="MeV", name="energy_true", interp="log"
    )
    geom_3D = iem_fermi.geom.to_image().to_cube([Ec_ax])

    iem_fermi_extra = Map.from_geom(geom_3D, data=iem_extra.astype("float32"))
    iem_fermi_extra.unit = "cm-2 s-1 MeV-1 sr-1"

    log.info(f"Writing {outfile}")
    iem_fermi_extra.write(outfile, overwrite=True)


@click.command()
@click.option(
    "--selection", type=click.Choice(["debug", "short", "long"]), default="short"
)
@click.option("--processes", type=int, default=4)
@click.option("--fit", type=bool, default=True)
def cli(selection, processes, fit):
    logging.basicConfig(level=logging.INFO)
    get_data()

    # TODO: move this code to the extrapolation functions
    # Use MapAxis and Gammapy for energy axis functionality
    # Don't call `np.log` or `10 **` for this!

    El_extra = 10 ** np.arange(3.8, 6.51, 0.1)  # MeV
    logEc_extra = (np.log10(El_extra)[1:] + np.log10(El_extra)[:-1]) / 2.0
    extrapolate_iso_model(logEc_extra)
    extrapolate_iem_model(logEc_extra)

    validation = Validation_3FHL(selection=selection, savefig=True)
    validation.run_all(run_fit=fit, get_diags=True, processes=processes)

if __name__ == "__main__":
    cli()
