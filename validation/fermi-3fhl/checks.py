#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'nb_black')
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from gammapy.data import EventList
from gammapy.irf import EnergyDependentTablePSF, EnergyDispersion
from gammapy.maps import Map, MapAxis, WcsNDMap, WcsGeom
from gammapy.modeling import Fit, Datasets
from gammapy.modeling.models import (
    SkyDiffuseCube,
    SkyModels,
    BackgroundModel,
    LogParabolaSpectralModel,
    PowerLawSpectralModel,
    create_fermi_isotropic_diffuse_model,
)
from gammapy.spectrum import FluxPoints, FluxPointsEstimator
from gammapy.cube import MapDataset, PSFKernel, MapEvaluator
from gammapy.catalog import SourceCatalog3FHL
from pathlib import Path
from time import time
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

def iscompatible(x, y, dx, dy):
    x, y, dx, dy = np.array(x), np.array(y), np.array(dx), np.array(dy)
    return abs(y - x) < dx + dy

def relative_error(x, y):
    x, y = np.array(x), np.array(y)
    return (y - x) / x

start_time = time()
plt.ioff()


# In[ ]:


# Settings

datadir = "/Users/qremy/Work/Data/gammapy-tutorials/datasets"  # "$GAMMAPY_DATA"
resdir = "/Users/qremy/Work/GitHub/gammapy-fermi-lat-data/3FHL/checks"  # "$GAMMAPY_FERMI_LAT_DATA/3FHL/checks/"

events = EventList.read(datadir + "/fermi_3fhl/fermi_3fhl_events_selected.fits.gz")
exposure_hpx = Map.read(datadir + "/fermi_3fhl/fermi_3fhl_exposure_cube_hpx.fits.gz")
exposure_hpx.unit = "cm2 s"
psf = EnergyDependentTablePSF.read(datadir + "/fermi_3fhl/fermi_3fhl_psf_gc.fits.gz")
iem_fermi = Map.read(datadir + "/catalogs/fermi/gll_iem_v06.fits.gz")
iem_fermi.unit = "cm-2 s-1 MeV-1 sr-1"
fileiso = datadir + "/fermi_3fhl/iso_P8R2_SOURCE_V6_v06.txt"
model_iso = create_fermi_isotropic_diffuse_model(
    filename=fileiso, norm=0.92, interp_kwargs={"fill_value": None}
)  # norm=0.92 see paper appendix A

file3fhl = Path(datadir + "/catalogs/fermi/gll_psch_v13.fit.gz")
FHL3 = SourceCatalog3FHL(file3fhl)
hdulist = fits.open(file3fhl)
ROIs = hdulist["ROIs"].data
nROIs = len(ROIs)
Scat = hdulist[1].data
order = np.argsort(Scat.Signif_Avg)[::-1]
ROIs_ord = Scat.ROI_num[order]
nmin = 0  # 10  # 90
nmax = 90  # 100  # 250
ROIs_sel = np.unique(ROIs_ord[nmin:nmax])
# ROIs_sel = [430, 135, 80, 118] #Crab, Vela, GC, high-lat

dlb = 0.05
El_fit = 10 ** np.arange(1, 3.31, 0.1)
El_flux = [10.0, 20.0, 50.0, 150.0, 500.0, 2000.0]  # see hdulist["EnergyBounds"].data
energy_axis = MapAxis.from_edges(El_fit, name="energy", unit="GeV", interp="log")
psf_r99max = psf.containment_radius(10 * u.GeV, fraction=0.99)
psf_margin = np.ceil(psf_r99max.value[0] * 10) / 10.0

run_fit = False
optimize_opts = {
    "backend": "minuit",
    "migrad_opts": {"precision": 1e-8},
}

sig_cut = 8.0
# calculate flux points only for sources significant above this threshold

PL_tags = []
PL_index = []
PL_amplitude = []
PL_BKG_norm = []

LP_tags = []
LP_amplitude = []
LP_alpha = []
LP_beta = []
LP_BKG_norm = []

stat = []
message = []
compatibility = {}
errel = {"flux_points": []}
cat_fp_sel = []
cat_dfp_sel = []


# In[ ]:


# Run fit

for kr in ROIs_sel:  # nROIs
    if run_fit:
        roi_time = time()
        print("ROI " + str(kr))
        ROI_pos = SkyCoord(
            ROIs.GLON[kr - 1], ROIs.GLAT[kr - 1], frame="galactic", unit="deg"
        )
        width = 2 * (ROIs.RADIUS[kr - 1] + psf_margin)
        # Counts
        counts = Map.create(
            skydir=ROI_pos,
            width=width,
            proj="CAR",
            coordsys="GAL",
            binsz=dlb,
            axes=[energy_axis],
            dtype=float,
        )
        counts.fill_by_coord({"skycoord": events.radec, "energy": events.energy})
        #    counts.sum_over_axes().smooth(2).plot(stretch="log", vmax=50, cmap="nipy_spectral")

        axis = MapAxis.from_nodes(
            counts.geom.axes[0].center, name="energy", unit="GeV", interp="log"
        )
        print(counts.geom)
        wcs = counts.geom.wcs
        geom = WcsGeom(wcs=wcs, npix=counts.geom.npix, axes=[axis])
        coords = counts.geom.get_coord()
        data = exposure_hpx.interp_by_coord(coords)
        exposure = WcsNDMap(geom, data, unit=exposure_hpx.unit, dtype=float)

        # read PSF
        psf_kernel = PSFKernel.from_table_psf(
            psf, counts.geom, max_radius=psf_margin * u.deg
        )

        # Energy Dispersion
        e_true = exposure.geom.axes[0].edges
        e_reco = counts.geom.axes[0].edges
        edisp = EnergyDispersion.from_diagonal_response(e_true=e_true, e_reco=e_reco)

        # fit mask
        if coords["lon"].min() < 90 * u.deg and coords["lon"].max() > 270 * u.deg:
            coords["lon"][coords["lon"].value > 180] -= 360 * u.deg
        mask = (
            (coords["lon"] >= coords["lon"].min() + psf_margin * u.deg)
            & (coords["lon"] <= coords["lon"].max() - psf_margin * u.deg)
            & (coords["lat"] >= coords["lat"].min() + psf_margin * u.deg)
            & (coords["lat"] <= coords["lat"].max() - psf_margin * u.deg)
        )
        mask_fermi = WcsNDMap(counts.geom, mask)
        print("Maps done")

        # IEM
        model_iem = SkyDiffuseCube(iem_fermi, norm=1.1, tilt=0.03, name="iem_v06")
        # norm=1.1, tilt=0.03 see paper appendix A
        eval_iem = MapEvaluator(
            model=model_iem, exposure=exposure, psf=psf_kernel, edisp=edisp
        )
        bkg_iem = eval_iem.compute_npred()

        print("Background IEM done")

        # ISO
        eval_iso = MapEvaluator(model=model_iso, exposure=exposure, edisp=edisp)
        bkg_iso = eval_iso.compute_npred()
        print("Background ISO done")

        # merge iem and iso, only one local normalization is fitted
        background_total = bkg_iem + bkg_iso
        background_model = BackgroundModel(background_total)
        background_model.parameters["norm"].min = 0.0

        # Sources model
        in_roi = FHL3.positions.galactic.contained_by(wcs)
        FHL3_roi = []
        for ks in range(len(FHL3.table)):
            if in_roi[ks] == True:
                model = FHL3[ks].sky_model()
                model.spatial_model.parameters.freeze_all()  # freeze spatial
                model.spectral_model.parameters["amplitude"].min = 0.0
                if isinstance(model.spectral_model, PowerLawSpectralModel):
                    model.spectral_model.parameters["index"].min = 0.1
                    model.spectral_model.parameters["index"].max = 10.0

                else:
                    model.spectral_model.parameters["alpha"].min = 0.1
                    model.spectral_model.parameters["alpha"].max = 10.0

                FHL3_roi.append(model)
        model_total = SkyModels(FHL3_roi)
        print("Nsources " + str(len(FHL3_roi)))

        # Dataset
        dataset = MapDataset(
            model=model_total,
            counts=counts,
            exposure=exposure,
            psf=psf_kernel,
            edisp=edisp,
            background_model=background_model,
            mask_fit=mask_fermi,
            name="3FHL_ROI_num" + str(kr),
        )
        cat_stat = dataset.likelihood()

        datasets = Datasets([dataset])
        results = Fit(datasets).run(optimize_opts=optimize_opts)
        fit_stat = datasets.likelihood()
        print(results)

        if results.message == "Optimization failed.":
            continue
        covariance = results.parameters.covariance
        np.save(resdir + "/3FHL_ROI_num" + str(kr) + "_covariance.npy", covariance)
        np.savez(
            resdir + "/3FHL_ROI_num" + str(kr) + "_fit_infos.npz",
            message=results.message,
            stat=[cat_stat, fit_stat],
        )

        exec_time = time() - roi_time
        print("ROI time (s): ", exec_time)

        for model in FHL3_roi:
            if (
                FHL3[model.name].data["ROI_num"] == kr
                and FHL3[model.name].data["Signif_Avg"] >= sig_cut
            ):
                flux_points = FluxPointsEstimator(
                    datasets=datasets, e_edges=El_flux, source=model.name, sigma_ul=2.0
                ).run()
                filename = resdir + "/" + model.name + "_flux_points.fits"
                flux_points.write(filename, overwrite=True)

        exec_time = time() - roi_time - exec_time
        print("Flux points time (s): ", exec_time)
        datasets.to_yaml(path=Path(resdir), prefix=dataset.name, overwrite=True)
    else:
        try:
            filedata = Path(resdir + "/3FHL_ROI_num" + str(kr) + "_datasets.yaml")
            filemodel = Path(resdir + "/3FHL_ROI_num" + str(kr) + "_models.yaml")
            dataset = list(Datasets.from_yaml(filedata, filemodel))[0]
        except:
            continue

    try:
        plt.figure(figsize=(6, 6), dpi=150)
        ax, cb = dataset.plot_residuals(
            method="diff/sqrt(model)",
            smooth_kernel="gauss",
            smooth_radius=0.1 * u.deg,
            region=None,
            figsize=(6, 6),
            cmap="jet",
            vmin=-3,
            vmax=3,
        )
        plt.title("Residuals: (Nobs-Npred)/sqrt(Npred)")
        plt.savefig(resdir + "/resi_" + dataset.name + ".png", dpi=150)

        plt.figure(figsize=(6, 6), dpi=150)
        nobs = dataset.counts.sum_over_axes()
        fig, ax, cb = nobs.plot(cmap=cm.nipy_spectral, add_cbar=True, norm=LogNorm())
        plt.title("Nobs")
        cl = cb.get_clim()
        plt.savefig(resdir + "/counts_" + dataset.name + ".png", dpi=plt.gcf().dpi)

        plt.figure(figsize=(6, 6), dpi=150)
        npred = dataset.npred().sum_over_axes()
        fig, ax, cb = npred.plot(cmap=cm.nipy_spectral, add_cbar=True, norm=LogNorm())
        plt.title("Npred")
        cb.set_clim(cl)
        plt.savefig(resdir + "/npred_" + dataset.name + ".png", dpi=plt.gcf().dpi)

    except:
        pass

    pars = dataset.parameters
    pars.covariance = np.load(resdir + "/" + dataset.name + "_covariance.npy")
    infos = np.load(resdir + "/3FHL_ROI_num" + str(kr) + "_fit_infos.npz")
    message.append(infos["message"])
    stat.append(infos["stat"])

    for model in list(dataset.model):
        if (
            FHL3[model.name].data["ROI_num"] == kr
            and FHL3[model.name].data["Signif_Avg"] >= sig_cut
        ):
            try:
                model.spatial_model.parameters.covariance = pars.get_subcovariance(
                    model.spatial_model.parameters
                )
                model.spectral_model.parameters.covariance = pars.get_subcovariance(
                    model.spectral_model.parameters
                )
                dataset.background_model.parameters.covariance = pars.get_subcovariance(
                    dataset.background_model.parameters
                )
                res_spec = model.spectral_model
                cat_spec = FHL3[model.name].spectral_model()

                res_fp = FluxPoints.read(
                    resdir + "/" + model.name + "_flux_points.fits"
                )
                res_fp.table["is_ul"] = res_fp.table["ts"] < 1.0
                cat_fp = FHL3[model.name].flux_points.to_sed_type("dnde")
                ind = ~(res_fp.is_ul) & ~(cat_fp.is_ul)
                errel["flux_points"] += list(
                    relative_error(cat_fp.table["dnde"][ind], res_fp.table["dnde"][ind])
                )
                cat_fp_sel += list(cat_fp.table["dnde"][ind])
                cat_dfp_sel += list(
                    np.maximum(
                        cat_fp.table["dnde_errn"][ind].data,
                        cat_fp.table["dnde_errp"][ind].data,
                    )
                )
                energy_range = [0.01, 2] * u.TeV
                plt.figure(figsize=(6, 6), dpi=150)
                ax = cat_spec.plot(
                    energy_range=energy_range,
                    energy_power=2,
                    label="3FHL Catalogue",
                    color="k",
                )
                cat_spec.plot_error(ax=ax, energy_range=energy_range, energy_power=2)
                res_spec.plot(
                    ax=ax,
                    energy_range=energy_range,
                    energy_power=2,
                    label="Gammapy fit",
                    color="b",
                )
                res_spec.plot_error(
                    ax=ax, energy_range=energy_range, energy_power=2, color="c"
                )

                cat_fp.plot(ax=ax, energy_power=2, color="k")
                res_fp.plot(ax=ax, energy_power=2, color="b")
                plt.legend()
                tag = model.name
                tag.replace(' ', '_')
                tag.replace('.', '_')
                plt.savefig(
                    resdir + "/spec_" + tag + "_ROI_num" + str(kr) + ".png",
                    dpi=plt.gcf().dpi,
                )

                if isinstance(model.spectral_model, PowerLawSpectralModel):

                    PL_index.append(
                        [
                            cat_spec.parameters["index"].value,
                            res_spec.parameters["index"].value,
                            cat_spec.parameters.error("index"),
                            res_spec.parameters.error("index"),
                        ]
                    )
                    PL_amplitude.append(
                        [
                            cat_spec.parameters["amplitude"].value,
                            res_spec.parameters["amplitude"].value,
                            cat_spec.parameters.error("amplitude"),
                            res_spec.parameters.error("amplitude"),
                        ]
                    )
                    PL_BKG_norm.append(
                        [
                            dataset.background_model.parameters["norm"].value,
                            dataset.background_model.parameters.error("norm"),
                        ]
                    )
                    PL_tags.append([dataset.name, model.name])

                if isinstance(model.spectral_model, LogParabolaSpectralModel):
                    LP_alpha.append(
                        [
                            cat_spec.parameters["alpha"].value,
                            res_spec.parameters["alpha"].value,
                            cat_spec.parameters.error("alpha"),
                            res_spec.parameters.error("alpha"),
                        ]
                    )
                    LP_beta.append(
                        [
                            cat_spec.parameters["beta"].value,
                            res_spec.parameters["beta"].value,
                            cat_spec.parameters.error("beta"),
                            res_spec.parameters.error("beta"),
                        ]
                    )
                    LP_amplitude.append(
                        [
                            cat_spec.parameters["amplitude"].value,
                            res_spec.parameters["amplitude"].value,
                            cat_spec.parameters.error("amplitude"),
                            res_spec.parameters.error("amplitude"),
                        ]
                    )
                    LP_BKG_norm.append(
                        [
                            dataset.background_model.parameters["norm"].value,
                            dataset.background_model.parameters.error("norm"),
                        ]
                    )
                    LP_tags.append([dataset.name, model.name])

            except:
                pass
    plt.close("all")


# In[ ]:


# likelihood
message = np.array(message)
print(
    "Optimization success. : ",
    sum(message == "Optimization terminated successfully.") / len(message),
)
print("Optimization failed. : ", sum(message == "Optimization failed.") / len(message))
print(
    "Optimization failed. Estimated distance to minimum too large. : ",
    sum(message == "Optimization failed. Estimated distance to minimum too large.")
    / len(message),
)

plt.figure(figsize=(6, 6), dpi=120)
stat = np.array(stat)
plt.plot(stat[:, 0], stat[:, 0] - stat[:, 1], "+ ")
xl = plt.xlim()
yl = plt.ylim()
plt.ylim([-100, 100])
plt.plot(xl, [0, 0], "-k")
plt.xlabel("Cash Catalog")
plt.ylabel("dCash Catalog - Fit")
plt.savefig(resdir + "/Cash_stat_corr" + ".png", dpi=plt.gcf().dpi)


# In[ ]:


# flux points
name = "flux_points"
errel[name] = np.array(errel[name])
print("\n" + name)
print("Rel. err. <10%:", sum(abs(errel[name]) < 0.1) / len(errel[name]))
print("Rel. err. <30%:", sum(abs(errel[name]) < 0.3) / len(errel[name]))
print("Rel. err. mean:", np.nanmean(errel[name]))

plt.figure(figsize=(6, 6), dpi=150)
plt.semilogx(cat_fp_sel, errel[name], " .")
xl = plt.xlim()
plt.plot(xl, [0, 0], "-k")
plt.ylim([-1, 1])
plt.xlabel(name + " Catalog")
plt.ylabel("Relative error: (Gammapy-Catalog)/Catalog")
if "amplitude" in name:
    ax = plt.gca()
    ax.set_xscale("log")
plt.savefig(resdir + "/" + name + "_errel" + ".png", dpi=plt.gcf().dpi)


# Parameters diagnostics

diags = [
    np.array(PL_index),
    np.array(PL_amplitude),
    np.array(LP_alpha),
    np.array(LP_beta),
    np.array(LP_amplitude),
]

PL_BKG_norm = np.array(PL_BKG_norm)
LP_BKG_norm = np.array(LP_BKG_norm)

names = ["PL_index", "PL_amplitude", "LP_alpha", "LP_beta", "LP_amplitude"]
ndiag = len(diags)
for kd in range(ndiag):
    diag = diags[kd]
    name = names[kd]
    if len(diag) != 0:
        # fit vs. catalogue parameters comparisons
        print("\n" + name)
        ind = diag[:, 3] / diag[:, 1] < 0.1
        print("dx/x <10% : ", sum(ind) / len(diag[:, 1]))
        ind = diag[:, 3] / diag[:, 1] < 0.3
        print("dx/x <30% : ", sum(ind) / len(diag[:, 1]))

        plt.figure(figsize=(6, 6), dpi=150)
        plt.errorbar(
            diag[ind, 0], diag[ind, 1], xerr=diag[ind, 2], yerr=diag[ind, 3], ls=" "
        )
        if "amplitude" in name:
            ax = plt.gca()
            ax.set_xscale("log")
            ax.set_yscale("log")
        xl = plt.xlim()
        plt.plot(xl, xl, "-k")
        plt.xlabel(name + " Catalog")
        plt.ylabel(name + " Gammapy")
        plt.savefig(resdir + "/" + name + "_corr" + ".png", dpi=plt.gcf().dpi)

        # relative error and compatibility
        compatibility[name] = iscompatible(
            diag[:, 0], diag[:, 1], diag[:, 2], diag[:, 3]
        )
        errel[name] = relative_error(diag[:, 0], diag[:, 1])

        print("Rel. err. <10%:", sum(abs(errel[name]) < 0.1) / len(errel[name]))
        print("Rel. err. <30%:", sum(abs(errel[name]) < 0.3) / len(errel[name]))
        print("Rel. err. mean:", np.nanmean(errel[name]))
        print("compatibility:", sum(compatibility[name]) / len(compatibility[name]))

        plt.figure(figsize=(6, 6), dpi=150)
        plt.plot(diag[:, 0], errel[name], " .")
        xl = plt.xlim()
        plt.plot(xl, [0, 0], "-k")
        plt.ylim([-0.3, 0.3])
        plt.xlabel(name + " Catalog")
        plt.ylabel("Relative error: (Gammapy-Catalog)/Catalog")
        if "amplitude" in name:
            ax = plt.gca()
            ax.set_xscale("log")
        plt.savefig(resdir + "/" + name + "_errel" + ".png", dpi=plt.gcf().dpi)

        # Background correlation
        plt.figure(figsize=(6, 6), dpi=150)
        if "PL" in name:
            bkg_norm = PL_BKG_norm
        else:
            bkg_norm = LP_BKG_norm
        plt.errorbar(
            bkg_norm[:, 0], errel[name], xerr=bkg_norm[:, 1], ls=" ",
        )
        plt.xlim([0.7, 1.3])
        plt.plot([0.7, 1.3], [0, 0], "-k")
        plt.xlabel("BKG norm")
        plt.ylabel(name + " Relative error")
        plt.savefig(resdir + "/" + name + "_errel_BKGcorr" + ".png", dpi=plt.gcf().dpi)

        plt.close("all")


# In[ ]:


exec_time = time() - start_time
print("\n Execution time (s): ", exec_time)

