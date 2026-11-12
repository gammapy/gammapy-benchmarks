# KSP_dSPh_material

This repository contains
- tables, files
- noteboook to reproduce graphics
for the KSP project of CTAO related to dark matter searches at dwarf Spheroidal Galaxies, which is under publication.

## Details on materials
* IRFs: as taken from Prod5.1 of CTA
* clumpy: file generated using CLUMPY
** DMdensity profiles
** Dfactors --> Decay Asttrophysical factor profiles and 2d-maps for Einasto and Burkert profiles
** Jfactors --> Annihilation Asttrophysical factor profiles and 2d-maps for Einasto and Burkert profiles


.
├── AtProduction_gammas. dat
├── CONTRIBUTING.md
├── IRFs
│   └── cta
│       └── prod5-v0.1
│           ├── bcf
│           └── caldb.indx
├  README.md
├── clumpy
│   ├── DMdensity
│   │   ├── README.md
│   │   ├── dm_densityprofile_burkert.dat
│   │   ├── dm_densityprofile_burkert_others.dat
│   │   ├── dm_densityprofile_einasto.dat
│   │   └── dm_densityprofile_einasto_others.dat
│   ├── Dfactors
│   │   ├── 2Dmaps
│   │   │   ├── Burkert
│   │   │   │   ├── decay_CBe2D_FOVdiameter10.0deg_nside1024-DFACTOR-Dtot-image.fits
│   │   │   │   ├── decay_DraI2D_FOVdiameter10.0deg_nside1024-DFACTOR-Dtot-image.fits
│   │   │   │   ├── decay_RetII2D_FOVdiameter10.0deg_nside1024-DFACTOR-Dtot-image.fits
│   │   │   │   ├── decay_Scl2D_FOVdiameter10.0deg_nside1024-DFACTOR-Dtot-image.fits
│   │   │   │   ├── decay_SgrII2D_FOVdiameter10.0deg_nside1024-DFACTOR-Dtot-image.fits
│   │   │   │   ├── decay_UMaII2D_FOVdiameter10.0deg_nside1024-DFACTOR-Dtot-image.fits
│   │   │   │   ├── decay_UMi2D_FOVdiameter10.0deg_nside1024-DFACTOR-Dtot-image.fits
│   │   │   │   └── decay_Wil12D_FOVdiameter10.0deg_nside1024-DFACTOR-Dtot-image.fits
│   │   │   ├── Einasto
│   │   │   │   ├── decay_CBe2D_FOVdiameter10.0deg_nside1024-DFACTOR-Dtot-image.fits
│   │   │   │   ├── decay_DraI2D_FOVdiameter10.0deg_nside1024-DFACTOR-Dtot-image.fits
│   │   │   │   ├── decay_RetII2D_FOVdiameter10.0deg_nside1024-DFACTOR-Dtot-image.fits
│   │   │   │   ├── decay_Scl2D_FOVdiameter10.0deg_nside1024-DFACTOR-Dtot-image.fits
│   │   │   │   ├── decay_SgrII2D_FOVdiameter10.0deg_nside1024-DFACTOR-Dtot-image.fits
│   │   │   │   ├── decay_UMaII2D_FOVdiameter10.0deg_nside1024-DFACTOR-Dtot-image.fits
│   │   │   │   ├── decay_UMi2D_FOVdiameter10.0deg_nside1024-DFACTOR-Dtot-image.fits
│   │   │   │   └── decay_Wil12D_FOVdiameter10.0deg_nside1024-DFACTOR-Dtot-image.fits
│   │   │   └── makeFitsImage.py
│   │   ├── README.md
│   │   └── profile
│   │       ├── Burkert
│   │       │   ├── rvir_BooI_MCMC.dat.Dalphaint_cls.output
│   │       │   ├── rvir_CBe_MCMC.dat.Dalphaint_cls.output
│   │       │   ├── rvir_DraI_MCMC.dat.Dalphaint_cls.output
│   │       │   ├── rvir_GruII_MCMC.dat.Dalphaint_cls.output
│   │       │   ├── rvir_RetII_MCMC.dat.Dalphaint_cls.output
│   │       │   ├── rvir_Scl_MCMC.dat.Dalphaint_cls.output
│   │       │   ├── rvir_Seg1_MCMC.dat.Dalphaint_cls.output
│   │       │   ├── rvir_Sex_MCMC.dat.Dalphaint_cls.output
│   │       │   ├── rvir_SgrII_MCMC.dat.Dalphaint_cls.output
│   │       │   ├── rvir_SgrI_MCMC.dat.Dalphaint_cls.output
│   │       │   ├── rvir_TriII_MCMC.dat.Dalphaint_cls.output
│   │       │   ├── rvir_UMaII_MCMC.dat.Dalphaint_cls.output
│   │       │   ├── rvir_UMi_MCMC.dat.Dalphaint_cls.output
│   │       │   └── rvir_Wil1_MCMC.dat.Dalphaint_cls.output
│   │       └── Einasto
│   │           ├── rvir_BooI_MCMC.dat.Dalphaint_cls.output
│   │           ├── rvir_CBe_MCMC.dat.Dalphaint_cls.output
│   │           ├── rvir_DraI_MCMC.dat.Dalphaint_cls.output
│   │           ├── rvir_GruII_MCMC.dat.Dalphaint_cls.output
│   │           ├── rvir_RetII_MCMC.dat.Dalphaint_cls.output
│   │           ├── rvir_Scl_MCMC.dat.Dalphaint_cls.output
│   │           ├── rvir_Seg1_MCMC.dat.Dalphaint_cls.output
│   │           ├── rvir_Sex_MCMC.dat.Dalphaint_cls.output
│   │           ├── rvir_SgrII_MCMC.dat.Dalphaint_cls.output
│   │           ├── rvir_SgrI_MCMC.dat.Dalphaint_cls.output
│   │           ├── rvir_TriII_MCMC.dat.Dalphaint_cls.output
│   │           ├── rvir_UMaII_MCMC.dat.Dalphaint_cls.output
│   │           ├── rvir_UMi_MCMC.dat.Dalphaint_cls.output
│   │           └── rvir_Wil1_MCMC.dat.Dalphaint_cls.output
│   ├── Jfactors
│   │   ├── 2Dmaps
│   │   │   ├── Burkert
│   │   │   │   ├── annihil_CBe2D_FOVdiameter10.0deg_nside1024-JFACTOR-Jtot-image.fits
│   │   │   │   ├── annihil_DraI2D_FOVdiameter10.0deg_nside1024-JFACTOR-Jtot-image.fits
│   │   │   │   ├── annihil_RetII2D_FOVdiameter10.0deg_nside1024-JFACTOR-Jtot-image.fits
│   │   │   │   ├── annihil_Scl2D_FOVdiameter10.0deg_nside1024-JFACTOR-Jtot-image.fits
│   │   │   │   ├── annihil_SgrII2D_FOVdiameter10.0deg_nside1024-JFACTOR-Jtot-image.fits
│   │   │   │   ├── annihil_UMaII2D_FOVdiameter10.0deg_nside1024-JFACTOR-Jtot-image.fits
│   │   │   │   ├── annihil_UMi2D_FOVdiameter10.0deg_nside1024-JFACTOR-Jtot-image.fits
│   │   │   │   └── annihil_Wil12D_FOVdiameter10.0deg_nside1024-JFACTOR-Jtot-image.fits
│   │   │   ├── Einasto
│   │   │   │   ├── annihil_CBe2D_FOVdiameter10.0deg_nside1024-JFACTOR-Jtot-image.fits
│   │   │   │   ├── annihil_DraI2D_FOVdiameter10.0deg_nside1024-JFACTOR-Jtot-image.fits
│   │   │   │   ├── annihil_RetII2D_FOVdiameter10.0deg_nside1024-JFACTOR-Jtot-image.fits
│   │   │   │   ├── annihil_Scl2D_FOVdiameter10.0deg_nside1024-JFACTOR-Jtot-image.fits
│   │   │   │   ├── annihil_SgrII2D_FOVdiameter10.0deg_nside1024-JFACTOR-Jtot-image.fits
│   │   │   │   ├── annihil_UMaII2D_FOVdiameter10.0deg_nside1024-JFACTOR-Jtot-image.fits
│   │   │   │   ├── annihil_UMi2D_FOVdiameter10.0deg_nside1024-JFACTOR-Jtot-image.fits
│   │   │   │   └── annihil_Wil12D_FOVdiameter10.0deg_nside1024-JFACTOR-Jtot-image.fits
│   │   │   └── makeFitsImage.py
│   │   ├── README.md
│   │   └── profile
│   │       ├── Burkert
│   │       │   ├── rvir_BooI_MCMC.dat.Jalphaint_cls.output
│   │       │   ├── rvir_CBe_MCMC.dat.Jalphaint_cls.output
│   │       │   ├── rvir_DraI_MCMC.dat.Jalphaint_cls.output
│   │       │   ├── rvir_GruII_MCMC.dat.Jalphaint_cls.output
│   │       │   ├── rvir_RetII_MCMC.dat.Jalphaint_cls.output
│   │       │   ├── rvir_Scl_MCMC.dat.Jalphaint_cls.output
│   │       │   ├── rvir_Seg1_MCMC.dat.Jalphaint_cls.output
│   │       │   ├── rvir_Sex_MCMC.dat.Jalphaint_cls.output
│   │       │   ├── rvir_SgrII_MCMC.dat.Jalphaint_cls.output
│   │       │   ├── rvir_SgrI_MCMC.dat.Jalphaint_cls.output
│   │       │   ├── rvir_TriII_MCMC.dat.Jalphaint_cls.output
│   │       │   ├── rvir_UMaII_MCMC.dat.Jalphaint_cls.output
│   │       │   ├── rvir_UMi_MCMC.dat.Jalphaint_cls.output
│   │       │   └── rvir_Wil1_MCMC.dat.Jalphaint_cls.output
│   │       └── Einasto
│   │           ├── rvir_BooI_MCMC.dat.Jalphaint_cls.output
│   │           ├── rvir_CBe_MCMC.dat.Jalphaint_cls.output
│   │           ├── rvir_DraI_MCMC.dat.Jalphaint_cls.output
│   │           ├── rvir_GruII_MCMC.dat.Jalphaint_cls.output
│   │           ├── rvir_RetII_MCMC.dat.Jalphaint_cls.output
│   │           ├── rvir_Scl_MCMC.dat.Jalphaint_cls.output
│   │           ├── rvir_Seg1_MCMC.dat.Jalphaint_cls.output
│   │           ├── rvir_Sex_MCMC.dat.Jalphaint_cls.output
│   │           ├── rvir_SgrII_MCMC.dat.Jalphaint_cls.output
│   │           ├── rvir_SgrI_MCMC.dat.Jalphaint_cls.output
│   │           ├── rvir_TriII_MCMC.dat.Jalphaint_cls.output
│   │           ├── rvir_UMaII_MCMC.dat.Jalphaint_cls.output
│   │           ├── rvir_UMi_MCMC.dat.Jalphaint_cls.output
│   │           └── rvir_Wil1_MCMC.dat.Jalphaint_cls.output
│   └── README.md
├── limits_wosj
│   ├── README.md
│   ├── annihilation
│   │   ├── combined
│   │   │   ├── Combined_annichilation_CBe_Wil1_t600_wosj_quantiles.csv
│   │   │   └── Combined_annihilation_AllSources_t600_wosj_quantiles.csv
│   │   └── single
│   │       ├── CBe_burkert_annihilation_t100h_wosj_quantiles.csv
│   │       ├── CBe_einasto_annihilation_t100h_wosj_quantiles.csv
│   │       ├── DraI_burkert_annihilation_t100h_wosj_quantiles.csv
│   │       ├── DraI_einasto_annihilation_t100h_wosj_quantiles.csv
│   │       ├── RetII_burkert_annihilation_t100h_wosj_quantiles.csv
│   │       ├── RetII_einasto_annihilation_t100h_wosj_quantiles.csv
│   │       ├── Scl_burkert_annihilation_t100h_wosj_quantiles.csv
│   │       ├── Scl_einasto_annihilation_t100h_wosj_quantiles.csv
│   │       ├── SgrII_burkert_annihilation_t100h_wosj_quantiles.csv
│   │       ├── SgrII_einasto_annihilation_t100h_wosj_quantiles.csv
│   │       ├── UMaII_burkert_annihilation_t100h_wosj_quantiles.csv
│   │       ├── UMaII_einasto_annihilation_t100h_wosj_quantiles.csv
│   │       ├── UMi_burkert_annihilation_t100h_wosj_quantiles.csv
│   │       ├── UMi_einasto_annihilation_t100h_wosj_quantiles.csv
│   │       ├── Wil1_burkert_annihilation_t100h_wosj_quantiles.csv
│   │       └── Wil1_einasto_annihilation_t100h_wosj_quantiles.csv
│   └── decay
│       ├── combined
│       │   ├── Combined_decay_einasto_AllSources_t600_wosj_quantiles.csv
│       │   └── Combined_decay_einasto_CBe_UMi_t600_wosj_quantiles.csv
│       └── single
│           ├── CBe_burkert_decay_t100h_wosj_quantiles.csv
│           ├── CBe_einasto_decay_t100h_wosj_quantiles.csv
│           ├── DraI_burkert_decay_t100h_wosj_quantiles.csv
│           ├── DraI_einasto_decay_t100h_wosj_quantiles.csv
│           ├── RetII_burkert_decay_t100h_wosj_quantiles.csv
│           ├── RetII_einasto_decay_t100h_wosj_quantiles.csv
│           ├── Scl_burkert_decay_t100h_wosj_quantiles.csv
│           ├── Scl_einasto_decay_t100h_wosj_quantiles.csv
│           ├── SgrII_burkert_decay_t100h_wosj_quantiles.csv
│           ├── SgrII_einasto_decay_t100h_wosj_quantiles.csv
│           ├── UMaII_burkert_decay_t100h_wosj_quantiles.csv
│           ├── UMaII_einasto_decay_t100h_wosj_quantiles.csv
│           ├── UMi_burkert_decay_t100h_wosj_quantiles.csv
│           ├── UMi_einasto_decay_t100h_wosj_quantiles.csv
│           ├── Wil1_burkert_decay_t100h_wosj_quantiles.csv
│           └── Wil1_einasto_decay_t100h_wosj_quantiles.csv
├── limits_wsj
│   ├── README.md
│   ├── annihilation
│   │   ├── combined
│   │   │   ├── combined_burkert_annihilation_DraI_UMi_t600_wsj_quantiles.csv
│   │   │   ├── combined_burkert_annihilation_allSources_t600_wsj_quantiles.csv
│   │   │   ├── combined_einasto_annihilation_CBe_Wil1_t600_wsj_quantiles.csv
│   │   │   ├── combined_einasto_annihilation_DraI_UMi_t600_wsj_quantiles.csv
│   │   │   └── combined_einasto_annihilation_allSources_t600_wsj_quantiles.csv
│   │   └── single
│   │       ├── CBe_burkert_annichilation_t100_wsj_quantiles.csv
│   │       ├── CBe_einasto_annihilation_t100_wsj_quantiles.csv
│   │       ├── DraI_burkert_annichilation_t100_wsj_quantiles.csv
│   │       ├── DraI_einasto_annihilation_t100_wsj_quantiles.csv
│   │       ├── RetII_burkert_annichilation_t100_wsj_quantiles.csv
│   │       ├── RetII_einasto_annihilation_t100_wsj_quantiles.csv
│   │       ├── Scl_burkert_annichilation_t100_wsj_quantiles.csv
│   │       ├── Scl_einasto_annihilation_t100_wsj_quantiles.csv
│   │       ├── SgrII_burkert_annichilation_t100_wsj_quantiles.csv
│   │       ├── SgrII_einasto_annihilation_t100_wsj_quantiles.csv
│   │       ├── UMaII_burkert_annichilation_t100_wsj_quantiles.csv
│   │       ├── UMaII_einasto_annihilation_t100_wsj_quantiles.csv
│   │       ├── UMi_burkert_annichilation_t100_wsj_quantiles.csv
│   │       ├── UMi_einasto_annihilation_t100_wsj_quantiles.csv
│   │       ├── Wil1_burkert_annichilation_t100_wsj_quantiles.csv
│   │       └── Wil1_einasto_annihilation_t100_wsj_quantiles.csv
│   └── decay
│       ├── combined
│       │   ├── combined_burkert_decay_DraI_UMi_t600_wsj_quantiles.csv
│       │   ├── combined_burkert_decay_allSources_t600_wsj_quantiles.csv
│       │   ├── combined_einasto_decay_CBe_Wil1_t600_wsj_quantiles.csv
│       │   ├── combined_einasto_decay_DraI_UMi_t600_wsj_quantiles.csv
│       │   └── combined_einasto_decay_allSources_t600_wsj_quantiles.csv
│       └── single
│           ├── CBe_burkert_decay_t100_wsj_quantiles.csv
│           ├── CBe_einasto_decay_t100_wsj_quantiles.csv
│           ├── DraI_burkert_decay_t100_wsj_quantiles.csv
│           ├── DraI_einasto_decay_t100_wsj_quantiles.csv
│           ├── RetII_burkert_decay_t100_wsj_quantiles.csv
│           ├── RetII_einasto_decay_t100_wsj_quantiles.csv
│           ├── Scl_burkert_decay_t100_wsj_quantiles.csv
│           ├── Scl_einasto_decay_t100_wsj_quantiles.csv
│           ├── SgrII_burkert_decay_t100_wsj_quantiles.csv
│           ├── SgrII_einasto_decay_t100_wsj_quantiles.csv
│           ├── UMaII_burkert_decay_t100_wsj_quantiles.csv
│           ├── UMaII_einasto_decay_t100_wsj_quantiles.csv
│           ├── UMi_burkert_decay_t100_wsj_quantiles.csv
│           ├── UMi_einasto_decay_t100_wsj_quantiles.csv
│           ├── Wil1_burkert_decay_t100_wsj_quantiles.csv
│           └── Wil1_einasto_decay_t100_wsj_quantiles.csv
