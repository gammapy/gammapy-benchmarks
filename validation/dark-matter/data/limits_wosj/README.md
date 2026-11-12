# Directory structure

These are quantiles limits for each DM mass for annihilation and decay **obtained NOT using the uncertainty in the astrophysical factor**. 
Each file contains:
- source name `name`
- type of profile: einasto or burkert `profile`
- channel for annihilation or decay `ch`
- dark matter mass `mass` in GeV
- the quantile of the limits (for three quantiles at median, 1, 2 sigma) `quantile`
- annihilation cross section limit `sigma_v` in cm-3s-1 or lifetime limit `tau` in s

In the **filename**
- t600 e.g. stands for `observation time of 600h`
- `wosj` that limits are computed without the uncertainties sigmaJ of the astrophysical factor
- `quantiles` that limits are reported in their quantiles


The directory is structured as follows:

```
├── README.md
├── annihilation
│   ├── combined
│   │   ├── Combined_annichilation_CBe_Wil1_t600_wosj_quantiles.csv
│   │   └── Combined_annihilation_AllSources_t600_wosj_quantiles.csv
│   └── single
│       ├── CBe_burkert_annihilation_t100h_wosj_quantiles.csv
│       ├── CBe_einasto_annihilation_t100h_wosj_quantiles.csv
│       ├── DraI_burkert_annihilation_t100h_wosj_quantiles.csv
│       ├── DraI_einasto_annihilation_t100h_wosj_quantiles.csv
│       ├── RetII_burkert_annihilation_t100h_wosj_quantiles.csv
│       ├── RetII_einasto_annihilation_t100h_wosj_quantiles.csv
│       ├── Scl_burkert_annihilation_t100h_wosj_quantiles.csv
│       ├── Scl_einasto_annihilation_t100h_wosj_quantiles.csv
│       ├── SgrII_burkert_annihilation_t100h_wosj_quantiles.csv
│       ├── SgrII_einasto_annihilation_t100h_wosj_quantiles.csv
│       ├── UMaII_burkert_annihilation_t100h_wosj_quantiles.csv
│       ├── UMaII_einasto_annihilation_t100h_wosj_quantiles.csv
│       ├── UMi_burkert_annihilation_t100h_wosj_quantiles.csv
│       ├── UMi_einasto_annihilation_t100h_wosj_quantiles.csv
│       ├── Wil1_burkert_annihilation_t100h_wosj_quantiles.csv
│       └── Wil1_einasto_annihilation_t100h_wosj_quantiles.csv
└── decay
    ├── combined
    │   ├── Combined_decay_einasto_AllSources_t600_wosj_quantiles.csv
    │   └── Combined_decay_einasto_CBe_UMi_t600)wosj_quantiles.csv
    └── single
        ├── CBe_burkert_decay_t100h_wosj_quantiles.csv
        ├── CBe_einasto_decay_t100h_wosj_quantiles.csv
        ├── DraI_burkert_decay_t100h_wosj_quantiles.csv
        ├── DraI_einasto_decay_t100h_wosj_quantiles.csv
        ├── RetII_burkert_decay_t100h_wosj_quantiles.csv
        ├── RetII_einasto_decay_t100h_wosj_quantiles.csv
        ├── Scl_burkert_decay_t100h_wosj_quantiles.csv
        ├── Scl_einasto_decay_t100h_wosj_quantiles.csv
        ├── SgrII_burkert_decay_t100h_wosj_quantiles.csv
        ├── SgrII_einasto_decay_t100h_wosj_quantiles.csv
        ├── UMaII_burkert_decay_t100h_wosj_quantiles.csv
        ├── UMaII_einasto_decay_t100h_wosj_quantiles.csv
        ├── UMi_burkert_decay_t100h_wosj_quantiles.csv
        ├── UMi_einasto_decay_t100h_wosj_quantiles.csv
        ├── Wil1_burkert_decay_t100h_wosj_quantiles.csv
        └── Wil1_einasto_decay_t100h_wosj_quantiles.csv
```
