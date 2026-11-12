# Directory structure

These are quantiles limits for each DM mass for annihilation and decay **obtained using the uncertainty in the astrophysical factor**. 
Each file contains:
- source name `name`
- type of profile: einasto or burkert `profile`
- channel for annihilation or decay `ch`
- dark matter mass `mass` in GeV
- the quantile of the limits (for three quantiles at median, 1, 2 sigma) `quantile`
- annihilation cross section limit `sigma_v` in cm-3s-1 or lifetime limit `tau` in s

In the **filename**
- t600 e.g. stands for `observation time of 600h`
- `wsj` that limits are computed with uncertainties sigmaJ of the astrophysical factor
- `quantiles` that limits are reported in their quantiles


The directory is structured as follows:

```
.
├── annihilation
│   ├── combined
│   │   ├── combined_einasto_annihilation_CBe_Wil1_t600_wsj_quantiles.csv
│   │   ├── combined_einasto_annihilation_DraI_UMi_t600_wsj_quantiles.csv
│   │   └── combined_einasto_annihilation_allSources_t600_wsj_quantiles.csv
│   └── single
│       ├── CBe_einasto_annihilation_t100_wsj_quantiles.csv
│       ├── DraI_einasto_annhilation_t100_wsj_quantiles.csv
│       ├── RetII_einasto_annhilation_t100_wsj_quantiles.csv
│       ├── Scl_einasto_annhilation_t100_wsj_quantiles.csv
│       ├── SgrII_einasto_annhilation_t100_wsj_quantiles.csv
│       ├── UMaII_einasto_annhilation_t100_wsj_quantiles.csv
│       ├── UMi_einasto_annhilation_t100_wsj_quantiles.csv
│       └── Wil1_einasto_annhilation_t100_wsj_quantiles.csv
└── decay
    ├── combined
    │   ├── combined_einasto_decay_CBe_Wil1_t600_wsj_quantiles.csv
    │   ├── combined_einasto_decay_DraI_UMi_t600_wsj_quantiles.csv
    │   └── combined_einasto_decay_allSources_t600_wsj_quantiles.csv
    └── single
        ├── CBe_einasto_decay_t100_wsj_quantiles.csv
        ├── DraI_einasto_decay_t100_wsj_quantiles.csv
        ├── RetII_einasto_decay_t100_wsj_quantiles.csv
        ├── Scl_einasto_decay_t100_wsj_quantiles.csv
        ├── SgrII_einasto_decay_t100_wsj_quantiles.csv
        ├── UMaII_einasto_decay_t100_wsj_quantiles.csv
        ├── UMi_einasto_decay_t100_wsj_quantiles.csv
        └── Wil1_einasto_decay_t100_wsj_quantiles.csv
```
