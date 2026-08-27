# Data

The input data for this study are deposited on Zenodo rather than committed to
git, because the assembled panel is large and because a versioned archive with a
DOI is the appropriate citable home for it.

**Deposit:** [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

## What the deposit contains

| File | Description |
|---|---|
| `Final_ML_Ready_Dataset.csv` | The assembled county-year panel, all candidate columns before gating |
| `studyA_reference_frame.parquet` | The exact frame the model consumed, after the engineered layer was added |
| `studyA_final_model.joblib` | Fitted model, feature list, and the imputation values used |
| `tables/` | Every result table cited in the paper |
| `figures/` | Every figure, PNG at 300 dpi and vector PDF |

The reference frame is included deliberately. It lets a reviewer verify any
intermediate quantity without re-running the pipeline, and it is the single
artefact that makes the analysis fully auditable.

## How to use it

Download the deposit and unpack it into this directory:

```bash
cd data
# download from the Zenodo link above, then
unzip how-long-not-how-much-data-v1.0.0.zip
```

Then follow [`../docs/REPRODUCE.md`](../docs/REPRODUCE.md).

## Primary sources

The panel is assembled from publicly available sources. We redistribute the
derived county-year aggregates, not the original products. If you need the
originals, go to the providers directly.

| Source | Role | Link |
|---|---|---|
| IHME county life expectancy | Outcome variable, county mean life expectancy at birth | [https://ghdx.healthdata.org/](https://ghdx.healthdata.org/) |
| CAMS global reanalysis | Atmospheric composition: reactive gases and speciated aerosol | [https://ads.atmosphere.copernicus.eu](https://ads.atmosphere.copernicus.eu) |
| ERA5 reanalysis | Atmospheric state: temperature, humidity, wind, pressure | [https://cds.climate.copernicus.eu](https://cds.climate.copernicus.eu) |
| PRISM climate | Interpolated US surface temperature, dew point, precipitation | [https://prism.oregonstate.edu](https://prism.oregonstate.edu) |
| TerraClimate | Monthly water balance and surface climate | [https://www.climatologylab.org/terraclimate.html](https://www.climatologylab.org/terraclimate.html) |
| MODIS via Google Earth Engine | Land-surface temperature and vegetation indices | [https://earthengine.google.com](https://earthengine.google.com) |
| Landsat via Google Earth Engine | Surface reflectance and derived spectral indices | [https://earthengine.google.com](https://earthengine.google.com) |
| Global Human Modification | Cumulative landscape transformation index | [https://doi.org/10.1111/gcb.14549](https://doi.org/10.1111/gcb.14549) |
| SoilGrids 2.0 | Soil bulk density, texture, pH, organic carbon | [https://soilgrids.org](https://soilgrids.org) |
| Harmonised night-time lights | DMSP and VIIRS reconciled long series | [https://doi.org/10.1038/s41597-020-0510-y](https://doi.org/10.1038/s41597-020-0510-y) |
| County Health Rankings | Social determinants of health (Study B only) | [https://www.countyhealthrankings.org](https://www.countyhealthrankings.org) |
| American Community Survey | Demographics, income, education, housing (Study B only) | [https://www.census.gov/programs-surveys/acs](https://www.census.gov/programs-surveys/acs) |

## License

Derived data in the deposit are released under CC-BY-4.0. The original
products remain under their providers' licenses, which you should check before
redistributing anything derived from them.

## Citing the data

Cite the Zenodo deposit alongside the paper:

```bibtex
@dataset{Ahmad2026DataPaper1,
  author    = {Ahmad, Faiz and Lary, David J. and Ruwali, Shisir and Shrestha, Samyak and Aker, Adam and Waczak, John and Madushanka, Prabuddha},
  title     = {Data for: How Long, Not How Much: Exposure Duration Outranks Concentration in an Eighteen-Modality Satellite and Reanalysis Fusion for United States County Life Expectancy},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.XXXXXXX}
}
```
