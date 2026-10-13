#!/usr/bin/env python3
# =============================================================================
#  engineered_features_v2.py
#  Physically motivated engineered layer for the 1,318-column superset.
#
#  WHAT IS NEW RELATIVE TO v1
#  ---------------------------------------------------------------------------
#  * Column names in the superset are lower-case and differently prefixed
#    (prism_tdmean_min, era5_2m_dew_point_temperature, cams_formaldehyde_x1e9),
#    so the whole layer is rewritten against the real schema.
#  * ERA5 already supplies wet-bulb temperature, relative and specific humidity,
#    and total column water vapour. We do NOT duplicate those. Instead we build
#    a PRISM-based wet-bulb that spans 2000-2023 (ERA5 stops in 2019), which
#    gives the composite index full coverage in both study windows and lets the
#    two independent wet-bulb estimates be cross-checked against each other.
#  * CAMS composition enables photochemistry indices that the atmospheric
#    literature treats as standard, above all the formaldehyde-to-nitrogen
#    dioxide ratio, which diagnoses whether ozone formation in a place is
#    limited by volatile organic compounds or by nitrogen oxides.
#
#  DESIGN RULES (unchanged)
#  * Row-local and deterministic: every feature is a function of same-row
#    inputs, so adding this layer before cross-validation introduces no leakage.
#  * Defensive: a feature is built only if all of its inputs exist, so the same
#    module runs unchanged on either study's column subset.
#  * Self-reporting: returns a coverage table so a reduced-coverage engineered
#    feature can never again be ranked without anyone noticing.
# =============================================================================

from __future__ import annotations
import numpy as np
import pandas as pd

TAG = "eng2"


# ---------------------------------------------------------------------------
#  physics helpers
# ---------------------------------------------------------------------------
def _c(df, name):
    """Column if present, else an all-NaN series (defensive access)."""
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


def _first(df, *names):
    """First present column among alternatives, else all-NaN."""
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(np.nan, index=df.index)


def sat_vapour_pressure_hpa(T):
    """Saturation vapour pressure over water (hPa), Tetens; T in degrees C."""
    return 6.1078 * np.exp(17.27 * T / (T + 237.3))


def rh_from_T_Td(T, Td):
    """Relative humidity (%) from temperature and dew point, both deg C."""
    a, b = 17.625, 243.04
    return np.clip(100.0 * np.exp((a * Td) / (b + Td)) / np.exp((a * T) / (b + T)),
                   1.0, 100.0)


def stull_wetbulb(T, RH):
    """Wet-bulb temperature (deg C). Stull, J Appl Meteorol Climatol, 2011.
    Valid roughly -20 to 50 deg C and 5 to 99 percent relative humidity."""
    RH = np.clip(RH, 1.0, 100.0)
    return (T * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
            + np.arctan(T + RH) - np.arctan(RH - 1.676331)
            + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH) - 4.686035)


def wbgt_shade(T, Td):
    """Wet-bulb globe temperature, shade approximation (Australian BoM)."""
    return 0.567 * T + 0.393 * sat_vapour_pressure_hpa(Td) + 3.94


def heat_index_c(T, RH):
    """US National Weather Service heat index (Rothfusz), returned in deg C."""
    Tf = T * 9 / 5 + 32
    HI = (-42.379 + 2.04901523 * Tf + 10.14333127 * RH - 0.22475541 * Tf * RH
          - 6.83783e-3 * Tf ** 2 - 5.481717e-2 * RH ** 2
          + 1.22874e-3 * Tf ** 2 * RH + 8.5282e-4 * Tf * RH ** 2
          - 1.99e-6 * Tf ** 2 * RH ** 2)
    HI = np.where(Tf < 80, Tf, HI)
    return (HI - 32) * 5 / 9


def humidex_c(T, Td):
    """Humidex (Masterton and Richardson), deg C."""
    e = 6.11 * np.exp(5417.7530 * (1 / 273.16 - 1 / (273.15 + Td)))
    return T + 0.5555 * (e - 10.0)


def _safe_ratio(num, den, eps=1e-9):
    den = pd.Series(den).replace(0, np.nan)
    return pd.Series(num) / (den + eps)


# ---------------------------------------------------------------------------
#  main entry point
# ---------------------------------------------------------------------------
def add_engineered_features(df: pd.DataFrame, verbose: bool = True):
    """Return (augmented_df, coverage_report). Adds features named eng2_*."""
    out = df.copy()
    new = {}

    def add(name, series, needs=()):
        """Register a feature only if every named input column exists."""
        if needs and not all(n in df.columns for n in needs):
            return
        new[f"{TAG}_{name}"] = series

    # ---- base fields -------------------------------------------------------
    # PRISM spans 2000-2023 and is the backbone for full-coverage indices.
    t_mean = _c(df, "prism_tmean_mean")
    t_max = _c(df, "prism_tmax_mean")
    t_min = _c(df, "prism_tmin_mean")
    td_mean = _c(df, "prism_tdmean_mean")
    td_min = _c(df, "prism_tdmean_min")
    ppt = _first(df, "prism_ppt_mean", "prism_ppt_sum_mean")

    # ERA5 spans 2003-2019 and supplies an independent thermodynamic view.
    e_t = _c(df, "era5_2m_temperature")
    e_td = _c(df, "era5_2m_dew_point_temperature")
    e_wb = _c(df, "era5_wet_bulb_temperature")
    e_q = _c(df, "era5_specific_humidity")
    e_tcwv = _c(df, "era5_total_column_water_vapour")
    e_sp = _c(df, "era5_surface_pressure")
    e_ws = _c(df, "era5_10m_wind_speed")

    lst_day = _c(df, "lst_day_1km_mean")
    lst_night = _c(df, "lst_night_1km_mean")

    # ---- 1. THERMAL COMFORT FROM PRISM (full 2000-2023 coverage) -----------
    # These matter because ERA5 stops in 2019; a PRISM-based composite keeps the
    # heat-and-humidity question answerable across the whole record.
    rh_mean = rh_from_T_Td(t_mean, td_mean)
    rh_hot = rh_from_T_Td(t_max, td_mean)
    rh_night = rh_from_T_Td(t_min, td_min)

    add("wetbulb_mean", stull_wetbulb(t_mean, rh_mean),
        needs=("prism_tmean_mean", "prism_tdmean_mean"))
    add("wetbulb_hot", stull_wetbulb(t_max, rh_hot),
        needs=("prism_tmax_mean", "prism_tdmean_mean"))
    add("wetbulb_night", stull_wetbulb(t_min, rh_night),
        needs=("prism_tmin_mean", "prism_tdmean_min"))
    add("wbgt_shade", wbgt_shade(t_mean, td_mean),
        needs=("prism_tmean_mean", "prism_tdmean_mean"))
    add("wbgt_hot", wbgt_shade(t_max, td_mean),
        needs=("prism_tmax_mean", "prism_tdmean_mean"))
    add("heat_index_hot", heat_index_c(t_max, rh_hot),
        needs=("prism_tmax_mean", "prism_tdmean_mean"))
    add("humidex_hot", humidex_c(t_max, td_mean),
        needs=("prism_tmax_mean", "prism_tdmean_mean"))
    add("discomfort_index", t_mean - 0.55 * (1 - 0.01 * rh_mean) * (t_mean - 14.5),
        needs=("prism_tmean_mean", "prism_tdmean_mean"))
    add("rel_humidity_mean", rh_mean, needs=("prism_tmean_mean", "prism_tdmean_mean"))
    add("rel_humidity_night", rh_night, needs=("prism_tmin_mean", "prism_tdmean_min"))
    add("dewpoint_depression", t_mean - td_mean,
        needs=("prism_tmean_mean", "prism_tdmean_mean"))
    add("nocturnal_dewpoint_depression", t_min - td_min,
        needs=("prism_tmin_mean", "prism_tdmean_min"))
    add("diurnal_temp_range", t_max - t_min,
        needs=("prism_tmax_mean", "prism_tmin_mean"))
    add("humid_heat_excess", np.clip(stull_wetbulb(t_max, rh_hot) - 26.0, 0, None),
        needs=("prism_tmax_mean", "prism_tdmean_mean"))
    add("cooling_degree_days", np.clip(t_mean - 18.0, 0, None),
        needs=("prism_tmean_mean",))
    add("heating_degree_days", np.clip(18.0 - t_mean, 0, None),
        needs=("prism_tmean_mean",))

    e_v = sat_vapour_pressure_hpa(td_mean)
    add("vapour_pressure", e_v, needs=("prism_tdmean_mean",))
    add("vpd_derived", sat_vapour_pressure_hpa(t_mean) - e_v,
        needs=("prism_tmean_mean", "prism_tdmean_mean"))
    add("specific_humidity_prism", 0.622 * e_v / (1013.25 - 0.378 * e_v),
        needs=("prism_tdmean_mean",))
    add("absolute_humidity_prism", (2.16679 * (e_v * 100.0)) / (273.15 + t_mean),
        needs=("prism_tmean_mean", "prism_tdmean_mean"))

    # ---- 2. ERA5 CROSS-CHECK AND DERIVED THERMODYNAMICS --------------------
    # A direct comparison of two independent wet-bulb estimates. Large
    # disagreement in a county is a data-quality signal worth having in hand.
    add("wetbulb_era5_minus_prism", e_wb - stull_wetbulb(t_mean, rh_mean),
        needs=("era5_wet_bulb_temperature", "prism_tmean_mean", "prism_tdmean_mean"))
    add("era5_dewpoint_depression", e_t - e_td,
        needs=("era5_2m_temperature", "era5_2m_dew_point_temperature"))
    add("era5_wetbulb_depression", e_t - e_wb,
        needs=("era5_2m_temperature", "era5_wet_bulb_temperature"))
    # Column water vapour per unit surface pressure: a moisture-loading index.
    add("column_moisture_loading", _safe_ratio(e_tcwv, e_sp),
        needs=("era5_total_column_water_vapour", "era5_surface_pressure"))
    add("ventilation_index", e_ws * np.clip(e_t - e_td, 0.1, None),
        needs=("era5_10m_wind_speed", "era5_2m_temperature",
               "era5_2m_dew_point_temperature"))

    # ---- 3. CAMS PHOTOCHEMISTRY -------------------------------------------
    # The formaldehyde-to-nitrogen-dioxide ratio is the standard satellite
    # diagnostic of the ozone-production regime: low values indicate formation
    # limited by volatile organic compounds, high values indicate formation
    # limited by nitrogen oxides. It is a genuinely different quantity from
    # either species alone, which is exactly what a fusion model needs.
    hcho = _first(df, "cams_formaldehyde_x1e9", "cams_total_column_formaldehyde_x1e9")
    no2 = _first(df, "cams_nitrogen_dioxide_x1e9", "cams_total_column_nitrogen_dioxide_x1e9")
    o3 = _first(df, "cams_ozone_x1e9", "cams_total_column_ozone")
    co = _first(df, "cams_carbon_monoxide_x1e9", "cams_total_column_carbon_monoxide")
    so2 = _first(df, "cams_sulphur_dioxide_x1e9", "cams_total_column_sulphur_dioxide_x1e9")
    oh = _first(df, "cams_hydroxyl_radical_x1e9", "cams_total_column_hydroxyl_radical_x1e9")
    isop = _first(df, "cams_isoprene_x1e9", "cams_total_column_isoprene_x1e9")
    pan = _first(df, "cams_peroxyacetyl_nitrate_x1e9",
                 "cams_total_column_peroxyacetyl_nitrate_x1e9")
    h2o2 = _first(df, "cams_hydrogen_peroxide_x1e9",
                  "cams_total_column_hydrogen_peroxide_x1e9")
    bc_aod = _c(df, "cams_black_carbon_aod_at_550_nm")
    om_aod = _c(df, "cams_organic_matter_aod_at_550_nm")
    su_aod = _c(df, "cams_sulphate_aod_at_550_nm")
    du_aod = _c(df, "cams_dust_aod_at_550_nm")
    ss_aod = _c(df, "cams_sea_salt_aod_at_550_nm")
    tot_aod = _c(df, "cams_total_aod_at_550_nm")
    pm25c = _first(df, "cams_pm2_5_x1e9", "pm25_mean")
    pm10c = _c(df, "cams_pm10_x1e9")

    if "cams_formaldehyde_x1e9" in df.columns or \
       "cams_total_column_formaldehyde_x1e9" in df.columns:
        add("hcho_no2_ratio", _safe_ratio(hcho, no2))
        add("oxidant_load", pd.Series(o3) + pd.Series(no2))
        add("photochem_activity", _safe_ratio(pd.Series(hcho) * pd.Series(oh), 1.0))
        add("biogenic_fraction", _safe_ratio(isop, pd.Series(isop) + pd.Series(hcho)))
        add("nox_reservoir_ratio", _safe_ratio(pan, no2))
        add("radical_termination", _safe_ratio(h2o2, pd.Series(h2o2) + pd.Series(no2)))
        add("combustion_tracer_ratio", _safe_ratio(co, no2))
        add("acid_precursor_load", pd.Series(so2) + pd.Series(no2))

    # Aerosol composition: what the particulate burden is made of, not just
    # how much there is. Anthropogenic fraction is the health-relevant part.
    anthro = pd.Series(bc_aod).fillna(0) + pd.Series(om_aod).fillna(0) + \
             pd.Series(su_aod).fillna(0)
    natural = pd.Series(du_aod).fillna(0) + pd.Series(ss_aod).fillna(0)
    if "cams_total_aod_at_550_nm" in df.columns:
        add("anthropogenic_aod_fraction", _safe_ratio(anthro, pd.Series(tot_aod)))
        add("natural_aod_fraction", _safe_ratio(natural, pd.Series(tot_aod)))
        add("anthro_natural_aod_ratio", _safe_ratio(anthro, natural))
    if "cams_black_carbon_aod_at_550_nm" in df.columns:
        add("black_carbon_share", _safe_ratio(bc_aod, anthro))
    if "cams_pm10_x1e9" in df.columns:
        add("fine_coarse_pm_ratio", _safe_ratio(pm25c, pm10c))

    # Chronic exposure interaction: fine particulate load carried in humid air.
    # Hygroscopic growth in humid conditions increases deposited dose.
    add("humid_particulate_load", pd.Series(pm25c) * (rh_mean / 100.0),
        needs=("prism_tmean_mean", "prism_tdmean_mean"))

    # ---- 4. SURFACE THERMAL BEHAVIOUR -------------------------------------
    add("day_night_lst_range", lst_day - lst_night,
        needs=("lst_day_1km_mean", "lst_night_1km_mean"))
    add("night_cooling_efficiency", _safe_ratio(lst_day - lst_night, lst_day),
        needs=("lst_day_1km_mean", "lst_night_1km_mean"))
    add("surface_air_night_offset", lst_night - 273.15 - t_min,
        needs=("lst_night_1km_mean", "prism_tmin_mean"))

    ndvi = _first(df, "ndvi_ndvi_mean_div10k", "ndvi_evi_mean_div10k",
                  "landsat_ndvi_mean", "ls57_ndvi_mean")
    add("vegetation_heat_buffer", pd.Series(ndvi) * (lst_day - lst_night),
        needs=("lst_day_1km_mean", "lst_night_1km_mean"))
    ghm = _first(df, "ghm_mean", "ghm_p50")
    add("impervious_heat_index", pd.Series(ghm) * (lst_day - 273.15),
        needs=("lst_day_1km_mean",))
    add("urban_moisture_trap", pd.Series(ghm) * pd.Series(td_min),
        needs=("prism_tdmean_min",))

    # ---- 5. WATER BALANCE --------------------------------------------------
    pet = _first(df, "terra_pet_mean", "terra_pet_p50")
    aet = _first(df, "terra_aet_mean", "terra_aet_p50")
    add("aridity_index", _safe_ratio(ppt, pet))
    add("evaporative_fraction", _safe_ratio(aet, pet))

    # ---- assemble and report ----------------------------------------------
    rows = []
    year = _c(df, "year")
    for name, series in new.items():
        s = pd.to_numeric(pd.Series(np.asarray(series, dtype="float64"),
                                    index=df.index), errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan)
        n = int(s.notna().sum())
        if n == 0:
            continue
        out[name] = s
        yy = year[s.notna()]
        rows.append(dict(feature=name, n_observed=n,
                         pct_coverage=round(100 * n / len(df), 1),
                         year_min=int(yy.min()) if len(yy) else None,
                         year_max=int(yy.max()) if len(yy) else None,
                         n_unique=int(s.nunique())))

    report = pd.DataFrame(rows).sort_values("pct_coverage", ascending=False)
    if verbose and len(report):
        full = (report.pct_coverage >= 99).sum()
        print(f"[engineered] added {len(report)} features "
              f"({full} at full coverage, {len(report)-full} partial)")
        partial = report[report.pct_coverage < 99]
        for _, r in partial.iterrows():
            print(f"    partial: {r.feature:<44} {r.pct_coverage:5.1f}%  "
                  f"{r.year_min}-{r.year_max}")
    return out, report


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "Final_ML_Ready_Dataset.csv"
    d = pd.read_csv(src, low_memory=False)
    d.columns = [c.strip().lower().replace(" ", "_") for c in d.columns]
    aug, rep = add_engineered_features(d)
    rep.to_csv("engineered_v2_coverage.csv", index=False)
    print(f"\naugmented: {aug.shape[0]:,} x {aug.shape[1]:,}")
