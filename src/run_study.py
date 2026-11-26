"""Full study: does richer data close the gap between random and spatial validation of PM2.5 models?

Usage (from the project root): python src/run_study.py
Raw inputs are expected in data/raw/ (see README for sources).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from covariates import (MET_COLUMNS, county_population_density, daily_weather, fetch_elevation,
                        load_gsod_conus)
from data import load_county_income, load_pm25_site_days
from evaluate import ISOLATION_LABELS, add_site_slices, site_sums, summarize, summarize_by_slice
from features import (BASE_FEATURES, N_FOLDS, NEIGHBOR_FEATURES, build_neighbor_features,
                      group_folds, random_row_folds, spatial_block_ids)
from models import MODELS, fit_predict_lgbm, impute_with_train_medians
from plots import plot_site_map
from plots_study import plot_coverage, plot_distance_curve, plot_importance, plot_two_metric_panels

ROOT = Path(__file__).resolve().parent.parent
RAW, INTERIM = ROOT / "data" / "raw", ROOT / "data" / "interim"
RESULTS, FIGURES = ROOT / "results" / "study", ROOT / "figures" / "study"
STATIC_COVARIATES = ["elevation_m", "log10_pop_density"]
COVARIATES = MET_COLUMNS + ["met_station_km"] + STATIC_COVARIATES
FEATURE_SETS = {"neighbors": BASE_FEATURES + NEIGHBOR_FEATURES,
                "full": BASE_FEATURES + NEIGHBOR_FEATURES + COVARIATES}
MAIN_BLOCKS = 40
BLOCK_SWEEP = [10, 20, 80, 160]
EXTRA_SEEDS = [1, 2]
SLICES = ["isolation_bin", "income_quintile", "poverty_quintile"]
UTAH_FIPS = "49"
IMPORTANCE_ROWS = 20000
IMPORTANCE_REPEATS = 5
SCHEME_LABELS = {"random": "Random K-fold", "sites": "Leave sites out",
                 "regions": f"Leave regions out ({MAIN_BLOCKS})"}


def load_inputs():
    df = load_pm25_site_days(RAW / "daily_88101_2024.zip")
    sites = df.groupby("site_id").agg(lat=("lat", "first"), lon=("lon", "first"),
                                      state=("state", "first"), county_fips=("county_fips", "first"))
    sites = sites.join(load_county_income(RAW / "est23all.txt").set_index("county_fips"), on="county_fips")
    density = county_population_density(RAW / "co-est2023-alldata.csv", RAW / "2023_Gaz_counties_national.zip")
    sites = add_site_slices(sites.assign(
        elevation_m=fetch_elevation(sites, INTERIM / "site_elevation.csv"),
        log10_pop_density=sites["county_fips"].map(density),
    ))
    gsod = load_gsod_conus(RAW / "gsod_2024.tar.gz", INTERIM / "gsod_2024_conus.parquet")
    weather = daily_weather(df, sites, gsod)
    static = sites[STATIC_COVARIATES].reindex(df["site_id"]).to_numpy()
    df = pd.concat([df, weather, pd.DataFrame(static, columns=STATIC_COVARIATES, index=df.index)], axis=1)
    return df, sites


def block_folds(df, sites, n_blocks, seed=0):
    blocks = spatial_block_ids(sites["lat"].to_numpy(), sites["lon"].to_numpy(), n_blocks, seed)
    return group_folds(df["site_id"].map(dict(zip(sites.index, blocks))).to_numpy(), seed)


def _drop_empty_bounds(output):
    if np.isnan(output["lo"]).all():
        return {"pred": output["pred"], "lo": None, "hi": None}
    return output


def run_cv(df, fold_ids, day_positions, combos):
    """Out-of-fold outputs for each (model, feature set) combo; neighbors rebuilt per fold."""
    coords, site_ids, y = df[["lat", "lon"]].to_numpy(), df["site_id"].to_numpy(), df["pm25"].to_numpy()
    base, cov = df[BASE_FEATURES].to_numpy(float), df[COVARIATES].to_numpy(float)
    outputs = {c: {k: np.full(len(df), np.nan) for k in ("pred", "lo", "hi")} for c in combos}
    nn_dist = np.full(len(df), np.nan)
    for fold in range(N_FOLDS):
        test = fold_ids == fold
        neighbors = build_neighbor_features(coords, site_ids, y, day_positions, ~test)
        nn_dist[test] = neighbors[test, NEIGHBOR_FEATURES.index("nn1_dist_km")]
        matrices = {"neighbors": np.hstack([base, neighbors])}
        matrices["full"] = np.hstack([matrices["neighbors"], cov])
        for model_name, feature_set in combos:
            X_train, X_test = impute_with_train_medians(matrices[feature_set][~test], matrices[feature_set][test])
            out, _ = MODELS[model_name](X_train, y[~test], X_test)
            for key, values in out.items():
                outputs[(model_name, feature_set)][key][test] = values
        print(f"    fold {fold + 1}/{N_FOLDS}", flush=True)
    return {c: _drop_empty_bounds(o) for c, o in outputs.items()}, nn_dist


def evaluate_outputs(df, sites, out, with_slices=True):
    sums = site_sums(df["site_id"].to_numpy(), df["pm25"].to_numpy(), out["pred"], out["lo"], out["hi"])
    result = {"overall": summarize(sums)}
    if with_slices:
        utah = sums.index.intersection(sites.index[sites["state"] == UTAH_FIPS])
        result |= {"utah": summarize(sums.loc[utah])}
        result |= {col: summarize_by_slice(sums, sites, col) for col in SLICES}
    return result, sums


def main_comparison(df, sites, day_positions):
    """E1: 3 validation schemes x 2 models x 2 feature sets."""
    schemes = {"random": random_row_folds(len(df)), "sites": group_folds(df["site_id"].to_numpy()),
               "regions": block_folds(df, sites, MAIN_BLOCKS)}
    combos = [(m, f) for m in MODELS for f in FEATURE_SETS]
    results, sums_store, distances = {}, {}, {}
    for scheme, folds in schemes.items():
        print(f"E1 scheme: {scheme}", flush=True)
        outputs, nn_dist = run_cv(df, folds, day_positions, combos)
        distances[scheme] = float(np.nanmedian(nn_dist))
        for model_name, feature_set in combos:
            key = f"{scheme}|{model_name}|{feature_set}"
            results[key], sums_store[key] = evaluate_outputs(df, sites, outputs[(model_name, feature_set)])
    return results, sums_store, distances


def block_sweep(df, sites, day_positions):
    """E2: how skill decays as held-out regions grow (QRF, full features)."""
    points = []
    for n_blocks in BLOCK_SWEEP:
        print(f"E2 blocks: {n_blocks}", flush=True)
        outputs, nn_dist = run_cv(df, block_folds(df, sites, n_blocks), day_positions, [("QRF", "full")])
        summary, _ = evaluate_outputs(df, sites, outputs[("QRF", "full")], with_slices=False)
        points.append({"label": f"{n_blocks} regions", "median_dist_km": float(np.nanmedian(nn_dist)),
                       "summary": summary["overall"]})
    return points


def seed_robustness(df, sites, day_positions):
    """E5: leave-regions-out with different block and fold seeds (QRF, full features)."""
    rows = []
    for seed in EXTRA_SEEDS:
        print(f"E5 seed: {seed}", flush=True)
        outputs, _ = run_cv(df, block_folds(df, sites, MAIN_BLOCKS, seed), day_positions, [("QRF", "full")])
        summary, _ = evaluate_outputs(df, sites, outputs[("QRF", "full")], with_slices=False)
        rows.append({"seed": seed, **{m: summary["overall"][m]["value"]
                                      for m in ("r2", "unhealthy_recall", "coverage90")}})
    return rows


def importance_in_unseen_region(df, sites, day_positions):
    """E4: permutation importance of LightGBM (full features) on one held-out-regions fold."""
    test = block_folds(df, sites, MAIN_BLOCKS) == 0
    coords, site_ids, y = df[["lat", "lon"]].to_numpy(), df["site_id"].to_numpy(), df["pm25"].to_numpy()
    neighbors = build_neighbor_features(coords, site_ids, y, day_positions, ~test)
    X = np.hstack([df[BASE_FEATURES].to_numpy(float), neighbors, df[COVARIATES].to_numpy(float)])
    X_train, X_test = impute_with_train_medians(X[~test], X[test])
    _, model = fit_predict_lgbm(X_train, y[~test], X_test)
    rows = np.random.default_rng(0).choice(len(X_test), size=min(IMPORTANCE_ROWS, len(X_test)), replace=False)
    result = permutation_importance(model, X_test[rows], y[test][rows], scoring="r2",
                                    n_repeats=IMPORTANCE_REPEATS, random_state=0, n_jobs=1)
    return pd.DataFrame({"mean": result.importances_mean, "std": result.importances_std},
                        index=FEATURE_SETS["full"])


def make_figures(e1, sums_e1, sweep_points, importance, sites):
    def pick(model, fset):
        return {SCHEME_LABELS[s]: e1[f"{s}|{model}|{fset}"]["overall"] for s in SCHEME_LABELS}
    labels = list(SCHEME_LABELS.values())
    plot_two_metric_panels({"Quantile random forest": pick("QRF", "full"), "LightGBM": pick("LightGBM", "full")},
                           labels, "Same models, three ways to validate them (full features)",
                           FIGURES / "fig1_models_by_scheme.png")
    plot_two_metric_panels({"Neighbors only": pick("QRF", "neighbors"),
                            "Neighbors + weather & land": pick("QRF", "full")},
                           labels, "Do richer covariates close the gap? (quantile random forest)",
                           FIGURES / "fig2_features_by_scheme.png")
    plot_distance_curve(sweep_points, FIGURES / "fig3_distance_curve.png")
    iso = e1["regions|QRF|full"]["isolation_bin"]
    plot_coverage(pick("QRF", "full"), {b: iso[b] for b in ISOLATION_LABELS if b in iso},
                  FIGURES / "fig4_interval_coverage.png")
    per_site = sites.join(sums_e1["regions|QRF|full"]).assign(nmae=lambda t: t["sae"] / t["sy"])
    plot_site_map(per_site, "nmae", FIGURES / "fig5_site_map_regions.png")
    plot_importance(importance, FIGURES / "fig6_importance.png")
    return per_site


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    start = time.time()
    df, sites = load_inputs()
    missing = df[COVARIATES].isna().mean().round(3).to_dict()
    print(f"{len(df):,} site-days, {len(sites):,} sites; covariate missing share: {missing}", flush=True)
    day_positions = list(df.groupby("date").indices.values())

    e1, sums_e1, distances = main_comparison(df, sites, day_positions)
    sweep = block_sweep(df, sites, day_positions)
    sweep += [{"label": SCHEME_LABELS[s], "median_dist_km": distances[s],
               "summary": e1[f"{s}|QRF|full"]["overall"]} for s in ("sites", "regions")]
    seeds = seed_robustness(df, sites, day_positions)
    importance = importance_in_unseen_region(df, sites, day_positions)

    output = {"main": e1, "median_dist_km": distances, "block_sweep": sweep, "seed_robustness": seeds,
              "covariate_missing_share": missing}
    (RESULTS / "study_metrics.json").write_text(json.dumps(output, indent=2))
    importance.to_csv(RESULTS / "importance_regions_fold0.csv")
    make_figures(e1, sums_e1, sweep, importance, sites).to_csv(RESULTS / "site_errors_regions_qrf.csv")
    for key, res in e1.items():
        o = res["overall"]
        cov = o.get("coverage90", {}).get("value", float("nan"))
        print(f"{key:28s} R2={o['r2']['value']:.3f} recall={o['unhealthy_recall']['value']:.3f} cov90={cov:.3f}")
    print(f"Done in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
