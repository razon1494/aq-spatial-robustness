"""Pilot: how much does random cross-validation overstate PM2.5 prediction skill at new locations?

Usage (from the project root): python src/run_pilot.py data/raw/daily_88101_2024.zip data/raw/est23all.txt
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from data import load_county_income, load_pm25_site_days
from evaluate import add_site_slices, site_sums, summarize, summarize_by_slice, ISOLATION_LABELS
from features import cross_validate, group_folds, random_row_folds, spatial_block_ids
from plots import plot_cv_gap, plot_site_map, plot_slice

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "pilot"
FIGURES = ROOT / "figures" / "pilot"
SCHEME_RANDOM = "Random K-fold (rows)"
SCHEME_SITES = "Leave sites out"
SCHEME_BLOCKS = "Leave regions out"
SLICE_COLUMNS = ["isolation_bin", "income_quintile", "poverty_quintile"]
UTAH_FIPS = "49"


def site_table_for(df, income):
    sites = df.groupby("site_id").agg(lat=("lat", "first"), lon=("lon", "first"),
                                      state=("state", "first"), county_fips=("county_fips", "first"))
    return add_site_slices(sites.join(income.set_index("county_fips"), on="county_fips"))


def fold_schemes(df, site_table):
    blocks = spatial_block_ids(site_table["lat"].to_numpy(), site_table["lon"].to_numpy())
    block_of_site = dict(zip(site_table.index, blocks))
    return {
        SCHEME_RANDOM: random_row_folds(len(df)),
        SCHEME_SITES: group_folds(df["site_id"].to_numpy()),
        SCHEME_BLOCKS: group_folds(df["site_id"].map(block_of_site).to_numpy()),
    }


def evaluate_scheme(df, pred, site_table):
    sums = site_sums(df["site_id"].to_numpy(), df["pm25"].to_numpy(), pred)
    utah_sites = site_table.index[site_table["state"] == UTAH_FIPS]
    return {
        "overall": summarize(sums),
        "utah": summarize(sums.loc[sums.index.intersection(utah_sites)]),
        **{col: summarize_by_slice(sums, site_table, col) for col in SLICE_COLUMNS},
    }, sums


def make_figures(results, site_table, site_sums_new):
    plot_cv_gap(results, FIGURES / "fig1_cv_gap.png")
    pair = {"Random K-fold": results[SCHEME_RANDOM], "New location (leave site out)": results[SCHEME_SITES]}
    plot_slice({k: v["isolation_bin"] for k, v in pair.items()}, ISOLATION_LABELS,
               "Errors grow where monitors are sparse", "Distance from site to nearest other monitor",
               FIGURES / "fig2_isolation.png")
    quintiles = [f"Q{i}" for i in range(1, 6)]
    plot_slice({k: v["income_quintile"] for k, v in pair.items()}, quintiles,
               "Error by county median household income", "County income quintile (Q1 = lowest)",
               FIGURES / "fig3_income.png")
    per_site = site_table.join(site_sums_new).assign(nmae=lambda t: t["sae"] / t["sy"])
    plot_site_map(per_site, "nmae", FIGURES / "fig4_site_map.png")
    return per_site


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("epa_zip")
    parser.add_argument("saipe_txt")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    start = time.time()
    df = load_pm25_site_days(args.epa_zip)
    site_table = site_table_for(df, load_county_income(args.saipe_txt))
    print(f"{len(df):,} site-days, {len(site_table):,} sites", flush=True)
    day_positions = list(df.groupby("date").indices.values())

    results, preds, sums_by_scheme = {}, {}, {}
    for name, folds in fold_schemes(df, site_table).items():
        print(f"Scheme: {name}", flush=True)
        preds[name] = cross_validate(df, folds, day_positions)
        results[name], sums_by_scheme[name] = evaluate_scheme(df, preds[name], site_table)

    (RESULTS / "metrics.json").write_text(json.dumps(results, indent=2))
    per_site = make_figures(results, site_table, sums_by_scheme[SCHEME_SITES])
    per_site.to_csv(RESULTS / "site_errors_leave_site_out.csv")
    df.assign(**{f"pred_{i}": p for i, p in enumerate(preds.values())}).sample(
        frac=0.05, random_state=0).to_csv(RESULTS / "predictions_sample.csv", index=False)
    for name, res in results.items():
        o = res["overall"]
        print(f"{name:24s} R2={o['r2']['value']:.3f} {np.round(o['r2']['ci'], 3)} "
              f"nMAE={o['nmae']['value']:.3f} unhealthy_recall={o['unhealthy_recall']['value']:.3f}")
    print(f"Done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
