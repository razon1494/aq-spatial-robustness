"""Regenerate the study figures from saved results, without re-running the models.

Usage (from the project root): python src/replot_study.py
"""
import json

import pandas as pd

from run_study import FIGURES, RESULTS, make_figures

SUM_COLUMNS = ["n", "sy", "syy", "sse", "sae", "n_unhealthy", "n_unhealthy_caught",
               "n_covered", "n_unhealthy_covered", "width_sum"]


def main():
    metrics = json.loads((RESULTS / "study_metrics.json").read_text())
    importance = pd.read_csv(RESULTS / "importance_regions_fold0.csv", index_col=0)
    per_site = pd.read_csv(RESULTS / "site_errors_regions_qrf.csv", index_col="site_id",
                           dtype={"site_id": str, "state": str, "county_fips": str})
    sums = per_site[[c for c in SUM_COLUMNS if c in per_site.columns]]
    sites = per_site.drop(columns=list(sums.columns) + ["nmae"])
    FIGURES.mkdir(parents=True, exist_ok=True)
    make_figures(metrics["main"], {"regions|QRF|full": sums}, metrics["block_sweep"], importance, sites)
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
