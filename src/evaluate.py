"""Pooled metrics with site-level (cluster) bootstrap confidence intervals, overall and by slice."""
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0
UNHEALTHY_PM25 = 35.0  # EPA 24-hour PM2.5 standard, ug/m3
N_BOOT = 1000
CI_LOW, CI_HIGH = 2.5, 97.5
ISOLATION_BINS_KM = [0, 10, 25, 50, 100, np.inf]
ISOLATION_LABELS = ["<10 km", "10-25 km", "25-50 km", "50-100 km", ">100 km"]
N_QUINTILES = 5
QUINTILE_LABELS = [f"Q{i}" for i in range(1, N_QUINTILES + 1)]


def site_sums(sites, y, pred, lo=None, hi=None):
    """Per-site sufficient statistics, so the bootstrap can resample whole sites cheaply."""
    err = pred - y
    unhealthy = y > UNHEALTHY_PM25
    columns = {
        "site_id": sites, "n": 1.0, "sy": y, "syy": y * y, "sse": err * err, "sae": np.abs(err),
        "n_unhealthy": unhealthy.astype(float),
        "n_unhealthy_caught": (unhealthy & (pred > UNHEALTHY_PM25)).astype(float),
    }
    if lo is not None and hi is not None:
        covered = (y >= lo) & (y <= hi)
        columns |= {
            "n_covered": covered.astype(float),
            "n_unhealthy_covered": (covered & unhealthy).astype(float),
            "width_sum": hi - lo,
        }
    frame = pd.DataFrame(columns)
    return frame.groupby("site_id")[[c for c in frame.columns if c != "site_id"]].sum()


def _metrics(matrix, columns):
    """matrix: (..., k) totals in `columns` order -> dict of metric arrays."""
    col = {name: matrix[..., i] for i, name in enumerate(columns)}
    n, sy = col["n"], col["sy"]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {
            "r2": 1.0 - col["sse"] / (col["syy"] - sy * sy / n),
            "rmse": np.sqrt(col["sse"] / n),
            "nmae": col["sae"] / sy,
            "unhealthy_recall": col["n_unhealthy_caught"] / col["n_unhealthy"],
        }
        if "n_covered" in col:
            out |= {
                "coverage90": col["n_covered"] / n,
                "coverage90_unhealthy": col["n_unhealthy_covered"] / col["n_unhealthy"],
                "mean_interval_width": col["width_sum"] / n,
            }
    return out


def summarize(sums_by_site, seed=0):
    """Point estimate plus 95% CI from resampling sites with replacement."""
    columns = list(sums_by_site.columns)
    matrix = sums_by_site.to_numpy()
    point = _metrics(matrix.sum(axis=0), columns)
    rng = np.random.default_rng(seed)
    n_sites = len(matrix)
    weights = rng.multinomial(n_sites, np.full(n_sites, 1.0 / n_sites), size=N_BOOT)
    boot = _metrics(weights @ matrix, columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # metric undefined in a slice (e.g. no unhealthy days)
        summary = {
            name: {"value": float(point[name]),
                   "ci": [float(np.nanpercentile(boot[name], CI_LOW)),
                          float(np.nanpercentile(boot[name], CI_HIGH))]}
            for name in point
        }
    return summary | {"n_sites": int(n_sites), "n_site_days": int(matrix[:, columns.index("n")].sum())}


def site_isolation_km(site_table):
    """Distance from each site to the nearest other monitor in the full network."""
    coords = np.radians(site_table[["lat", "lon"]].to_numpy())
    dist, _ = BallTree(coords, metric="haversine").query(coords, k=2)
    return dist[:, 1] * EARTH_RADIUS_KM


def add_site_slices(site_table):
    """Isolation bin and county income / poverty quintiles for each site."""
    isolation = site_isolation_km(site_table)
    return site_table.assign(
        isolation_km=isolation,
        isolation_bin=pd.cut(isolation, ISOLATION_BINS_KM, labels=ISOLATION_LABELS, right=False),
        income_quintile=pd.qcut(site_table["median_income"], N_QUINTILES, labels=QUINTILE_LABELS),
        poverty_quintile=pd.qcut(site_table["poverty_pct"], N_QUINTILES, labels=QUINTILE_LABELS),
    )


def summarize_by_slice(sums_by_site, site_table, slice_column):
    joined = sums_by_site.join(site_table[[slice_column]])
    return {
        str(level): summarize(group[list(sums_by_site.columns)])
        for level, group in joined.groupby(slice_column, observed=True)
    }
