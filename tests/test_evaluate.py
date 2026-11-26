import numpy as np
import pandas as pd

from evaluate import site_isolation_km, site_sums, summarize


def test_perfect_predictions_score_perfectly():
    y = np.array([5.0, 10.0, 40.0, 50.0, 8.0, 60.0])
    sites = np.array(["a", "a", "b", "b", "c", "c"])
    summary = summarize(site_sums(sites, y, y.copy()))
    assert summary["r2"]["value"] == 1.0
    assert summary["nmae"]["value"] == 0.0
    assert summary["unhealthy_recall"]["value"] == 1.0
    assert summary["n_sites"] == 3 and summary["n_site_days"] == 6


def test_interval_coverage_counts_rows_inside_bounds():
    y = np.array([1.0, 2.0, 3.0, 40.0])
    pred = y.copy()
    lo = np.array([0.0, 2.5, 2.0, 30.0])
    hi = np.array([2.0, 3.0, 4.0, 50.0])
    summary = summarize(site_sums(np.array(["s"] * 4), y, pred, lo, hi))
    assert summary["coverage90"]["value"] == 0.75
    assert summary["coverage90_unhealthy"]["value"] == 1.0
    assert summary["mean_interval_width"]["value"] == np.mean(hi - lo)


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.gamma(2.0, 4.0, size=500)
    pred = y + rng.normal(0, 2.0, size=500)
    sites = np.repeat(np.arange(50).astype(str), 10)
    r2 = summarize(site_sums(sites, y, pred))["r2"]
    assert r2["ci"][0] <= r2["value"] <= r2["ci"][1]


def test_isolation_is_distance_to_nearest_other_site():
    table = pd.DataFrame({"lat": [40.0, 41.0, 45.0], "lon": [-100.0, -100.0, -100.0]})
    km = site_isolation_km(table)
    assert abs(km[0] - 111.2) < 1.0  # one degree of latitude
    assert abs(km[2] - 4 * 111.2) < 3.0
