import numpy as np

from features import build_neighbor_features, group_folds, random_row_folds, N_FOLDS


def _one_day(values):
    coords = np.array([[40.0, -111.0], [40.1, -111.0], [40.2, -111.0], [41.0, -111.0]])
    sites = np.array(["A", "B", "C", "D"])
    return coords, sites, np.asarray(values, dtype=float), [np.arange(4)]


def test_neighbor_features_never_use_own_site():
    coords, sites, values, days = _one_day([100.0, 1.0, 2.0, 3.0])
    features = build_neighbor_features(coords, sites, values, days, np.ones(4, dtype=bool))
    nearest_value_for_a = features[0, 1]
    assert nearest_value_for_a == 1.0  # B, not A's own 100


def test_neighbor_features_ignore_test_rows():
    coords, sites, values, days = _one_day([5.0, 1.0, 2.0, 3.0])
    is_train = np.array([True, False, True, True])
    before = build_neighbor_features(coords, sites, values, days, is_train)
    changed = values.copy()
    changed[1] = 999.0  # perturb the held-out site's value
    after = build_neighbor_features(coords, sites, changed, days, is_train)
    np.testing.assert_allclose(before, after)


def test_group_folds_keep_each_group_in_one_fold():
    groups = np.repeat(np.arange(20), 7)
    folds = group_folds(groups, seed=3)
    for g in np.unique(groups):
        assert len(np.unique(folds[groups == g])) == 1
    assert set(np.unique(folds)) == set(range(N_FOLDS))


def test_random_row_folds_are_balanced():
    folds = random_row_folds(1000, seed=1)
    counts = np.bincount(folds)
    assert counts.min() == counts.max() == 200
