"""Same-day neighbor features, cross-validation fold schemes, and the RF model."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0
N_NEIGHBORS = 5
N_FOLDS = 5
N_SPATIAL_BLOCKS = 40
IDW_OFFSET_KM = 1.0
N_NEIGHBOR_FEATURES = 5
BASE_FEATURES = ["lat", "lon", "doy", "dow"]
NEIGHBOR_FEATURES = ["idw_pm25", "nn1_pm25", "knn_mean_pm25", "nn1_dist_km", "knn_mean_dist_km"]
RF_PARAMS = {
    "n_estimators": 150,
    "min_samples_leaf": 5,
    "max_features": 0.5,
    "max_samples": 0.5,
    "n_jobs": -1,
    "random_state": 0,
}


def _neighbor_stats(query_coords, query_sites, ref_coords, ref_sites, ref_values):
    """Neighbor summaries from reference monitors on the same day; a row never sees its own site."""
    k = min(N_NEIGHBORS + 1, len(ref_sites))
    tree = BallTree(np.radians(ref_coords), metric="haversine")
    dist, idx = tree.query(np.radians(query_coords), k=k)
    dist_km = np.where(ref_sites[idx] == query_sites[:, None], np.inf, dist * EARTH_RADIUS_KM)
    order = np.argsort(dist_km, axis=1)[:, :N_NEIGHBORS]
    d = np.take_along_axis(dist_km, order, axis=1)
    v = np.take_along_axis(ref_values[idx], order, axis=1)
    valid = np.isfinite(d)
    weights = np.where(valid, 1.0 / (np.where(valid, d, 0.0) + IDW_OFFSET_KM), 0.0)
    v_valid = np.where(valid, v, 0.0)
    n_valid = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        idw = (weights * v_valid).sum(axis=1) / weights.sum(axis=1)
        knn_mean = v_valid.sum(axis=1) / n_valid
        knn_dist = np.where(valid, d, 0.0).sum(axis=1) / n_valid
    nn1_val = np.where(valid[:, 0], v[:, 0], np.nan)
    nn1_dist = np.where(valid[:, 0], d[:, 0], np.nan)
    return np.column_stack([idw, nn1_val, knn_mean, nn1_dist, knn_dist])


def build_neighbor_features(coords, sites, values, day_positions, is_train):
    """Neighbor features for every row, computed only from training rows of the same day."""
    out = np.full((len(sites), N_NEIGHBOR_FEATURES), np.nan)
    for pos in day_positions:
        ref = pos[is_train[pos]]
        if len(ref) < 2:
            continue
        out[pos] = _neighbor_stats(coords[pos], sites[pos], coords[ref], sites[ref], values[ref])
    return out


def random_row_folds(n_rows, seed=0):
    rng = np.random.default_rng(seed)
    return rng.permutation(n_rows) % N_FOLDS


def group_folds(groups, seed=0):
    """Assign whole groups (sites or spatial blocks) to folds."""
    rng = np.random.default_rng(seed)
    unique = rng.permutation(np.unique(groups))
    fold_of = {g: i % N_FOLDS for i, g in enumerate(unique)}
    return np.array([fold_of[g] for g in groups])


def spatial_block_ids(lat, lon, n_blocks=N_SPATIAL_BLOCKS, seed=0):
    """K-means blocks on approximately equal-area coordinates (degrees scaled by cos(lat))."""
    xy = np.column_stack([lat, lon * np.cos(np.radians(lat))])
    return KMeans(n_clusters=n_blocks, n_init=10, random_state=seed).fit_predict(xy)


def cross_validate(df, fold_ids, day_positions):
    """Out-of-fold predictions; neighbor features are rebuilt per fold to prevent leakage."""
    coords = df[["lat", "lon"]].to_numpy()
    sites = df["site_id"].to_numpy()
    y = df["pm25"].to_numpy()
    base = df[BASE_FEATURES].to_numpy(dtype=float)
    pred = np.full(len(df), np.nan)
    for fold in range(N_FOLDS):
        is_test = fold_ids == fold
        is_train = ~is_test
        neighbors = build_neighbor_features(coords, sites, y, day_positions, is_train)
        X = np.hstack([base, neighbors])
        model = RandomForestRegressor(**RF_PARAMS).fit(X[is_train], y[is_train])
        pred[is_test] = model.predict(X[is_test])
        print(f"  fold {fold + 1}/{N_FOLDS} done", flush=True)
    return pred
