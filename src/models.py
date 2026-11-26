"""Model wrappers: quantile regression forest (point + 90% interval) and LightGBM (point)."""
import numpy as np
from lightgbm import LGBMRegressor
from quantile_forest import RandomForestQuantileRegressor

INTERVAL_QUANTILES = [0.05, 0.95]
QRF_PARAMS = {
    "n_estimators": 150,
    "min_samples_leaf": 5,
    "max_features": 0.5,
    "max_samples": 0.5,
    "max_samples_leaf": None,
    "n_jobs": -1,
    "random_state": 0,
}
LGBM_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "random_state": 0,
    "verbose": -1,
}


def impute_with_train_medians(X_train, X_test):
    """Fill NaNs with training-fold medians only (no test information)."""
    medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    return np.where(np.isnan(X_train), medians, X_train), np.where(np.isnan(X_test), medians, X_test)


def fit_predict_qrf(X_train, y_train, X_test):
    model = RandomForestQuantileRegressor(**QRF_PARAMS).fit(X_train, y_train)
    mean = model.predict(X_test, quantiles="mean")
    bounds = model.predict(X_test, quantiles=INTERVAL_QUANTILES)
    return {"pred": mean, "lo": bounds[:, 0], "hi": bounds[:, 1]}, model


def fit_predict_lgbm(X_train, y_train, X_test):
    model = LGBMRegressor(**LGBM_PARAMS).fit(X_train, y_train)
    return {"pred": model.predict(X_test)}, model


MODELS = {"QRF": fit_predict_qrf, "LightGBM": fit_predict_lgbm}
