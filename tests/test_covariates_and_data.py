import numpy as np
import pandas as pd

from covariates import clean_gsod, daily_weather, relative_humidity
from data import load_county_income
from models import impute_with_train_medians


def _gsod_row(station, date, lat, temp_f=50.0, visib=10.0, wdsp=10.0, prcp=0.1):
    return {"STATION": station, "DATE": date, "LATITUDE": lat, "LONGITUDE": -111.0,
            "TEMP": temp_f, "DEWP": 32.0, "SLP": 1015.0, "VISIB": visib, "WDSP": wdsp,
            "MAX": 60.0, "MIN": 40.0, "PRCP": prcp}


def test_clean_gsod_converts_units_and_drops_sentinels():
    raw = pd.DataFrame([_gsod_row("s1", "2024-01-01", 40.0, temp_f=32.0, visib=999.9, prcp=99.99)])
    clean = clean_gsod(raw).iloc[0]
    assert clean["temp_c"] == 0.0
    assert np.isnan(clean["visibility_km"]) and np.isnan(clean["precip_mm"])
    assert abs(clean["wind_ms"] - 5.14444) < 1e-4
    assert clean["diurnal_range_c"] == (60.0 - 40.0) * 5.0 / 9.0


def test_relative_humidity_is_100_when_saturated():
    assert abs(relative_humidity(15.0, 15.0) - 100.0) < 1e-9


def test_daily_weather_fills_gaps_from_next_nearest_station():
    raw = pd.DataFrame([
        _gsod_row("near", "2024-01-01", 40.01, temp_f=50.0),
        _gsod_row("far", "2024-01-01", 40.50, temp_f=68.0),
        _gsod_row("far", "2024-01-02", 40.50, temp_f=68.0),
        _gsod_row("third", "2024-01-01", 41.00, temp_f=86.0),
    ])
    gsod = clean_gsod(raw)
    sites = pd.DataFrame({"lat": [40.0], "lon": [-111.0]}, index=pd.Index(["site"], name="site_id"))
    site_days = pd.DataFrame({"site_id": ["site", "site"],
                              "date": pd.to_datetime(["2024-01-01", "2024-01-02"])})
    weather = daily_weather(site_days, sites, gsod)
    assert weather["temp_c"].iloc[0] == 10.0   # nearest station
    assert weather["temp_c"].iloc[1] == 20.0   # filled from the second-nearest station


def test_load_county_income_skips_state_rows(tmp_path):
    numbers = " ".join(["1"] * 18)
    lines = [f"49   0 {numbers} 70000 69000 71000 1 1 1 1 1 1 Utah UT x",
             f"49  35 100 90 110 9.5 9.0 10.0 {' '.join(['1'] * 12)} 90000 89000 91000 1 1 1 1 1 1 Salt Lake County UT x"]
    path = tmp_path / "saipe.txt"
    path.write_text("\n".join(lines))
    income = load_county_income(path)
    assert list(income["county_fips"]) == ["49035"]
    assert income["median_income"].iloc[0] == 90000.0
    assert income["poverty_pct"].iloc[0] == 9.5


def test_imputation_uses_training_medians_only():
    X_train = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    X_test = np.array([[np.nan, np.nan]])
    _, filled = impute_with_train_medians(X_train, X_test)
    np.testing.assert_allclose(filled, [[3.0, 5.0]])
