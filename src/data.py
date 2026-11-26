"""Load EPA AQS daily PM2.5 monitors and Census SAIPE county income."""
import numpy as np
import pandas as pd

NON_CONUS_STATES = frozenset({"02", "15", "66", "72", "78", "80", "CC"})
EPA_COLUMNS = [
    "State Code", "County Code", "Site Num", "Latitude", "Longitude",
    "Date Local", "Arithmetic Mean", "State Name",
]
SAIPE_POVERTY_PCT_IDX = 5
SAIPE_MEDIAN_INCOME_IDX = 20
SAIPE_MIN_TOKENS = 21


def load_pm25_site_days(zip_path):
    """One row per monitoring site per day (median across instruments), CONUS only."""
    raw = pd.read_csv(
        zip_path,
        usecols=EPA_COLUMNS,
        dtype={"State Code": str, "County Code": str, "Site Num": str},
    )
    state = raw["State Code"].str.zfill(2)
    county_fips = state + raw["County Code"].str.zfill(3)
    keyed = raw.assign(
        state=state,
        county_fips=county_fips,
        site_id=county_fips + raw["Site Num"].str.zfill(4),
    )
    conus = keyed[~keyed["state"].isin(NON_CONUS_STATES)].dropna(subset=["Arithmetic Mean"])
    daily = conus.groupby(["site_id", "Date Local"], as_index=False).agg(
        pm25=("Arithmetic Mean", "median"),
        lat=("Latitude", "first"),
        lon=("Longitude", "first"),
        state=("state", "first"),
        state_name=("State Name", "first"),
        county_fips=("county_fips", "first"),
    )
    date = pd.to_datetime(daily["Date Local"])
    return (
        daily.assign(date=date, doy=date.dt.dayofyear, dow=date.dt.dayofweek)
        .drop(columns="Date Local")
        .sort_values(["date", "site_id"])
        .reset_index(drop=True)
    )


def _to_float(token):
    try:
        return float(token)
    except ValueError:
        return np.nan


def load_county_income(saipe_path):
    """County median household income and poverty rate from the SAIPE fixed-width file."""
    records = []
    with open(saipe_path, encoding="latin-1") as handle:
        for line in handle:
            tokens = line.split()
            is_state_row = len(tokens) > 1 and tokens[1] == "0"
            if len(tokens) < SAIPE_MIN_TOKENS or is_state_row:
                continue
            records.append({
                "county_fips": tokens[0].zfill(2) + tokens[1].zfill(3),
                "poverty_pct": _to_float(tokens[SAIPE_POVERTY_PCT_IDX]),
                "median_income": _to_float(tokens[SAIPE_MEDIAN_INCOME_IDX]),
            })
    return pd.DataFrame.from_records(records)
