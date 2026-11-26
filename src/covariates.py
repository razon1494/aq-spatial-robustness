"""Covariates: NOAA GSOD daily weather, monitor elevation (Open-Meteo), county population density."""
import json
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0
CONUS_LAT = (24.0, 50.0)
CONUS_LON = (-125.5, -66.0)
GSOD_COLUMNS = ["STATION", "DATE", "LATITUDE", "LONGITUDE", "TEMP", "DEWP", "SLP",
                "VISIB", "WDSP", "MAX", "MIN", "PRCP"]
GSOD_MISSING = {"TEMP": 9999.9, "DEWP": 9999.9, "SLP": 9999.9, "MAX": 9999.9, "MIN": 9999.9,
                "VISIB": 999.9, "WDSP": 999.9, "PRCP": 99.99}
SENTINEL_TOLERANCE = 0.05
MIN_STATION_DAYS = 300
N_STATIONS = 3
MAX_STATION_KM = 150.0
KNOTS_TO_MS = 0.514444
MILES_TO_KM = 1.609344
INCH_TO_MM = 25.4
MAGNUS_A, MAGNUS_B = 17.625, 243.04
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
ELEVATION_BATCH = 100
ELEVATION_PAUSE_S = 12.0
RATE_LIMIT_WAIT_S = 65.0
MAX_RETRIES = 4
HTTP_STATUS_TOO_MANY = 429
HTTP_TIMEOUT_S = 30
MET_COLUMNS = ["temp_c", "dewpoint_c", "rh_pct", "slp_hpa", "wind_ms", "tmax_c", "tmin_c",
               "diurnal_range_c", "precip_mm", "visibility_km"]


def _f_to_c(fahrenheit):
    return (fahrenheit - 32.0) * 5.0 / 9.0


def relative_humidity(temp_c, dewpoint_c):
    """Magnus approximation, in percent."""
    return 100.0 * np.exp(MAGNUS_A * dewpoint_c / (MAGNUS_B + dewpoint_c)
                          - MAGNUS_A * temp_c / (MAGNUS_B + temp_c))


def clean_gsod(raw):
    """Replace GSOD missing-value sentinels with NaN and convert to SI units."""
    v = {col: raw[col].astype(float).mask(raw[col].astype(float) >= sentinel - SENTINEL_TOLERANCE)
         for col, sentinel in GSOD_MISSING.items()}
    temp_c, dew_c = _f_to_c(v["TEMP"]), _f_to_c(v["DEWP"])
    tmax_c, tmin_c = _f_to_c(v["MAX"]), _f_to_c(v["MIN"])
    return pd.DataFrame({
        "station": raw["STATION"].astype(str),
        "date": pd.to_datetime(raw["DATE"]),
        "lat": raw["LATITUDE"].astype(float),
        "lon": raw["LONGITUDE"].astype(float),
        "temp_c": temp_c,
        "dewpoint_c": dew_c,
        "rh_pct": relative_humidity(temp_c, dew_c),
        "slp_hpa": v["SLP"],
        "wind_ms": v["WDSP"] * KNOTS_TO_MS,
        "tmax_c": tmax_c,
        "tmin_c": tmin_c,
        "diurnal_range_c": tmax_c - tmin_c,
        "precip_mm": v["PRCP"] * INCH_TO_MM,
        "visibility_km": v["VISIB"] * MILES_TO_KM,
    })


def _in_conus(frame):
    lat, lon = frame["LATITUDE"].iloc[0], frame["LONGITUDE"].iloc[0]
    return CONUS_LAT[0] <= lat <= CONUS_LAT[1] and CONUS_LON[0] <= lon <= CONUS_LON[1]


def load_gsod_conus(tar_path, cache_path):
    """Read CONUS stations straight from the GSOD tar archive (no extraction); cache as parquet."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    frames = []
    with tarfile.open(tar_path) as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".csv"):
                continue
            frame = pd.read_csv(archive.extractfile(member), usecols=GSOD_COLUMNS,
                                dtype={"STATION": str})
            if len(frame) and frame["LATITUDE"].notna().iloc[0] and _in_conus(frame):
                frames.append(frame)
    gsod = clean_gsod(pd.concat(frames, ignore_index=True))
    days_with_temp = gsod.groupby("station")["temp_c"].transform("count")
    gsod = gsod[days_with_temp >= MIN_STATION_DAYS].reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    gsod.to_parquet(cache_path, index=False)
    return gsod


def nearest_stations(site_table, gsod):
    """For each site, the N nearest qualifying weather stations and their distances."""
    stations = gsod.groupby("station")[["lat", "lon"]].first()
    tree = BallTree(np.radians(stations[["lat", "lon"]].to_numpy()), metric="haversine")
    dist, idx = tree.query(np.radians(site_table[["lat", "lon"]].to_numpy()), k=N_STATIONS)
    return [
        pd.DataFrame({"site_id": site_table.index, "station": stations.index[idx[:, rank]],
                      "station_km": dist[:, rank] * EARTH_RADIUS_KM})
        for rank in range(N_STATIONS)
    ]


def daily_weather(site_days, site_table, gsod):
    """Weather per site-day from the nearest station, filling gaps from the 2nd and 3rd nearest."""
    keys = site_days[["site_id", "date"]]
    weather = gsod.drop_duplicates(["station", "date"]).set_index(["station", "date"])[MET_COLUMNS]
    combined = None
    for rank, match in enumerate(nearest_stations(site_table, gsod)):
        within = match[match["station_km"] <= MAX_STATION_KM]
        joined = keys.merge(within, on="site_id", how="left")
        values = weather.reindex(pd.MultiIndex.from_frame(joined[["station", "date"]])).to_numpy()
        frame = pd.DataFrame(values, columns=MET_COLUMNS, index=keys.index)
        if rank == 0:
            frame = frame.assign(met_station_km=joined["station_km"].to_numpy())
            combined = frame
        else:
            combined = combined.combine_first(frame)
    return combined[MET_COLUMNS + ["met_station_km"]]


def _get_json_with_backoff(url):
    """GET a JSON payload, waiting and retrying when the API rate-limits us."""
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != HTTP_STATUS_TOO_MANY or attempt == MAX_RETRIES - 1:
                raise
            print(f"  elevation API rate limit; waiting {RATE_LIMIT_WAIT_S:.0f}s", flush=True)
            time.sleep(RATE_LIMIT_WAIT_S)
    raise RuntimeError("unreachable")


def fetch_elevation(site_table, cache_path):
    """Monitor elevation (m) from the Open-Meteo elevation API (Copernicus 90 m DEM); cached."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pd.read_csv(cache_path, dtype={"site_id": str}).set_index("site_id")["elevation_m"]
    elevations = []
    for start in range(0, len(site_table), ELEVATION_BATCH):
        chunk = site_table.iloc[start:start + ELEVATION_BATCH]
        lats = ",".join(f"{x:.5f}" for x in chunk["lat"])
        lons = ",".join(f"{x:.5f}" for x in chunk["lon"])
        payload = _get_json_with_backoff(f"{ELEVATION_URL}?latitude={lats}&longitude={lons}")
        if len(payload.get("elevation", [])) != len(chunk):
            raise RuntimeError(f"Elevation API returned an unexpected payload for batch at {start}")
        elevations.extend(payload["elevation"])
        time.sleep(ELEVATION_PAUSE_S)
    result = pd.Series(elevations, index=site_table.index, name="elevation_m")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.rename_axis("site_id").to_csv(cache_path)
    return result


def county_population_density(population_csv, gazetteer_zip):
    """log10 people per square mile of land, by 5-digit county FIPS."""
    pop = pd.read_csv(population_csv, encoding="latin-1", dtype=str,
                      usecols=["SUMLEV", "STATE", "COUNTY", "POPESTIMATE2023"])
    pop = pop[pop["SUMLEV"] == "050"]
    pop = pd.DataFrame({"county_fips": pop["STATE"].str.zfill(2) + pop["COUNTY"].str.zfill(3),
                        "population": pop["POPESTIMATE2023"].astype(float)})
    with zipfile.ZipFile(gazetteer_zip) as archive:
        with archive.open(archive.namelist()[0]) as handle:
            gaz = pd.read_csv(handle, sep="\t", dtype={"GEOID": str})
    gaz.columns = gaz.columns.str.strip()
    land = gaz[["GEOID", "ALAND_SQMI"]].rename(columns={"GEOID": "county_fips"})
    merged = pop.merge(land, on="county_fips", how="inner")
    density = np.log10(merged["population"] / merged["ALAND_SQMI"].clip(lower=0.01))
    return pd.Series(density.to_numpy(), index=merged["county_fips"], name="log10_pop_density")
