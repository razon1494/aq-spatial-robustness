"""Download the public raw inputs into data/raw/ (about 120 MB, no API keys).

Usage (from the project root): python src/download_data.py
Elevation is fetched later, on first run, from the Open-Meteo elevation API and cached in data/interim/.
"""
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
SOURCES = {
    "daily_88101_2024.zip": "https://aqs.epa.gov/aqsweb/airdata/daily_88101_2024.zip",
    "est23all.txt": "https://www2.census.gov/programs-surveys/saipe/datasets/2023/2023-state-and-county/est23all.txt",
    "gsod_2024.tar.gz": "https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive/2024.tar.gz",
    "co-est2023-alldata.csv": "https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/counties/totals/co-est2023-alldata.csv",
    "2023_Gaz_counties_national.zip": "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_counties_national.zip",
}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        target = RAW / name
        if target.exists():
            print(f"exists: {name}")
            continue
        print(f"downloading {name} ...", flush=True)
        urllib.request.urlretrieve(url, target)
    print("done")


if __name__ == "__main__":
    main()
