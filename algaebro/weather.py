"""Daily weather from Open-Meteo (free, no key, non-commercial use).

  * Archive API  : reanalysis history back to 1940, available up to about 2 days ago.
  * Forecast API : the recent past days plus the next days (used to fill the last 2 days and
                   to look ahead for heat waves).

One point stands in for the whole basin because the weather grid cells are ~10 km wide.
Forecast-model values and reanalysis values differ slightly; we prefer reanalysis wherever both exist.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from . import config

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo daily variable -> our column
DAILY = {
    "temperature_2m_mean": "tmean",      # deg C
    "temperature_2m_max": "tmax",        # deg C
    "temperature_2m_min": "tmin",        # deg C
    "precipitation_sum": "precip",       # mm
    "shortwave_radiation_sum": "rad",    # MJ/m2
    "wind_speed_10m_max": "wind",        # km/h
    "sunshine_duration": "sun_s",        # seconds
}


def _get(url: str, params: dict, tries: int = 4) -> dict:
    last = None
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
        except requests.RequestException as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.json()
        last = f"HTTP {r.status_code}: {r.text[:200]}"
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(3 * (2 ** attempt))
            continue
        break
    raise RuntimeError(f"Open-Meteo request failed ({last})")


def _to_frame(payload: dict) -> pd.DataFrame:
    d = payload["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(d["time"])})
    for api_name, col in DAILY.items():
        df[col] = d.get(api_name)
    return df


def fetch_archive(start: str, end: str, point=None) -> pd.DataFrame:
    lat, lon = point or config.WEATHER_POINT
    payload = _get(
        ARCHIVE_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "daily": ",".join(DAILY),
            "timezone": config.TIMEZONE,
        },
    )
    return _to_frame(payload)


def fetch_recent(past_days: int = 14, forecast_days: int = 8, point=None) -> pd.DataFrame:
    lat, lon = point or config.WEATHER_POINT
    payload = _get(
        FORECAST_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "daily": ",".join(DAILY),
            "past_days": past_days,
            "forecast_days": forecast_days,
            "timezone": config.TIMEZONE,
        },
    )
    return _to_frame(payload)


def fetch_all(start: Optional[str] = None, end: Optional[str] = None, log=print) -> pd.DataFrame:
    """Archive history merged with the latest days; written to data/weather.csv."""
    start = start or (pd.Timestamp(config.START_DATE) - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    end = end or (date.today() - timedelta(days=2)).strftime("%Y-%m-%d")
    log(f"Open-Meteo archive {start} to {end} ...")
    arch = fetch_archive(start, end)
    log(f"  {len(arch)} days. Fetching the latest days and forecast ...")
    try:
        recent = fetch_recent()
    except RuntimeError as e:
        log(f"  (could not fetch recent days: {e}); continuing with the archive only")
        recent = arch.iloc[0:0]
    df = arch.set_index("date").combine_first(recent.set_index("date")).sort_index()
    df.index.name = "date"
    df = df.reset_index()
    df.to_csv(config.WEATHER_CSV, index=False)
    log(f"  wrote {len(df)} days to {config.WEATHER_CSV}")
    return df


def refresh_recent(log=print) -> pd.DataFrame:
    """Cheap update used while the bot runs: re-fetch only the latest days and merge into the CSV."""
    cur = load()
    recent = fetch_recent()
    # Keep reanalysis history as is (that is what the model was trained on); only the last
    # 3 days and the future are replaced with the freshest forecast-model values.
    cutoff = pd.Timestamp(date.today() - timedelta(days=3))
    older = cur[cur["date"] <= cutoff].set_index("date")
    older = older.combine_first(recent[recent["date"] <= cutoff].set_index("date"))
    fresh = recent[recent["date"] > cutoff].set_index("date")
    df = pd.concat([older, fresh]).sort_index()
    df.index.name = "date"
    df = df.reset_index()
    df.to_csv(config.WEATHER_CSV, index=False)
    return df


def load() -> pd.DataFrame:
    df = pd.read_csv(config.WEATHER_CSV, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)
