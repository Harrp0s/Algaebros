"""Generate FAKE data in the same file formats as the real pipeline, to test the code offline.

    python tools/make_synthetic_data.py          # writes ./data_synthetic (never touches ./data)

Everything produced is tagged source='synthetic' in manifest.json. Alerts and the map show
a loud SYNTHETIC banner. Nothing trained on this says anything about the real Charles River.
"""
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ALGAEBRO_DATA_DIR", str(ROOT / "data_synthetic"))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from algaebro import config, geo, zones


def make_river(h=220, w=330):
    """A sinuous river ~11 px wide with 'bridges' (gaps), plus a pond that should be ignored."""
    freq = np.zeros((h, w), dtype="float32")
    cols = np.arange(w)
    center = 110 + 28 * np.sin(cols / 45.0) + 10 * np.sin(cols / 13.0)
    for c in cols:
        lo, hi = int(center[c] - 5), int(center[c] + 6)
        freq[lo:hi, c] = 0.95
    for bridge in (60, 130, 200, 265):  # land crossing the river
        freq[:, bridge:bridge + 2] = 0.0
    freq[20:28, 20:28] = 0.95  # a small pond, far from the river
    return freq


def make_grid(freq):
    h, w = freq.shape
    x0, y0 = geo.lonlat_to_utm(config.BBOX[0], config.BBOX[1])
    x = x0 + config.RESOLUTION_M * (0.5 + np.arange(w))
    y = y0 + config.RESOLUTION_M * (0.5 + np.arange(h))[::-1]  # north-up, like real scenes
    return x, y


def make_weather(rng, start, end):
    days = pd.date_range(start, end, freq="D")
    n = len(days)
    doy = days.dayofyear.to_numpy()
    seasonal = 10 + 14 * np.sin(2 * np.pi * (doy - 110) / 365.25)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 1.6)
    tmean = seasonal + noise
    for yr in range(days.year.min(), days.year.max() + 1):  # a few heat waves each summer
        for _ in range(rng.integers(1, 4)):
            s = pd.Timestamp(year=yr, month=int(rng.integers(6, 9)), day=int(rng.integers(1, 25)))
            m = (days >= s) & (days < s + pd.Timedelta(days=int(rng.integers(3, 7))))
            tmean[m] += 6
    rain_day = rng.random(n) < 0.30
    precip = np.where(rain_day, rng.exponential(8.0, n), 0.0)
    wind = np.clip(rng.lognormal(np.log(20), 0.35, n) - 4 * (precip == 0) * rng.random(n), 5, 70)
    rad = np.clip(8 + 14 * (np.sin(2 * np.pi * (doy - 80) / 365.25) + 1) / 2 - 0.15 * precip + rng.normal(0, 2, n), 1, 32)
    df = pd.DataFrame({
        "date": days, "tmean": tmean, "tmax": tmean + 4 + rng.normal(0, 1, n), "tmin": tmean - 4 + rng.normal(0, 1, n),
        "precip": precip, "rad": rad, "wind": wind, "sun_s": np.clip(rad * 2400, 0, 50000),
    })
    return df


def make_ndci(rng, weather, zone_ids, weather_effect=1.0):
    w = weather.set_index("date")
    tmean7 = w["tmean"].rolling(7, min_periods=3).mean().bfill().fillna(15.0)
    rain3 = w["precip"].rolling(3, min_periods=1).sum().fillna(0.0)
    calm = (w["wind"] < 15).astype(float).rolling(7, min_periods=3).sum().bfill().fillna(3.0)
    zone_offset = {z: rng.normal(0, 0.02) for z in zone_ids}
    rows = []
    bloom = {z: 0.0 for z in zone_ids}
    d = w.index.min()
    obs_dates = set()
    for yr in sorted(set(w.index.year)):
        t = pd.Timestamp(year=yr, month=5, day=1)
        while t <= pd.Timestamp(year=yr, month=10, day=31):
            obs_dates.add(t + pd.Timedelta(days=int(rng.integers(-1, 2))))
            t += pd.Timedelta(days=5)
    for day in pd.date_range(w.index.min(), w.index.max(), freq="D"):
        shared = 0.03 * weather_effect * ((tmean7.get(day, 15) - 20) / 5) + 0.015 * weather_effect * (calm.get(day, 3) - 3) / 3
        shared -= 0.02 * weather_effect * min(rain3.get(day, 0), 30) / 30
        for z in zone_ids:
            bloom[z] = 0.93 * bloom[z] + shared + rng.normal(0, 0.012)
            bloom[z] = float(np.clip(bloom[z], -0.4, 1.2))
        if day in obs_dates and day.month in config.SEASON_MONTHS and day <= w.index.max() - pd.Timedelta(days=2):
            for z in zone_ids:
                if rng.random() < 0.30:  # cloud gap or too few clean pixels
                    continue
                v = -0.05 + 0.45 * bloom[z] + zone_offset[z] + rng.normal(0, 0.025)
                n = int(rng.integers(40, 400))
                rows.append({"date": day.strftime("%Y-%m-%d"), "zone_id": z, "item_id": f"SYN_{day:%Y%m%d}",
                             "ndci": round(float(v), 4), "p25": round(float(v - 0.02), 4), "p75": round(float(v + 0.02), 4),
                             "n_valid": n, "frac_valid": round(float(rng.uniform(0.35, 1.0)), 3)})
    return pd.DataFrame(rows)


def main(weather_effect=1.0, seed=0, years=(2019, 2026), end=None):
    """end: last day of data. Defaults to 'today minus 2 days'; tests pin it so the made-up data never changes with the calendar."""
    rng = np.random.default_rng(seed)
    freq = make_river()
    x, y = make_grid(freq)
    labels = zones.build_zone_labels(freq)
    recs = zones.save_zones(labels, x, y, freq)
    zones.save_preview(labels, freq, recs, x, y)
    end = end or (date.today() - timedelta(days=2))
    wdf = make_weather(rng, f"{years[0]}-03-15", end)
    # a "forecast" tail like the real weather file has
    tail = make_weather(rng, end + timedelta(days=1), end + timedelta(days=8))
    wdf = pd.concat([wdf, tail], ignore_index=True)
    wdf.to_csv(config.WEATHER_CSV, index=False)
    nd = make_ndci(rng, wdf[wdf["date"] <= pd.Timestamp(end)], [r["id"] for r in recs], weather_effect)
    nd.to_csv(config.NDCI_CSV, index=False)
    config.MANIFEST_JSON.write_text(json.dumps({"source": "synthetic", "weather_effect": weather_effect, "seed": seed}))
    print(f"SYNTHETIC data written to {config.DATA_DIR}: {len(recs)} zones, {len(nd)} readings, {len(wdf)} weather days")


if __name__ == "__main__":
    main()
