"""Turn satellite readings + weather into model features.

THE RULE THAT PREVENTS CHEATING
  To forecast the reading on target date D, we may only use information available at the
  issue date T = D - horizon:
    * weather up to and including T,
    * the most recent satellite reading taken on or before T - OBS_LATENCY_DAYS (scenes take a
      day or two to be processed and published) and how old it is.
  Nothing after T is ever used. tests/test_features.py checks the training table and the
  live forecast row agree exactly, which is the main guard against accidental leakage.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from . import config

CALM_KMH = 15.0  # a day whose strongest wind stays under this counts as "calm"

WEATHER_COLS = [
    "tmean_3", "tmean_7", "tmean_14", "tmax_7",
    "rain_3", "rain_7", "rain_14", "dry_days",
    "rad_7", "wind_7", "calm_7", "trend",
]
FEATURES = ["last_ndci", "last_age", "last3_mean", "basin_last"] + WEATHER_COLS + ["doy_sin", "doy_cos", "zone_idx"]

# Extra inputs the Forecast Lab can try (add_extra_features). Same no-peeking rule: all known at the issue date.
NEIGHBOR_COLS = ["nbr_last", "up_last", "nbr_diff"]          # the zones next to this one, on the same water body
MEMORY_COLS = ["mean30", "last_change", "prev_season_max"]   # a longer look back than the last reading


def _days_since(mask: np.ndarray, cap: int = 30) -> np.ndarray:
    out = np.empty(len(mask), dtype="float64")
    c = cap
    for i, hit in enumerate(mask):
        c = 0 if hit else min(c + 1, cap)
        out[i] = c
    return out


def weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    """Trailing-window weather features, one row per calendar day (windows end ON that day)."""
    w = weather.copy()
    w["date"] = pd.to_datetime(w["date"])
    w = w.set_index("date").sort_index()
    w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="D"))
    w = w.interpolate(limit=3)  # patch tiny gaps only
    f = pd.DataFrame(index=w.index)
    f["tmean_3"] = w["tmean"].rolling(3, min_periods=3).mean()
    f["tmean_7"] = w["tmean"].rolling(7, min_periods=6).mean()
    f["tmean_14"] = w["tmean"].rolling(14, min_periods=12).mean()
    f["tmax_7"] = w["tmax"].rolling(7, min_periods=6).max()
    f["rain_3"] = w["precip"].rolling(3, min_periods=3).sum()
    f["rain_7"] = w["precip"].rolling(7, min_periods=6).sum()
    f["rain_14"] = w["precip"].rolling(14, min_periods=12).sum()
    f["dry_days"] = _days_since((w["precip"] >= 5.0).to_numpy())
    f["rad_7"] = w["rad"].rolling(7, min_periods=6).mean()
    f["wind_7"] = w["wind"].rolling(7, min_periods=6).mean()
    calm = (w["wind"] < CALM_KMH).astype(float).where(w["wind"].notna())
    f["calm_7"] = calm.rolling(7, min_periods=6).sum()
    f["trend"] = f["tmean_7"] - w["tmean"].rolling(30, min_periods=24).mean()
    return f


def add_calendar(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    doy = pd.to_datetime(df[date_col]).dt.dayofyear.to_numpy()
    df = df.copy()
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def prepare_obs(ndci_df: pd.DataFrame) -> pd.DataFrame:
    df = ndci_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["zone_idx"] = df["zone_id"].str[1:].astype(int)
    if "site" not in df.columns:      # single-site data from older builds: all Lower Charles
        df["site"] = config.DEFAULT_SITE
    df["site"] = df["site"].fillna(config.DEFAULT_SITE)
    df = df.sort_values(["zone_id", "date"]).reset_index(drop=True)
    df["roll3"] = df.groupby("zone_id")["ndci"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    return df


def basin_series(obs: pd.DataFrame) -> pd.DataFrame:
    """Mean reading over each water body on each date ('basin' = the site the zone belongs to)."""
    b = obs.groupby(["site", "date"])["ndci"].mean().rename("basin_ndci").reset_index().sort_values("date")
    return b.reset_index(drop=True)


def build_training_table(
    obs: pd.DataFrame,
    wfeat: pd.DataFrame,
    horizon: int = config.HORIZON_DAYS,
    max_age: int = config.MAX_OBS_AGE_DAYS,
) -> pd.DataFrame:
    """One row per satellite reading we try to predict, using only what was known `horizon` days earlier."""
    left = obs[["zone_id", "zone_idx", "site", "date", "ndci"]].rename(columns={"date": "target_date", "ndci": "y"}).copy()
    left["T"] = left["target_date"] - pd.Timedelta(days=horizon)
    # a satellite scene only becomes usable a couple of days after it was taken
    left["T_obs"] = left["T"] - pd.Timedelta(days=config.OBS_LATENCY_DAYS)
    right = (
        obs[["zone_id", "date", "ndci", "roll3"]]
        .rename(columns={"date": "last_date", "ndci": "last_ndci", "roll3": "last3_mean"})
        .sort_values("last_date")
    )
    m = pd.merge_asof(left.sort_values("T_obs"), right, left_on="T_obs", right_on="last_date", by="zone_id", direction="backward")
    b = basin_series(obs).rename(columns={"date": "basin_date", "basin_ndci": "basin_last"})
    m = pd.merge_asof(m.sort_values("T_obs"), b, left_on="T_obs", right_on="basin_date", by="site", direction="backward")
    m["last_age"] = (m["T"] - m["last_date"]).dt.days
    m = m[m["last_age"].notna() & (m["last_age"] <= max_age)].copy()
    wf = wfeat.reindex(pd.DatetimeIndex(m["T"]))
    wf.index = m.index
    m = pd.concat([m, wf], axis=1)
    m = add_calendar(m, "target_date")
    m["year"] = m["target_date"].dt.year
    m = m.dropna(subset=FEATURES + ["y"]).sort_values(["target_date", "zone_id"]).reset_index(drop=True)
    return m


def feature_row(
    obs: pd.DataFrame,
    basin: pd.DataFrame,
    wfeat: pd.DataFrame,
    zone_id: str,
    as_of: pd.Timestamp,
    target_date: pd.Timestamp,
    max_age: int = config.MAX_OBS_AGE_DAYS,
) -> Tuple[Optional[dict], Optional[str]]:
    """Features for ONE zone as of `as_of`. Returns (features, None) or (None, reason)."""
    as_of = pd.Timestamp(as_of).normalize()
    usable_until = as_of - pd.Timedelta(days=config.OBS_LATENCY_DAYS)
    z = obs[(obs["zone_id"] == zone_id) & (obs["date"] <= usable_until)]
    if z.empty:
        return None, "no satellite reading yet"
    last = z.iloc[-1]
    age = int((as_of - last["date"]).days)
    if age > max_age:
        return None, f"latest clear satellite reading is {age} days old"
    site = last["site"] if "site" in z.columns else config.DEFAULT_SITE
    b = basin[(basin["date"] <= usable_until) & (basin["site"] == site)] if "site" in basin.columns else basin[basin["date"] <= usable_until]
    if b.empty:
        return None, "no basin reading yet"
    if as_of not in wfeat.index:
        return None, "no weather for this date"
    w = wfeat.loc[as_of, WEATHER_COLS]
    if w.isna().any():
        return None, "weather history incomplete for this date"
    cal = add_calendar(pd.DataFrame({"d": [pd.Timestamp(target_date)]}), "d").iloc[0]
    row = {
        "last_ndci": float(last["ndci"]),
        "last_age": age,
        "last3_mean": float(last["roll3"]),
        "basin_last": float(b.iloc[-1]["basin_ndci"]),
        **{c: float(w[c]) for c in WEATHER_COLS},
        "doy_sin": float(cal["doy_sin"]),
        "doy_cos": float(cal["doy_cos"]),
        "zone_idx": int(last["zone_idx"]),
    }
    return row, None


def add_extra_features(table: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """Add NEIGHBOR_COLS and MEMORY_COLS to a training table. Uses the table's T_obs (what was usable at issue time).

    nbr_last        mean of the latest readings of the zones either side, on the same water body (own reading if none)
    up_last         the zone to the west (upstream on the Charles, which flows west to east); own reading if none
    nbr_diff        nbr_last - last_ndci: is this zone greener or clearer than its neighbours?
    mean30          mean of this zone's readings in the 30 days up to T_obs
    last_change     the last reading minus the one before it (rising or falling?)
    prev_season_max the zone's highest reading in the previous calendar year (0 if none)
    """
    t = table.copy()
    t["_row"] = np.arange(len(t))
    site_of = obs.drop_duplicates("zone_idx").set_index("zone_idx")["site"].to_dict()
    for name, offset in (("_west", -1), ("_east", 1)):
        r = obs[["zone_idx", "date", "ndci"]].copy()
        r["key"] = r["zone_idx"] - offset                         # zone z+offset answers for zone z
        r = r[[site_of.get(k) == site_of.get(z) for k, z in zip(r["key"], r["zone_idx"])]]
        r = r.rename(columns={"date": name + "_date", "ndci": name})[["key", name + "_date", name]].sort_values(name + "_date")
        left = t[["_row", "zone_idx", "T_obs"]].rename(columns={"zone_idx": "key"}).sort_values("T_obs")
        got = pd.merge_asof(left, r, left_on="T_obs", right_on=name + "_date", by="key", direction="backward")
        stale = (got["T_obs"] - got[name + "_date"]).dt.days > config.MAX_OBS_AGE_DAYS
        got.loc[stale, name] = np.nan
        t[name] = got.set_index("_row")[name].reindex(t["_row"]).to_numpy()
    both = t[["_west", "_east"]]
    t["nbr_last"] = both.mean(axis=1).fillna(t["last_ndci"])
    t["up_last"] = t["_west"].fillna(t["last_ndci"])
    t["nbr_diff"] = t["nbr_last"] - t["last_ndci"]

    o = obs.sort_values(["zone_id", "date"]).copy()
    o["mean30"] = o.groupby("zone_id", group_keys=False).apply(
        lambda g: g.set_index("date")["ndci"].rolling("30D").mean()).to_numpy()
    o["last_change"] = o.groupby("zone_id")["ndci"].diff().fillna(0.0)
    r = o[["zone_id", "date", "mean30", "last_change"]].rename(columns={"date": "_mdate"}).sort_values("_mdate")
    left = t[["_row", "zone_id", "T_obs"]].sort_values("T_obs")
    got = pd.merge_asof(left, r, left_on="T_obs", right_on="_mdate", by="zone_id", direction="backward").set_index("_row")
    t["mean30"] = got["mean30"].reindex(t["_row"]).to_numpy()
    t["mean30"] = t["mean30"].fillna(t["last_ndci"])
    t["last_change"] = got["last_change"].reindex(t["_row"]).fillna(0.0).to_numpy()
    yearly = obs.assign(_y=obs["date"].dt.year).groupby(["zone_id", "_y"])["ndci"].max()
    t["prev_season_max"] = [float(yearly.get((z, y - 1), 0.0)) for z, y in zip(t["zone_id"], t["target_date"].dt.year)]
    return t.drop(columns=["_row", "_west", "_east"])
