"""Produce the forecast for every zone as of a given date.

Two uses:
  * LIVE   : as_of = today. Uses the freshest weather (and the weather forecast, only to spot heat waves).
  * REPLAY : as_of = a past date. Uses only data that existed on that date, then (if available)
             attaches what the satellite actually saw a week later so you can judge the forecast.
             This is how you demo during the off-season, and it is labelled as a replay everywhere.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd

from . import config, features, horizons, model as model_mod, weather, zones as zones_mod

_cache: Dict[str, dict] = {}


class Inputs:
    """Everything needed to forecast, loaded once."""

    def __init__(self):
        self.zones = zones_mod.load_zone_records()
        ndci = pd.read_csv(config.NDCI_CSV)
        self.obs = features.prepare_obs(ndci)
        self.basin = features.basin_series(self.obs)
        self.weather = weather.load()
        self.wfeat = features.weather_features(self.weather)
        self.bundle = model_mod.load_bundle()
        self.horizon_models = horizons.load_models()     # {days ahead: model}, from the 1-7 day training (may be empty)

    def reload_horizons(self) -> None:
        self.horizon_models = horizons.load_models()


def _context_flags(w: pd.Series, ref: dict) -> List[str]:
    flags = []
    if w["tmean_7"] >= ref["tmean_7_q80"]:
        flags.append("warm")
    if w["wind_7"] <= ref["wind_7_q25"]:
        flags.append("light winds")
    if w["rad_7"] >= ref["rad_7_q75"]:
        flags.append("very sunny")
    if w["dry_days"] >= ref["dry_days_q75"]:
        flags.append("dry spell")
    return flags


def _heat_wave_ahead(weather_df: pd.DataFrame, as_of: pd.Timestamp) -> Optional[bool]:
    """True/False if the next days' forecast has a run of very hot days; None if we cannot know."""
    nxt = weather_df[(weather_df["date"] > as_of) & (weather_df["date"] <= as_of + pd.Timedelta(days=7))]
    if len(nxt) < config.HEAT_WAVE_DAYS:
        return None
    run = best = 0
    for tmax in nxt["tmax"]:
        run = run + 1 if (pd.notna(tmax) and tmax >= config.HEAT_WAVE_TMAX_C) else 0
        best = max(best, run)
    return best >= config.HEAT_WAVE_DAYS


def make_forecast(as_of=None, inputs: Optional[Inputs] = None, horizon: Optional[int] = None) -> dict:
    inp = inputs or Inputs()
    bundle = inp.bundle
    horizon = int(horizon or bundle.get("horizon", config.HORIZON_DAYS))
    today = pd.Timestamp(date.today())
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else today
    live = as_of >= today - pd.Timedelta(days=1)
    target = as_of + pd.Timedelta(days=horizon)
    in_season = target.month in config.SEASON_MONTHS

    # weather context as of the issue date (trailing windows only)
    ctx: dict = {"flags": [], "heat_wave_ahead": None}
    wrow = None
    if as_of in inp.wfeat.index and not inp.wfeat.loc[as_of, features.WEATHER_COLS].isna().any():
        wrow = inp.wfeat.loc[as_of]
        ctx.update(
            {
                "tmean_7_c": round(float(wrow["tmean_7"]), 1),
                "rain_7_mm": round(float(wrow["rain_7"]), 1),
                "dry_days": int(wrow["dry_days"]),
                "calm_days_7": int(wrow["calm_7"]),
                "flags": _context_flags(wrow, bundle["weather_ref"]),
            }
        )
    if live:  # looking at the weather forecast is only legitimate when it IS the future
        ctx["heat_wave_ahead"] = _heat_wave_ahead(inp.weather, as_of)

    rows, ids, reasons = [], [], {}
    for z in inp.zones:
        row, why = features.feature_row(inp.obs, inp.basin, inp.wfeat, z["id"], as_of, target)
        if row is None:
            reasons[z["id"]] = why
        else:
            rows.append(row)
            ids.append(z["id"])
    preds = dict(zip(ids, model_mod.predict_rows(bundle, rows))) if rows else {}
    feats = dict(zip(ids, rows))
    hm = (getattr(inp, "horizon_models", None) or {}).get(horizon)
    ml = {}
    if hm is not None and rows:
        try:
            ml = horizons.predict(hm, rows, ids, inp.obs, inp.zones, as_of, target)
        except Exception as e:      # never lose the forecast over the optional ML view
            print(f"[forecast] ML forecast for {horizon} days failed: {type(e).__name__}: {e}")
    use_ml = bool(hm and hm.get("use_ml") and ml)

    out_zones = []
    for z in inp.zones:
        zid = z["id"]
        site = z.get("site", config.DEFAULT_SITE)
        entry = {"id": zid, "number": z["number"], "name": z["name"], "lat": z["lat"], "lon": z["lon"], "site": site,
                 "low_confidence": bool(config.SITE_BY_KEY.get(site, {}).get("low_confidence"))}
        if zid in preds:
            p = float(preds[zid])
            tier = model_mod.tier_for(bundle, p)
            rule_tier = tier
            if use_ml:                  # alerts switch to the ML only where it clearly won (see horizons.py)
                tier = ml[zid]["tier"]
            entry.update(
                {
                    "tier": tier,
                    "ndci_pred": round(p, 4),
                    "last_obs": round(feats[zid]["last_ndci"], 4),
                    "obs_age_days": int(feats[zid]["last_age"]),
                    "track": (bundle.get("tier_record") or {}).get(tier),
                    "inputs": {k: round(float(v), 4) for k, v in feats[zid].items()},   # what the forecast was built from
                    "rule_tier": rule_tier,
                }
            )
            if zid in ml:
                entry["ml"] = ml[zid]
        else:
            entry.update({"tier": "unknown", "reason": reasons.get(zid, "no forecast")})
        if not live:  # replay: what did the satellite actually see around the target date?
            zobs = inp.obs[(inp.obs["zone_id"] == zid) & ((inp.obs["date"] - target).abs() <= pd.Timedelta(days=3))]
            if not zobs.empty:
                near = zobs.iloc[(zobs["date"] - target).abs().argsort().iloc[0]]
                entry["verification"] = {
                    "observed": round(float(near["ndci"]), 4),
                    "observed_date": str(near["date"].date()),
                    "observed_tier": model_mod.tier_for(bundle, float(near["ndci"])),
                }
        out_zones.append(entry)

    fc = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": str(as_of.date()),
        "target_date": str(target.date()),
        "horizon_days": horizon,
        "live": bool(live),
        "replay": not live,
        "in_season": bool(in_season),
        "data_source": bundle.get("data_source", config.data_source()),
        "method": "model" if use_ml else bundle["method"],
        "method_reason": (f"For {horizon} days ahead the ML model ({hm.get('name')}) clearly beat the latest reading "
                          f"in testing and held up on the unseen season, so alerts use it." if use_ml else bundle["method_reason"]),
        "ml": ({"available": True, "name": hm.get("name"), "method": hm.get("method"), "ml_f1": hm.get("ml_f1"),
                "rule_f1": hm.get("rule_f1"), "used_for_alerts": use_ml} if hm else {"available": False}),
        "thresholds": {"elevated": round(bundle["thr_elev"], 4), "high": round(bundle["thr_high"], 4)},
        "weather": ctx,
        "zones": out_zones,
    }
    return fc


def save_forecast(fc: dict) -> None:
    config.FORECAST_JSON.write_text(json.dumps(fc, indent=1))


def get_forecast(as_of: Optional[str] = None, max_age_hours: float = 6.0, refresh_weather: bool = True) -> dict:
    """What the bot uses. Cached so every text message does not rebuild the forecast."""
    key = as_of or "live"
    now = datetime.now(timezone.utc)
    hit = _cache.get(key)
    if hit and (as_of or (now - datetime.fromisoformat(hit["generated_at"])).total_seconds() < max_age_hours * 3600):
        return hit
    if not as_of and refresh_weather:
        try:
            weather.refresh_recent()
        except Exception as e:  # keep working with what we have
            print(f"[forecast] could not refresh weather ({e}); using saved weather")
    fc = make_forecast(as_of)
    if not as_of:
        save_forecast(fc)
    _cache[key] = fc
    return fc


def make_replay(year: int, step_days: int = 7, inputs: Optional[Inputs] = None) -> dict:
    """Forecasts issued every `step_days` through a past season, with what happened next."""
    inp = inputs or Inputs()
    start = pd.Timestamp(year=year, month=config.SEASON_MONTHS[0], day=1)
    end = pd.Timestamp(year=year, month=config.SEASON_MONTHS[-1], day=28)
    end = min(end, pd.Timestamp(date.today()) - pd.Timedelta(days=inp.bundle.get("horizon", 7) + 1))
    frames = []
    for d in pd.date_range(start, end, freq=f"{step_days}D"):
        fc = make_forecast(d, inputs=inp)
        if all(z["tier"] == "unknown" for z in fc["zones"]):
            continue
        frames.append(
            {
                "as_of": fc["as_of"],
                "target_date": fc["target_date"],
                "weather": fc["weather"],
                "zones": {
                    z["id"]: {k: z[k] for k in ("tier", "ndci_pred", "last_obs", "verification", "reason") if k in z}
                    for z in fc["zones"]
                },
            }
        )
    replay = {
        "year": year,
        "data_source": inp.bundle.get("data_source", config.data_source()),
        "method": inp.bundle["method"],
        "thresholds": {"elevated": inp.bundle["thr_elev"], "high": inp.bundle["thr_high"]},
        "zones": [{k: z[k] for k in ("id", "number", "name", "lat", "lon", "points")} for z in inp.zones],
        "frames": frames,
    }
    config.REPLAY_JSON.write_text(json.dumps(replay))
    return replay


def parse_as_of(text: str) -> str:
    """argparse type for --as-of: a real calendar date written YYYY-MM-DD, or a plain-English complaint."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text or ""):
        raise argparse.ArgumentTypeError(f"'{text}' is not a full date. Write it as YYYY-MM-DD, for example 2025-10-03.")
    try:
        pd.Timestamp(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not a real calendar date. Example: 2025-10-03.") from None
    return text
