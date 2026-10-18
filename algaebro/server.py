"""One local process: the dashboard, its API, and (optionally) the live iMessage bot.

    python -m algaebro.server                      # dashboard + bot (needs .env for Photon)
    python -m algaebro.server --as-of 2025-10-03   # start in REPLAY mode
    python -m algaebro.server --no-photon          # dashboard only: alerts show in the feed, nothing is texted

It listens on this computer only (127.0.0.1). Photos and phone numbers never leave it.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import threading
import time
import webbrowser
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import pandas as pd

from . import (ask, config, forecast, heat, horizons, lab, labcore, nessie, photos, pixels, planner, population, proximity, screening,
               store as store_mod, weather, xai, zones as zones_mod)
from .bot import Bot
from .messenger import describe_error
from .run_bot import handle_incoming, load_dotenv
from .store import Store, mask_handle

WEB_DIR = Path(__file__).resolve().parent / "web"


def parse_days(days) -> int:
    """Days ahead for the slider: 1-7, anything else (or nothing) means the standard horizon."""
    try:
        d = int(days)
    except (TypeError, ValueError):
        return config.HORIZON_DAYS
    return d if 1 <= d <= 7 else config.HORIZON_DAYS


class App:
    """Everything the HTTP handlers and the bot thread share."""

    def __init__(self, messenger=None, as_of: Optional[str] = None, db_path=None, refresh_weather: bool = True):
        self.messenger = messenger
        self.as_of = as_of
        self.db_path = db_path
        self.zones = zones_mod.load_zone_records()
        self.zone_by_id = {z["id"]: z for z in self.zones}
        if refresh_weather and not as_of:
            try:
                weather.refresh_recent()
            except Exception as e:
                print(f"[server] could not refresh weather ({describe_error(e)}); using saved weather")
        self.inputs = forecast.Inputs()
        self._local = threading.local()
        self._fc_cache = {}
        self.memo = {}                       # simulated-alert cooldowns: (user, zone) -> time
        self.lock = threading.RLock()
        self.jobs = lab.Jobs()
        self._hits = collections.deque()     # recent chat/voice requests, for a simple rate limit
        try:
            self.labels, self.grid_x, self.grid_y = zones_mod.load_grid()
        except (OSError, ValueError):
            self.labels = self.grid_x = self.grid_y = None
        self.anchors = {z["id"]: heat.medoid(z) for z in self.zones}     # a point certainly on the water, for markers
        self._heat_meta = {}                                              # cache key -> meta
        self._heat_png = collections.OrderedDict()                        # token -> png bytes

    # ------------------------------------------------------------------ plumbing
    def store(self) -> Store:
        """One database connection per thread."""
        st = getattr(self._local, "store", None)
        if st is None:
            st = Store(self.db_path)
            self._local.store = st
        return st

    def forecast_now(self, days: Optional[int] = None) -> dict:
        """The forecast for the date being shown. `days` ahead (1-7) is for the dashboard's slider only:
        alerts and the bot always use the standard horizon (config.HORIZON_DAYS)."""
        days = parse_days(days)
        key = (self.as_of or "live", days)
        with self.lock:
            hit = self._fc_cache.get(key)
            if hit and (self.as_of or time.time() - hit[0] < 3 * 3600):
                return hit[1]
            fc = forecast.make_forecast(self.as_of, inputs=self.inputs, horizon=days)
            self._fc_cache[key] = (time.time(), fc)
            return fc

    # ------------------------------------------------------------------ demo state
    def demo_enabled(self) -> bool:
        return self.store().kv_get("demo_enabled", "0") == "1"

    def demo_user(self) -> Optional[str]:
        return self.store().kv_get("demo_user")

    def users(self):
        out = []
        for u in self.store().list_users():
            out.append({
                "user_id": u["user_id"], "label": mask_handle(u["handle"]),
                "reachable": bool(u["space_id"] and u["agent_id"]),
                "has_location": u["lat"] is not None, "alerts_off": bool(u["alerts_off"]),
            })
        return out

    def set_demo(self, enabled: bool, user_id: Optional[str] = None) -> dict:
        st = self.store()
        if enabled:
            users = [u for u in st.list_users() if u["space_id"] and u["agent_id"]]
            chosen = user_id or self.demo_user() or (users[0]["user_id"] if users else None)
            if chosen is None or st.get_user(chosen) is None:
                return {"ok": False, "error": "No one has texted the bot yet. Text it from your phone first."}
            st.kv_set("demo_user", chosen)
        st.kv_set("demo_enabled", "1" if enabled else "0")
        return {"ok": True}

    def set_pin(self, lat: float, lon: float) -> dict:
        if not self.demo_enabled():
            return {"ok": False, "error": "Turn on Demo mode first."}
        uid = self.demo_user()
        st = self.store()
        if uid is None or st.get_user(uid) is None:
            return {"ok": False, "error": "Choose a phone first."}
        st.set_location(uid, float(lat), float(lon), "pin")
        res = self.check_user(uid)
        near = proximity.nearest_zones(self.zones, float(lat), float(lon))[0]
        fc = self.forecast_now()
        fz = next(z for z in fc["zones"] if z["id"] == near[1]["id"])
        return {"ok": True, "alerted": bool(res["sent"]), "result": res,
                "nearest": {"name": fz["name"], "distance_m": round(near[0]), "tier": fz["tier"],
                            "flagged": fz["tier"] == "high" and not fz.get("low_confidence"),
                            "low_confidence": bool(fz.get("low_confidence")),
                            "within_radius": near[0] <= config.PROXIMITY_RADIUS_M}}

    def set_as_of(self, as_of: Optional[str], notify: bool = True) -> dict:
        """Change the date being shown. notify=False is for scrubbing the timeline: no phone is texted on the way."""
        if as_of:
            pd.Timestamp(as_of)  # raises on nonsense
        with self.lock:
            self.as_of = as_of or None
        res = self.check_everyone() if notify else {"sent": []}
        return {"ok": True, "as_of": self.as_of, "alerts": len(res["sent"])}

    # ------------------------------------------------------------------ alerts
    def check_user(self, user_id: str) -> dict:
        st = self.store()
        u = st.get_user(user_id)
        if u is None:
            return {"sent": [], "skipped": []}
        fc = self.forecast_now()
        demo = self.demo_enabled() and self.demo_user() == user_id
        return proximity.check_user(st, self.messenger, fc, self.zones, u, memo=self.memo, demo=demo,
                                    ignore_quiet=demo)

    def check_everyone(self) -> dict:
        st = self.store()
        fc = self.forecast_now()
        demo_user = self.demo_user() if self.demo_enabled() else None
        return proximity.check_everyone(st, self.messenger, fc, self.zones, memo=self.memo,
                                        ignore_quiet=bool(demo_user), demo_user=demo_user)

    # ------------------------------------------------------------------ Grok, voice, lab
    def capabilities(self) -> dict:
        return {"grok": xai.configured(), "model": xai.model(), "voice": xai.voice(),
                "daytona": lab.DaytonaRunner.available(), "pixels": bool(pixels.list_scenes()), "nessie": nessie.configured(),
                "vision": xai.configured()}

    def rate_ok(self, limit: int = 20, per_s: int = 60) -> bool:
        """At most `limit` Grok/voice calls per minute, so a stuck button cannot burn the xAI credits."""
        now = time.time()
        with self.lock:
            while self._hits and now - self._hits[0] > per_s:
                self._hits.popleft()
            if len(self._hits) >= limit:
                return False
            self._hits.append(now)
            return True

    def heat_overlay(self, prefer: str = "auto", days=None, view: str = "rule") -> dict:
        """Meta for the heat image of the forecast being shown. The PNG itself is served at meta['url'].
        view='ml' colours each area by the ML model's forecast for `days` ahead instead of the latest reading."""
        if prefer not in ("auto", "pixels", "interpolated"):
            prefer = "auto"
        fc = self.forecast_now(days)
        ml_view = view == "ml" and any("ml" in z for z in fc["zones"])
        if ml_view:     # the ML forecast is one number per area, so it is always the smooth blend, never real pixels
            prefer = "interpolated"
            fc = {**fc, "zones": [{**z, "last_obs": z["ml"]["display"]} if "ml" in z else {**z, "last_obs": None}
                                  for z in fc["zones"]]}
        key = (f"{fc['as_of']}|{prefer}|{len(pixels.list_scenes())}|{sum(1 for z in fc['zones'] if z.get('last_obs') is not None)}"
               + (f"|ml{fc['horizon_days']}" if ml_view else ""))
        with self.lock:
            meta = self._heat_meta.get(key)
        if meta is None:
            png, meta = heat.heat_overlay(fc, self.zones, self.labels, self.grid_x, self.grid_y, prefer)
            if ml_view:
                meta["note"] = (f"ML forecast for {fc['target_date']} ({fc['horizon_days']} days ahead), blended between areas. "
                                "Colours are shifted so the model's own alert cutoff sits on the flag line.")
            token = hashlib.md5(key.encode()).hexdigest()[:12]
            meta["url"] = f"/api/heat.png?k={token}"
            with self.lock:
                if len(self._heat_meta) > 60:
                    self._heat_meta.clear()
                    self._heat_png.clear()
                self._heat_meta[key] = meta
                self._heat_png[token] = png
        return meta

    def heat_png(self, token: str) -> Optional[bytes]:
        with self.lock:
            return self._heat_png.get(token)

    def chat(self, message: str, history: list) -> dict:
        return xai.answer(message, self.state(), history)

    def lab_start(self, mode: str) -> str:
        running = self.jobs.running("lab")
        if running:
            return running
        source = self.forecast_now()["data_source"]
        return self.jobs.start("lab", lambda job: lab.run_lab(job, mode="full" if mode == "full" else "quick", data_source=source))

    def train_start(self) -> str:
        """Daytona searches for the best model for each day ahead (1-7); the winners are then used by the forecast."""
        running = self.jobs.running("train")
        if running:
            return running
        source = self.forecast_now()["data_source"]

        def work(job):
            out = horizons.run_training(job, data_source=source)
            with self.lock:
                self.inputs.reload_horizons()
                self._fc_cache.clear()
                self._heat_meta.clear()
                self._heat_png.clear()
            return out

        return self.jobs.start("train", work)

    def ask_start(self, question: str) -> str:
        th = self.forecast_now()["thresholds"]
        return self.jobs.start("ask", lambda job: ask.run_ask(job, question, th))

    def job_view(self, jid: str) -> Optional[dict]:
        j = self.jobs.get(jid)
        if j is None:
            return None
        return {"id": j["id"], "kind": j["kind"], "status": j["status"], "done": j["done"], "total": j["total"],
                "runner": j["runner"], "log": j["log"][-8:], "result": j["result"], "error": j["error"],
                "seconds": round((j["finished"] or time.time()) - j["started"], 1)}

    # ------------------------------------------------------------------ API payloads
    def state(self, days=None) -> dict:
        fc = self.forecast_now(days)
        st = self.store()
        bundle = self.inputs.bundle
        report = bundle.get("report") or {}
        by = (report.get("by_method") or {}).get("model" if fc["method"] == "model" else "persistence") or {}
        uid = self.demo_user()
        pin = None
        if uid:
            u = st.get_user(uid)
            if u is not None and u["lat"] is not None and u["loc_source"] == "pin":
                pin = {"lat": u["lat"], "lon": u["lon"]}
        zones_out = []
        for z in fc["zones"]:
            zones_out.append({
                "id": z["id"], "number": z["number"], "name": z["name"], "lat": z["lat"], "lon": z["lon"],
                "tier": z["tier"], "flagged": z["tier"] == "high" and not z.get("low_confidence"), "last_obs": z.get("last_obs"),
                "low_confidence": z.get("low_confidence", False), "rule_tier": z.get("rule_tier"), "ml": z.get("ml"),
                "obs_age_days": z.get("obs_age_days"), "track": z.get("track"), "reason": z.get("reason"),
                "verification": z.get("verification"), "anchor": list(self.anchors[z["id"]]),
                "site": self.zone_by_id[z["id"]].get("site"), "site_name": self.zone_by_id[z["id"]].get("site_name"),
                "inputs": z.get("inputs"),
            })
        return {
            "mode": "replay" if fc["replay"] else "live", "as_of": fc["as_of"], "target_date": fc["target_date"],
            "data_source": fc["data_source"], "method": fc["method"], "in_season": fc["in_season"],
            "thresholds": fc["thresholds"], "weather": fc.get("weather", {}), "zones": zones_out,
            "reliability": {"precision": by.get("precision"), "recall": by.get("recall"),
                            "test_years": report.get("test_years"), "n_high_dates": report.get("n_actual_high_dates"),
                            "n_high": report.get("n_actual_high")},
            "demo": {"enabled": self.demo_enabled(), "user_id": uid, "pin": pin, "radius_m": config.PROXIMITY_RADIUS_M},
            "photon_connected": self.messenger is not None, "official_info": config.OFFICIAL_INFO,
            "capabilities": self.capabilities(), "photo_counts": st.photo_counts(),
            "horizon": fc["horizon_days"], "ml": fc.get("ml", {"available": False}),
            "ml_days": sorted(int(k) for k in (self.inputs.horizon_models or {})),
            "skill": self.skill(), "late_season": pd.Timestamp(fc["target_date"]).month >= 10,
        }

    def skill(self) -> list:
        """Measured score per day ahead (from the last 1-7 day training), for the slider and Science mode."""
        s = (horizons.latest().get("summary") or {}).get("days") or []
        return [{k: d.get(k) for k in ("days", "rule_f1", "ml_f1", "ml_name", "use_ml")} for d in s]

    def zones_geo(self) -> list:
        return [{"id": z["id"], "number": z["number"], "name": z["name"], "lat": z["lat"], "lon": z["lon"],
                 "site": z.get("site"), "site_name": z.get("site_name"),
                 "anchor": list(self.anchors[z["id"]]), "points": z["points"]} for z in self.zones]

    # ------------------------------------------------------------------ science mode
    def science(self) -> dict:
        """Everything behind the forecast, for people who want to check it: scorecard, errors, inputs, experiments."""
        bundle = self.inputs.bundle
        rep_ = bundle.get("report") or {}
        thr = {"elevated": bundle["thr_elev"], "high": bundle["thr_high"]}
        out = {"method": bundle["method"], "method_reason": bundle["method_reason"], "model_name": bundle.get("model_name"),
               "horizon": bundle.get("horizon"), "trained_rows": bundle.get("trained_rows"),
               "trained_through": bundle.get("trained_through"), "data_source": bundle.get("data_source"),
               "thresholds": thr, "report": rep_, "tier_record": bundle.get("tier_record"),
               "features": [{"id": f, "label": labcore.FEATURE_LABELS.get(f, f)} for f in bundle.get("features", [])]}
        site_of = {z["id"]: z.get("site", config.DEFAULT_SITE) for z in self.zones}
        try:
            oos = pd.read_csv(config.DATA_DIR / "oos_predictions.csv")
            sample = oos.sample(min(900, len(oos)), random_state=0) if len(oos) else oos
            out["scatter_total"] = int(len(oos))
            out["scatter"] = [{"y": round(float(r.y), 4), "model": round(float(r.pred_model), 4), "persist": round(float(r.pred_persist), 4),
                               "site": site_of.get(r.zone_id, config.DEFAULT_SITE), "high": round(float(r.thr_high), 4)}
                              for r in sample.itertuples()]
        except (OSError, ValueError, AttributeError):
            out["scatter"] = []
        obs = self.inputs.obs
        pop = population.load()
        sites = []
        for s_ in config.SITES:
            o = obs[obs["site"] == s_["key"]]
            if o.empty:
                continue
            sites.append({"key": s_["key"], "name": s_["name"], "kind": s_.get("kind"), "zones": int(o["zone_id"].nunique()),
                          "readings": int(len(o)), "dates": int(o["date"].nunique()),
                          "first": str(o["date"].min().date()), "last": str(o["date"].max().date()),
                          "median": round(float(o["ndci"].median()), 4),
                          "flagged_share": round(float((o["ndci"] >= thr["high"]).mean()), 3),
                          "people": (pop.get("sites") or {}).get(s_["key"], {}).get("people")})
        out["sites"] = sites
        out["population_note"] = pop.get("note")
        latest = lab.latest()
        hz = horizons.latest()
        out["horizons"] = ({"finished_at": hz.get("finished_at"), "runner": hz.get("runner"), "n": hz.get("n"),
                            "holdout": hz.get("holdout"), "summary": hz.get("summary"), "notes": hz.get("notes")} if hz else None)
        out["lab"] = {"finished_at": latest.get("finished_at"), "runner": latest.get("runner"), "mode": latest.get("mode"),
                      "holdout": latest.get("holdout"), "summary": latest.get("summary")} if latest else None
        return out

    # ------------------------------------------------------------------ cleanup planner
    def plan(self, site: str, budget=None, lbs=None) -> dict:
        if site not in config.SITE_BY_KEY:
            raise ValueError("unknown water body")
        facts = planner.site_facts(site, self.zones, population.load(), self.inputs.obs, self.inputs.bundle["thr_high"])
        out = planner.plan(facts, None if budget in (None, "") else float(budget), None if lbs in (None, "") else float(lbs))
        try:
            out["ledger"] = nessie.Ledger().status(site)
        except nessie.NessieError as e:
            out["ledger"] = {"error": str(e), "label": nessie.Ledger().label}
        return out

    def plan_sites(self) -> list:
        have = {z.get("site", config.DEFAULT_SITE) for z in self.zones}
        return [{"key": s_["key"], "name": s_["name"], "kind": s_.get("kind")} for s_ in config.SITES if s_["key"] in have]

    def businesses(self, site: str) -> dict:
        zs = [z for z in self.zones if z.get("site", config.DEFAULT_SITE) == site]
        if not zs:
            raise ValueError("unknown water body")
        lat = sum(z["lat"] for z in zs) / len(zs)
        lon = sum(z["lon"] for z in zs) / len(zs)
        return nessie.merchants_near(lat, lon, miles=1.0)

    def plan_ask(self, site: str, question: str, budget=None) -> dict:
        p = self.plan(site, budget)
        return planner.answer(question, p, chat=xai.chat if xai.configured() else None)

    def history(self, zone_id: str) -> dict:
        fc = self.forecast_now()
        as_of = pd.Timestamp(fc["as_of"])
        obs = self.inputs.obs
        z = obs[obs["zone_id"] == zone_id]
        before = z[(z["date"] <= as_of) & (z["date"] > as_of - pd.Timedelta(days=700))]
        out = {"zone": zone_id, "name": self.zone_by_id[zone_id]["name"], "as_of": fc["as_of"],
               "thresholds": fc["thresholds"],
               "readings": [{"d": str(r.date.date()), "v": round(float(r.ndci), 4)} for r in before.itertuples()]}
        if fc["replay"]:
            after = z[(z["date"] > as_of) & (z["date"] <= as_of + pd.Timedelta(days=21))]
            out["after"] = [{"d": str(r.date.date()), "v": round(float(r.ndci), 4)} for r in after.itertuples()]
        return out

    def frames(self) -> dict:
        obs = self.inputs.obs
        years = sorted({int(y) for y in obs["date"].dt.year.unique()})
        daily = obs.groupby("date")["ndci"].median()
        peak = daily.idxmax()
        return {"years": years, "min": str(obs["date"].min().date()), "max": str(obs["date"].max().date()),
                "suggested": str((peak + pd.Timedelta(days=config.OBS_LATENCY_DAYS)).date()),
                "today": str(date.today())}

    def feed(self, limit: int = 15) -> list:
        return [{"ts": r["ts"], "kind": r["kind"], "to": r["user_label"], "zone": r["zone_name"], "distance_m": r["distance_m"],
                 "text": r["text"], "delivered": bool(r["delivered"]), "note": r["note"], "simulated": bool(r["replay"])}
                for r in self.store().recent_feed(limit)]

    def photos_for(self, zone_id: str) -> list:
        """The public gallery for one area: verified photos only."""
        return [{"id": r["id"], "at": r["received_at"], "status": r["status"], "bloom": r["bloom"], "description": r["description"]}
                for r in self.store().photos_for_zone(zone_id)]

    # ------------------------------------------------------------------ photo moderation (this computer only)
    def photo_queue(self, status: str) -> dict:
        statuses = ("pending", "unverified") if status == "pending" else (status,)
        if status not in store_mod.PHOTO_STATUSES:
            raise ValueError("unknown status")
        rows = self.store().photos_with_status(statuses, limit=80)
        return {"counts": self.store().photo_counts(), "photos": [{
            "id": r["id"], "at": r["received_at"], "status": r["status"], "zone": r["zone_id"],
            "zone_name": self.zone_by_id.get(r["zone_id"] or "", {}).get("name"),
            "site_name": self.zone_by_id.get(r["zone_id"] or "", {}).get("site_name"),
            "reason": r["reason"], "bloom": r["bloom"], "description": r["description"], "decided_by": r["decided_by"],
            "from": mask_handle((self.store().get_user(r["user_id"]) or {"handle": ""})["handle"])} for r in rows]}

    def moderate(self, photo_id: int, action: str, bloom: Optional[str] = None) -> dict:
        st = self.store()
        row = st.get_photo(photo_id)
        if row is None:
            raise KeyError("photo")
        if action == "delete":
            path = st.delete_photo(photo_id)
            if path and config.PHOTO_DIR.resolve() in Path(path).resolve().parents:
                Path(path).unlink(missing_ok=True)
            return {"ok": True, "deleted": photo_id, "counts": st.photo_counts()}
        status = {"verify": "verified", "reject": "rejected", "review": "review"}.get(action)
        if status is None:
            raise ValueError("unknown action")
        st.set_photo_status(photo_id, status, by="manual", reason=f"set by hand: {status}")
        if bloom in screening.BLOOM_LABELS:
            st.db.execute("UPDATE photos SET bloom=? WHERE id=?", (bloom, photo_id))
            st.db.commit()
        return {"ok": True, "id": photo_id, "status": status, "counts": st.photo_counts()}

    def screen_start(self) -> str:
        running = self.jobs.running("photos")
        if running:
            return running

        def work(job):
            job["log"].append("asking Grok about each pending photo")
            st = Store(self.db_path)          # this job's own connection (it runs on another thread)
            return photos.screen_pending(st, self.zones)

        return self.jobs.start("photos", work)

    def stories(self, days: int = 7) -> list:
        """Verified photos from the last `days` days, grouped by area, for the story bubbles on the map."""
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
        groups = {}
        for r in self.store().verified_since(since):
            z = self.zone_by_id.get(r["zone_id"])
            if z is None:
                continue
            g = groups.setdefault(z["id"], {"zone": z["id"], "name": z["name"], "site_name": z.get("site_name"),
                                            "anchor": list(self.anchors[z["id"]]), "photos": []})
            g["photos"].append({"id": r["id"], "at": r["received_at"], "bloom": r["bloom"],
                                "label": screening.BLOOM_LABELS.get(r["bloom"] or "unclear"), "description": r["description"]})
        out = list(groups.values())
        for g in out:
            g["bloom"] = any(p["bloom"] == "bloom" for p in g["photos"])
        return sorted(out, key=lambda g: (not g["bloom"], g["zone"]))


# ---------------------------------------------------------------------- HTTP
def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Algaebro"

        def log_message(self, *args):  # keep the terminal readable
            pass

        # ---- helpers
        def _send(self, code: int, body: bytes, ctype: str, extra: Optional[dict] = None) -> None:
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json; charset=utf-8")

        def _raw(self, limit: int = 6_000_000) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(min(max(n, 0), limit))

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return {}
            try:
                return json.loads(self.rfile.read(min(n, 1_000_000)))
            except ValueError:
                return {}

        # ---- GET
        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    return self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
                if u.path.startswith("/static/"):
                    name = u.path[len("/static/"):]
                    if not re.fullmatch(r"[\w.-]+\.(js|css|svg|woff2)", name) or not (WEB_DIR / name).is_file():
                        return self._json({"error": "not found"}, 404)
                    ext = name.rsplit(".", 1)[1]
                    ctype = {"js": "text/javascript; charset=utf-8", "css": "text/css; charset=utf-8",
                             "svg": "image/svg+xml", "woff2": "font/woff2"}[ext]
                    return self._send(200, (WEB_DIR / name).read_bytes(), ctype)
                if u.path == "/api/state":
                    return self._json(app.state(q.get("days")))
                if u.path == "/api/heat":
                    return self._json(app.heat_overlay(q.get("kind", "auto"), q.get("days"), q.get("view", "rule")))
                if u.path == "/api/train/latest":
                    return self._json(horizons.latest())
                if u.path == "/api/heat.png":
                    png = app.heat_png(q.get("k", ""))
                    if png is None:
                        return self._json({"error": "not found"}, 404)
                    return self._send(200, png, "image/png")
                if u.path == "/api/lab/latest":
                    return self._json(lab.latest())
                if u.path == "/api/photos/queue":
                    try:
                        return self._json(app.photo_queue(q.get("status", "review")))
                    except ValueError as e:
                        return self._json({"error": str(e)}, 400)
                if u.path == "/api/stories":
                    return self._json(app.stories())
                if u.path in ("/api/lab/status", "/api/ask/status", "/api/photos/status", "/api/train/status"):
                    view = app.job_view(q.get("id", ""))
                    return self._json(view or {"error": "unknown job"}, 200 if view else 404)
                if u.path == "/api/zones":
                    return self._json(app.zones_geo())
                if u.path == "/api/history":
                    zid = q.get("zone", "")
                    if zid not in app.zone_by_id:
                        return self._json({"error": "unknown zone"}, 404)
                    return self._json(app.history(zid))
                if u.path == "/api/frames":
                    return self._json(app.frames())
                if u.path == "/api/science":
                    return self._json(app.science())
                if u.path == "/api/plan/sites":
                    return self._json(app.plan_sites())
                if u.path == "/api/plan":
                    try:
                        return self._json(app.plan(q.get("site", ""), q.get("budget"), q.get("lbs")))
                    except ValueError as e:
                        return self._json({"error": str(e)}, 400)
                if u.path == "/api/plan/businesses":
                    try:
                        return self._json(app.businesses(q.get("site", "")))
                    except ValueError as e:
                        return self._json({"error": str(e)}, 400)
                if u.path == "/api/feed":
                    return self._json(app.feed())
                if u.path == "/api/users":
                    return self._json(app.users())
                if u.path == "/api/photos":
                    return self._json(app.photos_for(q.get("zone", "")))
                m = re.fullmatch(r"/photos/(\d+)\.jpg", u.path)
                if m:
                    row = app.store().get_photo(int(m.group(1)))
                    path = Path(row["path"]).resolve() if row else None
                    if row is None or not path.is_file() or config.PHOTO_DIR.resolve() not in path.parents:
                        return self._json({"error": "not found"}, 404)
                    return self._send(200, path.read_bytes(), "image/jpeg")
                return self._json({"error": "not found"}, 404)
            except Exception as e:  # never kill the server over one bad request
                print(f"[server] GET {u.path} failed: {type(e).__name__}: {str(e)[:200]}")
                return self._json({"error": "server error"}, 500)

        # ---- POST
        def do_POST(self):
            u = urlparse(self.path)
            raw_audio = self._raw() if u.path == "/api/stt" else None     # binary: must not be parsed (or consumed) as JSON
            b = {} if raw_audio is not None else self._body()
            try:
                if u.path == "/api/as_of":
                    return self._json(app.set_as_of(b.get("as_of"), notify=bool(b.get("notify", True))))
                if u.path == "/api/demo":
                    r = app.set_demo(bool(b.get("enabled")), b.get("user_id"))
                    return self._json(r, 200 if r["ok"] else 409)
                if u.path == "/api/pin":
                    r = app.set_pin(b["lat"], b["lon"])
                    return self._json(r, 200 if r["ok"] else 409)
                if u.path == "/api/chat":
                    if not app.rate_ok():
                        return self._json({"error": "Slow down a little: too many questions in the last minute."}, 429)
                    hist = b.get("history") if isinstance(b.get("history"), list) else []
                    return self._json(app.chat(str(b.get("message", "")), hist))
                if u.path == "/api/tts":
                    if not app.rate_ok():
                        return self._json({"error": "Too many requests in the last minute."}, 429)
                    audio, ctype = xai.speak(str(b.get("text", "")))
                    return self._send(200, audio, ctype)
                if u.path == "/api/stt":
                    if not app.rate_ok():
                        return self._json({"error": "Too many requests in the last minute."}, 429)
                    raw = raw_audio
                    if not raw:
                        return self._json({"error": "No audio received."}, 400)
                    ctype = (self.headers.get("X-Audio-Type") or "audio/webm").split(";")[0]
                    ext = {"audio/webm": "webm", "audio/mp4": "m4a", "audio/ogg": "ogg", "audio/wav": "wav", "audio/mpeg": "mp3"}.get(ctype, "webm")
                    return self._json({"text": xai.transcribe(raw, f"speech.{ext}", ctype)})
                if u.path in ("/api/plan/fund", "/api/plan/spend"):
                    site = str(b.get("site", ""))
                    if site not in config.SITE_BY_KEY:
                        return self._json({"error": "unknown water body"}, 400)
                    ledger = nessie.Ledger()
                    try:
                        if u.path == "/api/plan/fund":
                            return self._json(ledger.fund(site, float(b["amount"])))
                        opt = planner.OPTION_BY_ID.get(str(b.get("option", "")))
                        if opt is None:
                            return self._json({"error": "unknown option"}, 400)
                        return self._json(ledger.spend(site, float(b["amount"]), f"{opt['name']} ({config.SITE_BY_KEY[site]['name']})"))
                    except nessie.NessieError as e:
                        return self._json({"error": str(e)}, 502)
                    except ValueError as e:
                        return self._json({"error": str(e)}, 409)
                if u.path == "/api/photos/moderate":
                    try:
                        return self._json(app.moderate(int(b["id"]), str(b.get("action", "")), b.get("bloom")))
                    except KeyError:
                        return self._json({"error": "no such photo"}, 404)
                if u.path == "/api/photos/screen":
                    if not xai.configured():
                        return self._json({"error": "Grok is not set up: add XAI_API_KEY to .env and restart."}, 503)
                    return self._json({"job_id": app.screen_start()})
                if u.path == "/api/plan/ask":
                    if not app.rate_ok():
                        return self._json({"error": "Too many requests in the last minute."}, 429)
                    try:
                        return self._json(app.plan_ask(str(b.get("site", "")), str(b.get("question", "")), b.get("budget")))
                    except ValueError as e:
                        return self._json({"error": str(e)}, 400)
                if u.path == "/api/train/run":
                    return self._json({"job_id": app.train_start()})
                if u.path == "/api/lab/run":
                    return self._json({"job_id": app.lab_start(str(b.get("mode", "quick")))})
                if u.path == "/api/ask":
                    if not lab.DaytonaRunner.available():
                        return self._json({"error": "Ask-the-data needs Daytona (DAYTONA_API_KEY and the 'daytona' package). "
                                                    "AI-written code is never run on this computer."}, 503)
                    if not xai.configured():
                        return self._json({"error": "Grok is not set up: add XAI_API_KEY to .env and restart."}, 503)
                    if not app.rate_ok(limit=10):
                        return self._json({"error": "Too many requests in the last minute."}, 429)
                    return self._json({"job_id": app.ask_start(str(b.get("question", "")))})
                return self._json({"error": "not found"}, 404)
            except xai.XAIError as e:
                code = 503 if not xai.configured() else 502
                return self._json({"error": str(e)}, code)
            except (KeyError, ValueError, TypeError) as e:
                return self._json({"error": f"bad request: {type(e).__name__}"}, 400)
            except Exception as e:
                print(f"[server] POST {u.path} failed: {type(e).__name__}: {str(e)[:200]}")
                return self._json({"error": "server error"}, 500)

    return Handler


# ---------------------------------------------------------------------- Photon loop
def photon_loop(app: App, stop: threading.Event) -> None:
    """Poll Photon, answer texts, and check proximity every so often. Runs in its own thread."""
    store = app.store()                      # this thread's own connection
    bot = Bot(store, app.forecast_now, app.zones)
    last_check = time.time()
    while not stop.is_set():
        try:
            for inc in app.messenger.poll():
                handle_incoming(
                    store, bot, app.messenger, inc,
                    on_new_subscription=app.check_user,
                    photo_handler=lambda i: photos.handle_photo_message(store, app.messenger, app.zones, i),
                )
        except Exception as e:
            print(f"[bot] poll error: {describe_error(e)}")
        if time.time() - last_check > config.ALERT_CHECK_MINUTES * 60:
            try:
                app.check_everyone()
            except Exception as e:
                print(f"[alerts] error: {describe_error(e)}")
            last_check = time.time()
        stop.wait(config.POLL_SECONDS)


def build_server(app: App, port: int = 8765, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer.allow_reuse_address = True
    return ThreadingHTTPServer((host, port), make_handler(app))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--as-of", type=forecast.parse_as_of, help="start in REPLAY mode at this date (YYYY-MM-DD)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-photon", action="store_true", help="dashboard only: nothing is texted")
    ap.add_argument("--base-url", help="override the Photon API address (testing)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    load_dotenv(config.ROOT / ".env")
    messenger = None
    if not args.no_photon:
        try:
            from .photon_messenger import PhotonMessenger

            messenger = PhotonMessenger.from_env(Store(), base_url=args.base_url)
        except RuntimeError as e:
            print(f"[server] Photon not connected ({e}). Running dashboard only.")

    app = App(messenger=messenger, as_of=args.as_of)
    fc = app.forecast_now()
    print(f"Algaebro: data={fc['data_source']} method={fc['method']} as_of={fc['as_of']}{' (REPLAY)' if fc['replay'] else ''} "
          f"photon={'connected' if messenger else 'off'}")
    stop = threading.Event()
    if messenger is not None:
        threading.Thread(target=photon_loop, args=(app, stop), daemon=True).start()

    try:
        httpd = build_server(app, args.port)
    except OSError as e:
        raise SystemExit(f"Could not start on port {args.port} ({e}). Try --port {args.port + 1}.")
    url = f"http://localhost:{args.port}/"
    print(f"Dashboard: {url}   (press Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stop.set()


if __name__ == "__main__":
    main()
