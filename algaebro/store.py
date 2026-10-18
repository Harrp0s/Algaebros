"""Tiny SQLite store: who is subscribed to what, who was alerted when, and which messages we handled.

Stores phone numbers / handles you receive from Photon, so treat data/algaebro.db as private
and do not commit or share it.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  user_id TEXT PRIMARY KEY, handle TEXT, space_id TEXT, agent_id TEXT,
  lat REAL, lon REAL, created_at TEXT, last_seen TEXT);
CREATE TABLE IF NOT EXISTS subs(
  user_id TEXT, zone_id TEXT, source TEXT, PRIMARY KEY(user_id, zone_id));
CREATE TABLE IF NOT EXISTS alerts(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, zone_id TEXT, sent_at TEXT, tier TEXT, target_date TEXT);
CREATE TABLE IF NOT EXISTS seen(message_id TEXT PRIMARY KEY, seen_at TEXT);
CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS reports(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, zone_id TEXT, note TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS feed(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT, user_label TEXT, zone_id TEXT, zone_name TEXT,
  distance_m REAL, text TEXT, delivered INTEGER, note TEXT, replay INTEGER);
CREATE TABLE IF NOT EXISTS photos(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, zone_id TEXT, lat REAL, lon REAL,
  received_at TEXT, path TEXT, attachment_id TEXT, status TEXT DEFAULT 'unverified');
"""

USER_COLUMNS = {"loc_source": "TEXT", "loc_at": "TEXT", "alerts_off": "INTEGER DEFAULT 0"}
# Photo screening (screening.py). status: pending -> verified | rejected | review ('unverified' = from before screening).
PHOTO_COLUMNS = {"sha256": "TEXT", "reason": "TEXT", "bloom": "TEXT", "description": "TEXT", "screened_at": "TEXT", "decided_by": "TEXT"}
PHOTO_STATUSES = ("pending", "verified", "rejected", "review")


def mask_handle(handle: str) -> str:
    """Show only the last 4 digits of a phone number anywhere it could be seen by others."""
    digits = "".join(c for c in (handle or "") if c.isdigit())
    return f"+\u2022\u2022\u2022{digits[-4:]}" if len(digits) >= 4 else "unknown"


@dataclass
class UserCtx:
    """Who is talking to us and how to reach them (Photon needs the space id + agent id to message)."""
    user_id: str
    handle: str = ""
    space_id: str = ""
    agent_id: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path=None):
        self.path = str(path or config.DB_PATH)
        # One connection per thread is the safe pattern (see server.py); WAL + a busy timeout let them share the file.
        self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=15)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
        self.db.executescript(SCHEMA)
        have = {r["name"] for r in self.db.execute("PRAGMA table_info(users)").fetchall()}
        for col, decl in USER_COLUMNS.items():
            if col not in have:
                self.db.execute(f"ALTER TABLE users ADD COLUMN {col} {decl}")
        have = {r["name"] for r in self.db.execute("PRAGMA table_info(photos)").fetchall()}
        for col, decl in PHOTO_COLUMNS.items():
            if col not in have:
                self.db.execute(f"ALTER TABLE photos ADD COLUMN {col} {decl}")
        self.db.commit()

    # users -----------------------------------------------------------
    def upsert_user(self, ctx: UserCtx) -> None:
        now = _now()
        self.db.execute(
            """INSERT INTO users(user_id, handle, space_id, agent_id, created_at, last_seen)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(user_id) DO UPDATE SET
                 handle=excluded.handle,
                 space_id=CASE WHEN excluded.space_id!='' THEN excluded.space_id ELSE users.space_id END,
                 agent_id=CASE WHEN excluded.agent_id!='' THEN excluded.agent_id ELSE users.agent_id END,
                 last_seen=excluded.last_seen""",
            (ctx.user_id, ctx.handle, ctx.space_id, ctx.agent_id, now, now),
        )
        self.db.commit()

    def set_location(self, user_id: str, lat: float, lon: float, source: str = "shared") -> None:
        self.db.execute("UPDATE users SET lat=?, lon=?, loc_source=?, loc_at=? WHERE user_id=?",
                        (lat, lon, source, _now(), user_id))
        self.db.commit()

    def set_alerts_off(self, user_id: str, off: bool) -> None:
        self.db.execute("UPDATE users SET alerts_off=? WHERE user_id=?", (1 if off else 0, user_id))
        self.db.commit()

    def list_users(self) -> List[sqlite3.Row]:
        return self.db.execute("SELECT * FROM users ORDER BY last_seen DESC").fetchall()

    def get_user(self, user_id: str) -> Optional[sqlite3.Row]:
        return self.db.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()

    # subscriptions ---------------------------------------------------
    def add_sub(self, user_id: str, zone_id: str, source: str = "manual") -> None:
        # A manual follow always wins; a location-derived follow never overwrites a manual one.
        verb = "INSERT OR REPLACE" if source == "manual" else "INSERT OR IGNORE"
        self.db.execute(f"{verb} INTO subs(user_id, zone_id, source) VALUES(?,?,?)", (user_id, zone_id, source))
        self.db.commit()

    def clear_subs(self, user_id: str, source: Optional[str] = None) -> int:
        if source:
            cur = self.db.execute("DELETE FROM subs WHERE user_id=? AND source=?", (user_id, source))
        else:
            cur = self.db.execute("DELETE FROM subs WHERE user_id=?", (user_id,))
        self.db.commit()
        return cur.rowcount

    def get_subs(self, user_id: str) -> List[str]:
        rows = self.db.execute("SELECT zone_id FROM subs WHERE user_id=? ORDER BY zone_id", (user_id,)).fetchall()
        return [r["zone_id"] for r in rows]

    def subscribers_for_zone(self, zone_id: str) -> List[sqlite3.Row]:
        return self.db.execute(
            "SELECT u.* FROM users u JOIN subs s ON s.user_id=u.user_id WHERE s.zone_id=?", (zone_id,)
        ).fetchall()

    # alerts ----------------------------------------------------------
    def last_alert(self, user_id: str, zone_id: str) -> Optional[datetime]:
        r = self.db.execute(
            "SELECT sent_at FROM alerts WHERE user_id=? AND zone_id=? ORDER BY id DESC LIMIT 1", (user_id, zone_id)
        ).fetchone()
        return datetime.fromisoformat(r["sent_at"]) if r else None

    def log_alert(self, user_id: str, zone_id: str, tier: str, target_date: str, when: Optional[datetime] = None) -> None:
        """Record an alert. `when` should be the same clock the alert engine used for its cooldown check."""
        stamp = (when or datetime.now(timezone.utc)).isoformat(timespec="seconds")
        self.db.execute(
            "INSERT INTO alerts(user_id, zone_id, sent_at, tier, target_date) VALUES(?,?,?,?,?)",
            (user_id, zone_id, stamp, tier, target_date),
        )
        self.db.commit()

    # message bookkeeping ---------------------------------------------
    def already_seen(self, message_id: str) -> bool:
        return self.db.execute("SELECT 1 FROM seen WHERE message_id=?", (message_id,)).fetchone() is not None

    def mark_seen(self, message_id: str) -> None:
        self.db.execute("INSERT OR IGNORE INTO seen(message_id, seen_at) VALUES(?,?)", (message_id, _now()))
        self.db.commit()

    def kv_get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        r = self.db.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return r["v"] if r else default

    def kv_set(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO kv(k, v) VALUES(?,?)", (key, value))
        self.db.commit()

    # citizen reports -------------------------------------------------
    def add_report(self, user_id: str, zone_id: Optional[str], note: str) -> None:
        self.db.execute(
            "INSERT INTO reports(user_id, zone_id, note, created_at) VALUES(?,?,?,?)", (user_id, zone_id or "", note[:500], _now())
        )
        self.db.commit()

    # alert feed (what the dashboard shows) ---------------------------
    def log_feed(self, kind: str, user_handle: str, zone_id: str, zone_name: str, distance_m, text: str,
                 delivered: bool, note: str = "", replay: bool = False, when: Optional[datetime] = None) -> None:
        stamp = (when or datetime.now(timezone.utc)).isoformat(timespec="seconds")
        self.db.execute(
            "INSERT INTO feed(ts, kind, user_label, zone_id, zone_name, distance_m, text, delivered, note, replay) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (stamp, kind, mask_handle(user_handle), zone_id, zone_name, distance_m, text, 1 if delivered else 0, note, 1 if replay else 0),
        )
        self.db.commit()

    def recent_feed(self, limit: int = 20) -> List[sqlite3.Row]:
        return self.db.execute("SELECT * FROM feed ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

    # photos ----------------------------------------------------------
    def add_photo(self, user_id: str, zone_id: Optional[str], lat, lon, path: str, attachment_id: str = "",
                  sha256: Optional[str] = None) -> int:
        cur = self.db.execute(
            "INSERT INTO photos(user_id, zone_id, lat, lon, received_at, path, attachment_id, status, sha256) VALUES(?,?,?,?,?,?,?,?,?)",
            (user_id, zone_id, lat, lon, _now(), path, attachment_id, "pending", sha256),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def photos_for_zone(self, zone_id: str, limit: int = 12) -> List[sqlite3.Row]:
        """What the public gallery shows: verified photos only."""
        return self.db.execute("SELECT id, zone_id, received_at, status, bloom, description FROM photos "
                               "WHERE zone_id=? AND status='verified' ORDER BY id DESC LIMIT ?", (zone_id, limit)).fetchall()

    def photo_hash_known(self, sha256: str) -> bool:
        return self.db.execute("SELECT 1 FROM photos WHERE sha256=?", (sha256,)).fetchone() is not None

    def photos_since(self, user_id: str, since_iso: str) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM photos WHERE user_id=? AND received_at>=?", (user_id, since_iso)).fetchone()[0])

    def set_photo_review(self, photo_id: int, status: str, reason: str, bloom: str, description: str, by: str) -> None:
        if status not in PHOTO_STATUSES:
            raise ValueError(f"unknown photo status {status!r}")
        self.db.execute("UPDATE photos SET status=?, reason=?, bloom=?, description=?, screened_at=?, decided_by=? WHERE id=?",
                        (status, reason, bloom, description, _now(), by, photo_id))
        self.db.commit()

    def set_photo_status(self, photo_id: int, status: str, by: str = "manual", reason: Optional[str] = None) -> None:
        if status not in PHOTO_STATUSES:
            raise ValueError(f"unknown photo status {status!r}")
        self.db.execute("UPDATE photos SET status=?, decided_by=?, reason=COALESCE(?, reason), screened_at=? WHERE id=?",
                        (status, by, reason, _now(), photo_id))
        self.db.commit()

    def photos_with_status(self, statuses, limit: int = 60) -> List[sqlite3.Row]:
        marks = ",".join("?" * len(statuses))
        return self.db.execute(f"SELECT * FROM photos WHERE status IN ({marks}) ORDER BY id DESC LIMIT ?", (*statuses, limit)).fetchall()

    def photo_counts(self) -> dict:
        rows = self.db.execute("SELECT status, COUNT(*) AS n FROM photos GROUP BY status").fetchall()
        out = {s: 0 for s in PHOTO_STATUSES}
        for r in rows:
            key = "pending" if r["status"] in (None, "unverified") else r["status"]
            out[key] = out.get(key, 0) + r["n"]
        return out

    def verified_since(self, since_iso: str) -> List[sqlite3.Row]:
        return self.db.execute("SELECT id, zone_id, received_at, bloom, description FROM photos WHERE status='verified' "
                               "AND zone_id IS NOT NULL AND zone_id<>'' AND received_at>=? ORDER BY id DESC", (since_iso,)).fetchall()

    def delete_photo(self, photo_id: int) -> Optional[str]:
        row = self.get_photo(photo_id)
        if row is None:
            return None
        self.db.execute("DELETE FROM photos WHERE id=?", (photo_id,))
        self.db.commit()
        return row["path"]

    def get_photo(self, photo_id: int) -> Optional[sqlite3.Row]:
        return self.db.execute("SELECT * FROM photos WHERE id=?", (photo_id,)).fetchone()

    def unplaced_photos(self, user_id: str) -> List[sqlite3.Row]:
        return self.db.execute("SELECT * FROM photos WHERE user_id=? AND (zone_id IS NULL OR zone_id='')", (user_id,)).fetchall()

    def place_photo(self, photo_id: int, zone_id: str, lat, lon) -> None:
        self.db.execute("UPDATE photos SET zone_id=?, lat=?, lon=? WHERE id=?", (zone_id, lat, lon, photo_id))
        self.db.commit()

    def delete_user_photos(self, user_id: str) -> List[str]:
        """Remove a person's photos from the database and return the file paths so the caller can delete the files."""
        rows = self.db.execute("SELECT path FROM photos WHERE user_id=?", (user_id,)).fetchall()
        self.db.execute("DELETE FROM photos WHERE user_id=?", (user_id,))
        self.db.commit()
        return [r["path"] for r in rows]

    def photo_attachment_known(self, attachment_id: str) -> bool:
        return self.db.execute("SELECT 1 FROM photos WHERE attachment_id=?", (attachment_id,)).fetchone() is not None
