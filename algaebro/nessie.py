"""Cleanup budgets on Capital One's Nessie API, plus businesses near a water body.

Nessie is Capital One's hackathon banking API: practice customers, accounts and transactions, and a shared list of
merchants with locations. We use it for two things:
  * a CLEANUP FUND per water body: one Nessie account whose balance is the budget; choosing a plan records it as a
    withdrawal, so the remaining budget is a real ledger anyone with the key can inspect;
  * BUSINESSES NEAR THE WATER: Nessie's merchant search by location. Nessie's merchants are shared sandbox data that
    hackers worldwide add to, so this is an illustration, not a census of local businesses (the dashboard says so).

Without NESSIE_API_KEY the same ledger is kept in data/budget_ledger.json and labelled "local ledger". Only budget
amounts and plan names are sent to Nessie: never phone numbers, photos or locations of people.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import date
from typing import Dict, List, Optional

import requests

from . import config

API = os.environ.get("NESSIE_BASE_URL", "https://api.nessieisreal.com")
IDS_JSON = config.DATA_DIR / "nessie_ids.json"          # which Nessie customer/account belongs to which site (no secrets)
LOCAL_JSON = config.DATA_DIR / "budget_ledger.json"
_lock = threading.Lock()


class NessieError(Exception):
    pass


def configured() -> bool:
    return bool(os.environ.get("NESSIE_API_KEY"))


def _key() -> Dict[str, str]:
    return {"key": os.environ.get("NESSIE_API_KEY", "")}


def _call(method: str, path: str, body: Optional[dict] = None, params: Optional[dict] = None, timeout: int = 20):
    try:
        r = requests.request(method, f"{API}{path}", params={**_key(), **(params or {})}, json=body, timeout=timeout)
    except requests.RequestException as e:
        raise NessieError(f"Could not reach Nessie ({type(e).__name__}).") from None
    if r.status_code in (401, 403):
        raise NessieError("Nessie refused the key. Check NESSIE_API_KEY in .env.")
    if r.status_code >= 400:
        try:
            msg = r.json().get("message") or r.text[:120]
        except ValueError:
            msg = r.text[:120]
        raise NessieError(f"Nessie said {r.status_code}: {msg}")
    try:
        return r.json()
    except ValueError:
        return {}


def _created_id(resp) -> str:
    obj = (resp or {}).get("objectCreated") or resp or {}
    if not obj.get("_id"):
        raise NessieError("Nessie did not return an id for what it created.")
    return obj["_id"]


def _read(path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _write(path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=1))


# ---------------------------------------------------------------------- the ledger (Nessie, or local)
class Ledger:
    """fund(site, amount) / spend(site, amount, memo) / status(site), on Nessie when a key is set."""

    @property
    def backend(self) -> str:
        return "nessie" if configured() else "local"

    @property
    def label(self) -> str:
        return "Capital One Nessie" if configured() else "local ledger (Nessie not connected)"

    # -- Nessie
    def _customer(self, ids: dict) -> str:
        if not ids.get("customer"):
            ids["customer"] = _created_id(_call("POST", "/customers", {
                "first_name": "Algaebro", "last_name": "Cleanup Fund",
                "address": {"street_number": "1", "street_name": "Charles River Dam Rd", "city": "Boston", "state": "MA", "zip": "02114"}}))
        return ids["customer"]

    def _account(self, site: str, create_with: Optional[int] = None) -> Optional[str]:
        ids = _read(IDS_JSON)
        acc = (ids.get("accounts") or {}).get(site)
        if acc or create_with is None:
            return acc
        cust = self._customer(ids)
        name = config.SITE_BY_KEY.get(site, {}).get("name", site)
        acc = _created_id(_call("POST", f"/customers/{cust}/accounts", {
            "type": "Savings", "nickname": f"{name} cleanup fund"[:40], "rewards": 0, "balance": int(create_with)}))
        ids.setdefault("accounts", {})[site] = acc
        _write(IDS_JSON, ids)
        return acc

    def fund(self, site: str, amount: float) -> dict:
        amount = int(round(amount))
        if amount <= 0:
            raise ValueError("The budget must be more than $0.")
        with _lock:
            if self.backend == "local":
                led = _read(LOCAL_JSON)
                s = led.setdefault(site, {"funded": 0, "spent": []})
                s["funded"] += amount
                _write(LOCAL_JSON, led)
                return self.status(site)
            acc = self._account(site)
            if acc is None:
                self._account(site, create_with=amount)
            else:
                _call("POST", f"/accounts/{acc}/deposits", {"medium": "balance", "transaction_date": date.today().isoformat(),
                                                            "amount": amount, "description": "Cleanup budget added"})
            ids = _read(IDS_JSON)
            ids.setdefault("funded", {})[site] = int(ids.get("funded", {}).get(site, 0)) + amount
            _write(IDS_JSON, ids)
        return self.status(site)

    def spend(self, site: str, amount: float, memo: str) -> dict:
        amount = int(round(amount))
        st = self.status(site)
        if not st["funded"]:
            raise ValueError("Set a budget for this water body first.")
        if amount > st["remaining"]:
            raise ValueError(f"That needs ${amount:,} but only ${st['remaining']:,} is left in the fund.")
        with _lock:
            if self.backend == "local":
                led = _read(LOCAL_JSON)
                led[site]["spent"].append({"amount": amount, "memo": memo[:120], "date": date.today().isoformat()})
                _write(LOCAL_JSON, led)
            else:
                _call("POST", f"/accounts/{self._account(site)}/withdrawals", {
                    "medium": "balance", "transaction_date": date.today().isoformat(), "amount": amount, "description": memo[:120]})
        return self.status(site)

    def status(self, site: str) -> dict:
        if self.backend == "local":
            s = _read(LOCAL_JSON).get(site) or {"funded": 0, "spent": []}
            spent = sum(x["amount"] for x in s["spent"])
            return {"site": site, "backend": "local", "label": self.label, "funded": s["funded"], "spent": spent,
                    "remaining": s["funded"] - spent, "transactions": s["spent"][::-1][:10], "account_id": None}
        acc = self._account(site)
        if acc is None:
            return {"site": site, "backend": "nessie", "label": self.label, "funded": 0, "spent": 0, "remaining": 0,
                    "transactions": [], "account_id": None}
        info = _call("GET", f"/accounts/{acc}")
        wd = _call("GET", f"/accounts/{acc}/withdrawals") or []
        # We add up what we funded and what was withdrawn ourselves, because Nessie can leave new transactions
        # 'pending' for a while before it changes the account balance. Its own balance is shown alongside.
        funded = int((_read(IDS_JSON).get("funded") or {}).get(site, 0))
        spent = sum(int(w.get("amount", 0)) for w in wd)
        tx = [{"amount": int(w.get("amount", 0)), "memo": w.get("description", ""), "date": w.get("transaction_date", ""),
               "status": w.get("status")} for w in wd][::-1][:10]
        return {"site": site, "backend": "nessie", "label": self.label, "funded": funded, "spent": spent,
                "remaining": funded - spent, "nessie_balance": info.get("balance"), "transactions": tx, "account_id": acc}


# ---------------------------------------------------------------------- businesses near the water
def merchants_near(lat: float, lon: float, miles: float = 1.0) -> dict:
    if not configured():
        return {"ok": False, "reason": "Add NESSIE_API_KEY to .env to look up businesses near the water.", "merchants": []}
    try:
        got = _call("GET", "/merchants", params={"lat": round(lat, 5), "lng": round(lon, 5), "rad": miles})
    except NessieError as e:
        return {"ok": False, "reason": str(e), "merchants": []}
    rows = []
    for m in got if isinstance(got, list) else []:
        g = m.get("geocode") or {}
        cat = m.get("category")
        rows.append({"name": str(m.get("name", ""))[:60], "category": ", ".join(cat) if isinstance(cat, list) else str(cat or ""),
                     "lat": g.get("lat"), "lon": g.get("lng")})
    return {"ok": True, "merchants": rows[:40], "count": len(rows), "radius_miles": miles,
            "note": "From Nessie's shared sandbox merchant list: an illustration of exposure, not a full count of local businesses."}
