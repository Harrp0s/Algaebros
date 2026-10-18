"""Cleanup planner: what could be done about blooms on each water body, roughly what it costs, and for whom.

Every cost below comes from a published source (listed with it) and is shown as a RANGE, because real projects vary
several-fold with dose, depth, sediment and local prices. These are planning-level numbers to compare options with,
not quotes. The arithmetic is done here, in code; Grok only explains it (it is given the computed table and told not
to invent numbers).

Costs are for a 10-year horizon: one-time work counted as many times as it would need repeating, plus yearly running
costs. Water area comes from the satellite zones (open water away from the banks), so it slightly UNDERestimates the
full surface area.
"""
from __future__ import annotations

import json
import math
import re
from typing import Dict, List, Optional

from . import config

ACRES_PER_KM2 = 247.105
YD3_PER_ACRE_FT = 1613.33
YEARS = 10

ITRC = "https://hcb-1.itrcweb.org/management-and-control-strategies-for-hcbs/"
OPTIONS = [
    {
        "id": "alum", "name": "Alum treatment (lock up phosphorus in the sediment)",
        "what": "Aluminium sulfate is spread on the water; it binds phosphorus and settles, starving algae of the nutrient released from the lake bed.",
        "cost": {"kind": "per_acre_once", "low": 280, "high": 2970, "lasts_years": [4, 20]},
        "cost_text": "$280 to $700 per acre (Wisconsin DNR, avg about $450); a recent 153-acre NH project was about $2,970 per acre. Lasts about 4 to 20 years depending on dose.",
        "evidence": "established", "fits": {"lake": "good", "river": "poor"},
        "fit_why": {"lake": "Works best in still water where phosphorus comes back out of the sediment.",
                    "river": "Flowing water carries the floc away, and most river phosphorus arrives with stormwater, not from the bed."},
        "sources": [["Wisconsin DNR alum brochure", "https://apps.dnr.wi.gov/lakes/publications/documents/alum_brochure.pdf"],
                    ["NH DES, Lake Kanasatka treatment plan", "https://www.des.nh.gov/sites/g/files/ehbemt341/files/inline-documents/sonh/lake-kanasatka-treatment-plan.pdf"],
                    ["Seattle, Green Lake alum results", "https://www.seattle.gov/Documents/Departments/ParksAndRecreation/Projects/GreenLake/GreenLakeAlumPresentation.pdf"]],
    },
    {
        "id": "aeration", "name": "Aeration or mixing",
        "what": "Pumps or diffusers mix or oxygenate the water so the bottom stays oxygenated and releases less phosphorus.",
        "cost": {"kind": "capital_plus_yearly", "capital_per_acre": [730, 730], "yearly_per_acre": [1000, 1200]},
        "cost_text": "Running costs about $2,500 to $3,000 per hectare per year (about $1,000 to $1,200 per acre); Lake Carmi's system cost over $1.0M for 1,375 acres to install (about $730 per acre).",
        "evidence": "mixed", "fits": {"lake": "risky", "river": "no"},
        "fit_why": {"lake": "Can help deep lakes, but at Lake Carmi (Vermont) the state found it made blooms worse and removed it.",
                    "river": "A flowing river is already mixed."},
        "sources": [["ITRC: hypolimnetic oxygenation and aeration", "https://hcb-1.itrcweb.org/hypolimnetic-oxygenation-and-aeration/"],
                    ["VTDigger: Lake Carmi aeration to be removed (2024)", "https://vtdigger.org/2024/05/15/state-to-remove-lake-carmi-aeration-system-after-determining-it-made-cyanobacteria-blooms-worse/"]],
    },
    {
        "id": "copper", "name": "Copper algaecide (treat a bloom)",
        "what": "Copper sulfate or chelated copper kills algae already in the water. It treats the symptom, not the cause.",
        "cost": {"kind": "per_acre_per_treatment", "low": 933, "high": 933, "treatments_per_year": [1, 4]},
        "cost_text": "One New York study: about $933 per acre per treatment (ITRC Appendix C.2); one to four treatments a season.",
        "evidence": "symptom", "fits": {"lake": "short-term", "river": "poor"},
        "fit_why": {"lake": "Clears a bloom quickly, but dying cells can release toxins and use up oxygen; copper builds up in sediment.",
                    "river": "The treated water moves on downstream within days."},
        "drinking_water_warning": "Copper use on a drinking-water reservoir is tightly regulated and needs the water utility's approval.",
        "sources": [["ITRC: copper algaecides", "https://hcb-1.itrcweb.org/copper-algaecides/"]],
    },
    {
        "id": "peroxide", "name": "Peroxide algaecide (treat a bloom)",
        "what": "Hydrogen-peroxide products break down cyanobacteria cells and degrade to water and oxygen. Short-term relief.",
        "cost": {"kind": "unpriced"},
        "cost_text": "No published per-acre price found. ITRC rates it as a modest cost per season ($$ on its scale); the price is on each product label, so get an applicator's quote.",
        "evidence": "symptom", "fits": {"lake": "short-term", "river": "poor"},
        "fit_why": {"lake": "Targets cyanobacteria more than other algae and leaves no residue, but blooms come back if nutrients stay high.",
                    "river": "The treated water moves on downstream within days."},
        "sources": [["ITRC: peroxide application", "https://hcb-1.itrcweb.org/peroxide-application/"]],
    },
    {
        "id": "ultrasonic", "name": "Ultrasonic buoys",
        "what": "Floating units emit sound that is meant to stop cyanobacteria from floating up to the light.",
        "cost": {"kind": "per_unit", "unit_cost": [47000, 54000], "acres_per_unit": 50},
        "cost_text": "A California city paid about $47,000 to $54,000 per buoy in 2024; one unit is said to cover about 50 acres. Excludes installation and monitoring fees.",
        "evidence": "emerging", "fits": {"lake": "uncertain", "river": "poor"},
        "fit_why": {"lake": "ITRC lists field evidence as limited and emerging, and it does nothing about nutrient inputs.",
                    "river": "Designed for still water."},
        "sources": [["ITRC: ultrasound", "https://hcb-1.itrcweb.org/ultrasound/"],
                    ["Clear Lake LG Sonic buoy project (CEQA)", "https://ceqanet.lci.ca.gov/2026050086"]],
    },
    {
        "id": "dredging", "name": "Dredging (remove nutrient-rich sediment)",
        "what": "Removes the top layer of lake-bed sediment that stores and releases phosphorus.",
        "cost": {"kind": "per_yd3_once", "low": 6, "high": 30, "depth_ft": 1.0},
        "cost_text": "Hydraulic dredging about $8 to $24 per cubic yard (Illinois EPA via ITRC), $6 to $30 for lakes overall; priced here for one foot of sediment.",
        "evidence": "established", "fits": {"lake": "costly", "river": "costly"},
        "fit_why": {"lake": "Long-lasting if the inflow of new phosphorus is also cut, but disruptive and needs a disposal site.",
                    "river": "The Charles carries new sediment in, and old river sediment can hold contaminants that need special handling."},
        "sources": [["ITRC: dredging", "https://hcb-1.itrcweb.org/dredging/"]],
    },
    {
        "id": "watershed", "name": "Cut phosphorus from stormwater (root cause)",
        "what": "Rain gardens, infiltration, street sweeping and similar measures stop phosphorus washing off streets and roofs into the water.",
        "cost": {"kind": "per_lb", "low": 31700, "high": 138000, "lbs_default": 25},
        "cost_text": "EPA Region 1 (2022): median about $73,000 and mean about $138,000 per pound of yearly phosphorus removed; an earlier Charles estimate was about $31,700 per pound. Excludes upkeep.",
        "evidence": "root-cause", "fits": {"lake": "good", "river": "good"},
        "fit_why": {"lake": "Stops the nutrient supply, so in-lake treatments last longer.",
                    "river": "This is the Lower Charles plan: EPA's 2007 TMDL calls for cutting phosphorus by 54%."},
        "sources": [["EPA Region 1 cost-benefit resource kit (2022)", "https://www3.epa.gov/region1/npdes/rda/ar/appendixr-7-cost-benefit-resource-kit-revised-07132022.pdf"],
                    ["EPA Charles River steering committee (2011)", "https://19january2017snapshot.epa.gov/www3/region1/npdes/charlesriver/pdfs/SteeringCommitteeMtg4Presentation2011Jun29.pdf"]],
    },
]
OPTION_BY_ID = {o["id"]: o for o in OPTIONS}
FIT_SCORE = {"good": 3, "root-cause": 3, "short-term": 2, "costly": 1, "uncertain": 1, "risky": 1, "poor": 0, "no": -1}


def cost_10y(option: dict, acres: float, lbs: Optional[float] = None) -> Optional[List[float]]:
    """[low, high] dollars over YEARS, or None when there is no published price."""
    c = option["cost"]
    k = c["kind"]
    if k == "per_acre_once":
        lo_last, hi_last = c["lasts_years"]
        return [acres * c["low"] * math.ceil(YEARS / hi_last), acres * c["high"] * math.ceil(YEARS / lo_last)]
    if k == "capital_plus_yearly":
        return [acres * (c["capital_per_acre"][0] + YEARS * c["yearly_per_acre"][0]),
                acres * (c["capital_per_acre"][1] + YEARS * c["yearly_per_acre"][1])]
    if k == "per_acre_per_treatment":
        return [acres * c["low"] * c["treatments_per_year"][0] * YEARS, acres * c["high"] * c["treatments_per_year"][1] * YEARS]
    if k == "per_unit":
        units = max(1, math.ceil(acres / c["acres_per_unit"]))
        return [units * c["unit_cost"][0], units * c["unit_cost"][1]]
    if k == "per_yd3_once":
        yd3 = acres * c["depth_ft"] * YD3_PER_ACRE_FT
        return [yd3 * c["low"], yd3 * c["high"]]
    if k == "per_lb":
        n = c["lbs_default"] if lbs is None else lbs
        return [n * c["low"], n * c["high"]]
    return None


def site_facts(site: str, zones: List[dict], pop: dict, obs=None, thr_high: Optional[float] = None) -> dict:
    zs = [z for z in zones if z.get("site", config.DEFAULT_SITE) == site]
    km2 = sum(z.get("area_km2", 0) for z in zs)
    s = config.SITE_BY_KEY.get(site, {"name": site, "kind": "lake"})
    facts = {"site": site, "name": s["name"], "kind": s.get("kind", "lake"), "drinking_water": bool(s.get("drinking_water")),
             "zones": [z["id"] for z in zs], "area_km2": round(km2, 3), "acres": round(km2 * ACRES_PER_KM2, 1),
             "people": (pop.get("sites") or {}).get(site, {}).get("people"),
             "density_per_km2": (pop.get("sites") or {}).get(site, {}).get("density_per_km2"),
             "flagged_share": None, "n_readings": 0}
    if obs is not None and thr_high is not None:
        o = obs[obs["site"] == site] if "site" in obs.columns else obs.iloc[0:0]
        facts["n_readings"] = int(len(o))
        if len(o):
            facts["flagged_share"] = round(float((o["ndci"] >= thr_high).mean()), 3)
    return facts


def plan(facts: dict, budget: Optional[float] = None, lbs: Optional[float] = None) -> dict:
    """Every option costed for this water body, judged on fit, and ranked: what is feasible within the budget."""
    kind = facts["kind"]
    rows = []
    for o in OPTIONS:
        c10 = cost_10y(o, facts["acres"], lbs)
        fit = o["fits"].get(kind, "uncertain")
        row = {"id": o["id"], "name": o["name"], "what": o["what"], "fit": fit, "fit_why": o["fit_why"].get(kind, ""),
               "evidence": o["evidence"], "cost_10y": [round(v) for v in c10] if c10 else None, "cost_text": o["cost_text"],
               "sources": o["sources"], "warning": o.get("drinking_water_warning") if facts.get("drinking_water") else None}
        if c10 and facts.get("people"):
            row["per_person_10y"] = [round(c10[0] / facts["people"], 2), round(c10[1] / facts["people"], 2)]
        if budget is not None and c10:
            row["within_budget"] = "yes" if c10[1] <= budget else "maybe" if c10[0] <= budget else "no"
        rows.append(row)

    def rank(r):
        afford = {"yes": 0, "maybe": 1, "no": 2}.get(r.get("within_budget"), 1)
        return (-FIT_SCORE.get(r["fit"], 0), afford, (r["cost_10y"] or [math.inf])[0])

    usable = [r for r in rows if FIT_SCORE.get(r["fit"], 0) >= 2 and r["cost_10y"]]
    feasible = [r for r in usable if budget is None or r.get("within_budget") in ("yes", "maybe")]
    best = sorted(feasible, key=rank)[0] if feasible else None
    if best:
        rec = (f"{best['name']} fits {facts['name']} ({best['fit']}) and costs about ${best['cost_10y'][0]:,.0f} to "
               f"${best['cost_10y'][1]:,.0f} over {YEARS} years.")
        if budget is not None and best.get("within_budget") == "maybe":
            rec += " The budget covers the low end of that range only."
    elif budget is not None and usable:
        cheapest = min(usable, key=lambda r: r["cost_10y"][0])
        rec = (f"Nothing that fits {facts['name']} is affordable on ${budget:,.0f}. The cheapest suitable option is "
               f"{cheapest['name']} at about ${cheapest['cost_10y'][0]:,.0f} or more.")
    else:
        rec = "No priced option fits this water body well."
    return {"facts": facts, "years": YEARS, "budget": budget, "lbs": lbs, "options": sorted(rows, key=rank),
            "recommendation": rec, "recommended": best["id"] if best else None,
            "caution": "Planning-level ranges from published projects, not quotes. Real costs depend on depth, dose, "
                       "sediment, permits and local prices; a lake-management engineer would refine them."}


# ---------------------------------------------------------------------- asking Grok about a plan
SYSTEM = ("You help compare ways to reduce algal blooms on a Boston-area water body. Use ONLY the facts and numbers in the "
          "JSON you are given; never invent a cost, a population figure or a study. If the JSON does not answer the question, "
          "say so. Keep it under 150 words, plain language, and say that the figures are planning ranges, not quotes. "
          "Do not say whether the water is safe.")


def plan_messages(question: str, p: dict) -> List[Dict[str, str]]:
    slim = {"water_body": p["facts"], "budget": p["budget"], "years": p["years"], "recommendation": p["recommendation"],
            "options": [{k: r.get(k) for k in ("name", "fit", "fit_why", "evidence", "cost_10y", "per_person_10y", "within_budget", "warning")}
                        for r in p["options"]]}
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "FACTS (JSON):\n" + json.dumps(slim) + "\n\nQUESTION: " + question.strip()[:500]}]


def offline_answer(question: str, p: dict) -> str:
    """No Grok: compare the options the question names (or give the recommendation)."""
    q = question.lower()
    named = [r for r in p["options"] if any(w in q for w in _keywords(r["id"]))]
    if len(named) >= 2:
        a, b = named[0], named[1]
        if a["cost_10y"] and b["cost_10y"]:
            cheaper = a if a["cost_10y"][0] <= b["cost_10y"][0] else b
            other = b if cheaper is a else a
            return (f"Over {p['years']} years, {cheaper['name']} (about ${cheaper['cost_10y'][0]:,.0f} to ${cheaper['cost_10y'][1]:,.0f}) "
                    f"is likely cheaper than {other['name']} (about ${other['cost_10y'][0]:,.0f} to ${other['cost_10y'][1]:,.0f}). "
                    f"Fit for this water body: {cheaper['fit']} vs {other['fit']}. These are planning ranges, not quotes.")
        unpriced = a if not a["cost_10y"] else b
        return f"{unpriced['name']} has no published per-acre price, so I cannot compare costs; get an applicator's quote."
    if len(named) == 1 and named[0]["cost_10y"]:
        r = named[0]
        return (f"{r['name']}: about ${r['cost_10y'][0]:,.0f} to ${r['cost_10y'][1]:,.0f} over {p['years']} years here. "
                f"Fit: {r['fit']}. {r['fit_why']}")
    return p["recommendation"]


def _keywords(option_id: str) -> List[str]:
    return {"alum": ["alum"], "aeration": ["aerat", "mixing", "bubbler"], "copper": ["copper"], "peroxide": ["peroxide"],
            "ultrasonic": ["ultrason", "sonic", "sound"], "dredging": ["dredg"],
            "watershed": ["stormwater", "watershed", "runoff", "rain garden"]}.get(option_id, [option_id])


def answer(question: str, p: dict, chat=None) -> dict:
    question = re.sub(r"\s+", " ", question or "").strip()
    if not question:
        raise ValueError("Type a question first.")
    if chat is None:
        return {"reply": offline_answer(question, p), "source": "fixed"}
    return {"reply": chat(plan_messages(question, p), temperature=0.2), "source": "grok"}
