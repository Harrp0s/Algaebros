# Algaebro: algae-bloom risk alerts for the Charles River and nearby lakes, by iMessage

Flags zones of the Charles River whose latest satellite chlorophyll reading is in the top 10% of the river's
own history, and texts people (via Photon 2.0 iMessage) when a zone they follow is flagged HIGH. Every alert
says how often that signal was still true a week later in past years. A planner map replays any past season
next to what the satellite actually saw.

**What it is not:** a toxin test or a safety determination. It reads a *proxy* (chlorophyll in surface
water, from satellite red-edge reflectance), and every message says so and points to official sources.

## What we found (8 seasons of real Sentinel-2 + weather data, tested on years the method never saw)

* We trained a weather-driven model and compared it with "the next reading will equal the latest reading".
* The model had slightly lower average error, but it almost never reached the HIGH range, so as an alert system it was
  far worse at catching HIGH readings (recall 7% vs 46%). **Weather did not add skill at a 7-day horizon.**
* So alerts are based on the latest clear satellite reading, with its measured track record:
  when a zone read HIGH, 42% of readings were still HIGH a week later and 68% were at least elevated;
  when it read below 'elevated', 5% reached HIGH. The code picks the method automatically (`choose_method`).
* Caveats we state out loud: the 206 HIGH readings in the test years fall on only 38 dates (a basin lights up at once),
  so these rates are rough; differences of ~0.15 in F1 between methods are within noise.
* We removed a "% chance of HIGH" estimate because, checked against outcomes, it said 89% when the real rate was 46%.

---

## Status: read this first

| Piece | Status |
|---|---|
| Feature engineering, leakage guards, model evaluation, bot, alerts, map | Tested (68 tests, pandas 2.2 and 3.0, Python 3.12) |
| Photon client code | Tested against a fake that returns Photon's own response models. **Not tested against the live service** |
| Satellite download (Planetary Computer) | Written against current library docs. **Not run against the live service** |
| Weather download (Open-Meteo) | Same. **Not run against the live service** |
| Trained and evaluated on real data | Yes, by the team on 8 seasons of real data (see 'What we found'). The numbers in the tests are synthetic |
| Python 3.10 / 3.11 | Not run. A scanner (`tools/check_py310_compat.py`) confirms no 3.12-only syntax |

All numbers in `tests/` come from **synthetic** data and say nothing about the river. Synthetic data is
tagged in `manifest.json`, and every alert and the map carry a loud SYNTHETIC banner if it is ever used.

---

## Setup (Windows PowerShell; use Python 3.11 or 3.12)

```powershell
cd algaebro
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
copy .env.example .env                # then fill in the two Photon values (see below)
```

### Photon 2.0 setup (needs a Photon account; Beta platform)
1. https://app.photon.codes: create a project. A Beta project ID starts with `pho_prj_`. If yours does not, you are on their older "Spectrum" product and this code does not apply.
2. Create a project API key with `project:read` and `project:write`. Put the key and project ID in `.env`.
3. In the project's **iMessage** page, add the phone number you will text FROM, then assign a shared line. Copy the assigned number.
4. Text that number from your phone. **A shared line cannot send the first message, and only handles you add can reach it.** Everyone who will try the demo must be added (ask Photon staff about seat limits).

## Run it, in this order

```powershell
python scripts/00_check_apis.py --photon     # STEP 0: stop here if anything FAILS
python scripts/01_build_zones.py             # STEP 1: then OPEN data\zones_preview.png and check the zones sit on the river
python scripts/02_fetch_satellite.py --limit 5   # STEP 2a: quick test
python scripts/02_fetch_satellite.py             # STEP 2b: full history; safe to stop and re-run
python scripts/03_fetch_weather.py           # STEP 3
python scripts/04_train.py                   # STEP 4: prints out-of-sample results vs baselines
python scripts/05_forecast.py --as-of 2025-10-03      # try a REPLAY forecast
python scripts/05_forecast.py --replay-year 2025      # weekly replay for the map
python scripts/06_make_map.py                # open data\map.html
python scripts/chat_console.py --as-of 2025-10-03     # try the bot in the terminal
python -m algaebro.run_bot --as-of 2025-10-03         # the real iMessage bot, in replay mode
```

If step 1 puts zones in the wrong place, edit `BBOX` in `algaebro/config.py` (the numbers are from memory) and re-run it.
To name zones, create `data/zone_names.json` like `{"Z03": "Near the BU Bridge"}` and re-run step 1.

## Demo plan (it is October, so use replay; the real data's biggest event is early October 2025)

1. Text the Photon number from your phone: `help`, `zones`, then `subscribe 3` (this also opens the door for alerts).
2. Start `python -m algaebro.run_bot --as-of 2025-10-03` (or another date where your map shows HIGH). The alert arrives tagged `[REPLAY of ...]`.
3. Show `map.html`: scrub the slider, point at the satellite-vs-forecast columns, **including the misses**, and the "How good is it" box.
4. Say the limits out loud: proxy not toxin; zones are ~20 m pixels of a narrow river; weather is one point; 'high' means top 10% of this river's own history.

## How the forecast avoids cheating
* To forecast date D it uses only weather up to D minus 7 days and the latest satellite scene that had been *published* by then (2-day processing delay, `OBS_LATENCY_DAYS`).
* Evaluation is expanding-window by year: test year Y is scored by a model trained on years before Y.
* The model is judged on how well it CATCHES 'high' readings (F1, AUC), not on average error. Unless it clearly beats "next reading = last reading", alerts use that simple rule and say so.
* `tests/test_features_no_leakage.py` poisons everything after the issue date and checks the forecast does not change.


## The dashboard (v8): 3D map, heat gradient, Grok chat with voice, Forecast Lab

```powershell
pip install -r requirements.txt
pip install daytona                                    # optional: Forecast Lab sandboxes + "Ask the data"
copy .env.example .env                                 # then fill in the keys you have (see below)
python scripts/check_ai.py                             # checks Grok, voice, Nessie and Daytona; never prints a key
python scripts/suggest_zone_names.py --write           # place names from OpenStreetMap; then OPEN data\zone_names.json and fix any that look wrong
python -m algaebro.server --as-of 2025-10-03           # dashboard + the live iMessage bot, in REPLAY mode
python -m algaebro.server                              # same, live
python -m algaebro.server --no-photon                  # dashboard only (alerts show in the feed, nothing is texted)
python scripts/07_fetch_pixels.py --years 2025         # optional, 10-15 min: real satellite pixels for the heat gradient
```

Opens at http://localhost:8765 . Nothing is loaded from the internet except the map's own tiles (MapLibre and the font are bundled).
If the detailed 3D base map cannot load (no network, blocked), a plain base map is used and everything else keeps working.

**Keys** (in `.env`, never in code or chat): `XAI_API_KEY` (Grok chat and voice), `DAYTONA_API_KEY` (sandboxes), `NESSIE_API_KEY` (cleanup funds), plus the Photon keys.
Model names change quickly. If `XAI_MODEL` (default `grok-4.7`) is rejected, the app asks xAI which models exist, uses the newest plain text one and tells you.

### What is on the screen
* **Map** (MapLibre GL, tiltable 3D with extruded buildings). Flagged areas get a large pulsing red marker with their name and a translucent 3D column. Click any part of the river to select it.
* **Heat gradient.** With saved satellite pixels (`07_fetch_pixels.py`) it paints the real 20 m pixels from the freshest usable clear scene; cloud stays transparent.
  Without them it shows a smooth blend between the measured areas and says so in the legend. A "Heat" switch turns it off.
* **Timeline** (bottom centre): Live or Replay, play button, week steps, a date slider and "Biggest event". Scrubbing never texts anyone; releasing the slider or pressing a button does.
* **Details panel** (right): hide with the X or the key `P`, collapse with the chevron. **Alerts sent** (bottom left): collapse, "Clear" hides what is shown now, "Show more".
* **Demo mode** (toggle, top right): a draggable pin simulates one chosen phone. Drag it into a flagged area and that phone gets an iMessage tagged `[DEMO: simulated location]`.
  The phone must have texted the bot once (shared Photon lines cannot start a conversation). Switch it off and the pin disappears and the pin API refuses.
* **Ask Grok** (button bottom right, key `C`): answers only from what the dashboard is showing. It cannot say water is safe: "is it safe / can my dog swim" questions get a fixed careful answer
  without calling Grok, and every Grok answer is checked for safety verdicts before you see it. Mic button: your voice is recorded in the browser, converted to a small WAV, transcribed by xAI,
  and answers are read aloud by xAI's voice (the browser's own voice is the fallback; with no key it still answers from fixed wording).
* **Forecast Lab** (button, key `L`): tests 7 forecasting ideas, each in its own Daytona sandbox, all in parallel (quick: 8 sandboxes; full: 22), on seasons they never saw, with 95% ranges.
  The latest season is held out; the best idea per time window is then re-tested on it alone (2 or 6 more sandboxes), so a lucky win does not count. One sandbox measures which inputs matter.
  It tells you which runner actually ran each result. **Ask the data**: Grok writes a short analysis, Daytona runs it with the network blocked, you see the code and the result.
  AI-written code is **never** run on this computer: without Daytona the question is refused.
* **1-7 day forecasts** (Forecast Lab > *Train 1-7 day forecasts*, or `python scripts/09_train_horizons.py`): for every day ahead, Daytona tests
  5 model ideas x 3 settings plus the simple rule (112 sandboxes), confirms each day's best on the held-out season (14 more), then the winners are refitted
  here and saved (`data/horizon_models.joblib`). Real run, 2026-10-18: 126 experiments in 149 s on Daytona (about 25 minutes on a laptop).
  **One timeline slider**: left of *Now* replays past dates, *Now* is the latest reading, and right of *Now* is +1 to +7 days ahead. There,
  **Latest reading / ML model** switches what colours the map, and the area panel shows both side by side. Demo link: `http://127.0.0.1:8765/#ahead=3&ml`.
  Alerts switch to the ML for a day only if it was clearly better on earlier seasons AND on the unseen one. On the real data it never was
  (best: day 1, 0.50 vs 0.48, within noise), so alerts and the bot keep the latest reading at the standard 7 days. Science mode charts score by days ahead.
* **Charles at Watertown** (Z11-12) is marked low-confidence (30-60 m wide, edge pixels mix in the banks): shown on the map, never alerted.
  Flags read "High chlorophyll reading (possible algae)", and a late-season banner says a high reading is not a confirmed bloom.

* **Science mode** (button, key `S`): the scorecard (F1, AUC, recall: model vs simple rule), average error by test year, the alerts' track record,
  predicted-vs-actual scatter plots, input importance (from the Lab), the ideas tested and whether they held up, every water body's data, and the exact inputs behind the selected area's forecast.
* **Cleanup planner** (button, key `B`): for each water body, seven options (alum, aeration, copper or peroxide algaecide, ultrasonic buoys, dredging, stormwater phosphorus cuts),
  each with a 10-year cost range from a cited published source, how well it fits a river or a lake, cost per nearby resident (US Census) and whether it fits your budget.
  A budget can be put into a **cleanup fund**: a Capital One **Nessie** account per water body (a local ledger without `NESSIE_API_KEY`); "Fund" records a plan as a withdrawal.
  Nessie's merchant search lists businesses near the water. "Ask about the options" sends Grok only the computed table and tells it not to invent numbers.

### Water bodies (v9)
`config.SITES` lists them: the Lower Charles (Z01-Z10, unchanged), the Charles at Watertown (Z11-Z12), the Mystic Lakes (Z13-Z15), Spy Pond (Z16) and Fresh Pond (Z17).
Each site has its own satellite grid and cache under `data/sites/<site>/`; the combined grid feeds the map and the model, and the "water-body average" input is per site.
```powershell
python scripts/01_build_zones.py --list                # which sites are built
python scripts/01_build_zones.py --site mystic_lakes   # (re)build one; then OPEN data\sites\mystic_lakes\zones_preview.png
python scripts/02_fetch_satellite.py --site mystic_lakes
python scripts/04_train.py                             # retrain on every site
python scripts/08_fetch_population.py                  # Census population near each zone, for the planner (no key)
```
The Upper Charles between Waltham and Watertown is mostly narrower than two 20 m pixels, so only the wide Watertown reach is used.
All sites share one weather point (they lie within ~12 km, about one weather-model grid cell).

### Phone side
People text a location (or a map link) to get alerts near them, a photo to add a community report, and `STATUS`, `WHY`, `STOP`, `START`, `DELETE PHOTOS`.
Photos are stored in `data/photos`; EXIF metadata including GPS is removed; iPhone HEIC is converted to JPEG. The server listens on this computer only.

**Photo check (Grok vision).** Each stored photo is sent once to xAI, already stripped of metadata, and Grok answers: is this a real photo
taken at the water (not a meme, screenshot, selfie, satellite image or map), and does the water look like a bloom?
* Confident "not the water" or offensive -> **rejected** (kept but hidden, reversible). Confident "yes" -> **verified**. Anything else, or no xAI key -> **needs review**.
* Duplicates, tiny images and more than 10 photos a day per person are refused before anything is sent to xAI.
* The sender gets a short reply saying which of these happened. Photos never change a forecast or send an alert.
* **Photos panel** (button, key `V`): tabs for Needs review / Not screened / Verified / Rejected, buttons to Verify, Reject, send back to review, or Delete for good,
  and "Screen unscreened photos with Grok" for photos stored before the check existed.
* **Story bubbles on the map:** verified photos from the last 7 days show as round thumbnails at their area; a red pulsing ring means Grok thinks it looks
  like a bloom. Tap to view (story style, step through several). The "Community photos" switch in the legend hides them.
* "Verified" means a plausible photo of the water, not proof of where or when it was taken, and "looks like a bloom" is not a toxin test.
  `python scripts/check_ai.py` checks that your model accepts images; set `XAI_VISION_MODEL` in `.env` if it does not.

## Known limits (be upfront about these)
* NDCI is a chlorophyll proxy for all algae, not cyanobacteria or toxins. Narrow river pixels mix with banks and bridges; clouds leave gaps.
* Weather history is reanalysis; the last days come from a forecast model, a small mismatch.
* 'Near you' alerts need a shared location or a zone number. Without an app there is no background location.
* Photon docs call polling a connection-test pattern. Fine for a hackathon; use their live-event feed for production.
* Photos are checked by Grok (xAI) and labelled "looks like a bloom" at most; no photo changes a forecast or sends an alert.
* Reports (`report 4 ...`) are logged as unverified and do not change the forecast.
* `data/algaebro.db` stores phone numbers. Keep it private.
* Check `OFFICIAL_INFO` in `config.py` (default `crwa.org`) is the right place to point people before you demo.

* The heat gradient is an interpolation between measured areas until the real pixels are downloaded. It is labelled as such.
* The Forecast Lab says what happened on past seasons. On synthetic data it will "find" a weather effect because the synthetic data was built with one: the dashboard shows the SYNTHETIC banner on those results. On the real data the honest answer may well be that nothing beat the simple rule.
* The first Daytona run builds a sandbox image (pandas, scikit-learn) and can take several minutes. Run `python scripts/check_ai.py` once before you present.
* Voice input needs a microphone permission in the browser. Without an xAI key it uses the browser's own speech recognition (Chrome or Edge).
* xAI model names and voice endpoints change. `check_ai.py` is the quickest way to find out if something moved.

* Cleanup costs are planning-level ranges from published projects (Wisconsin DNR, NH DES, ITRC, EPA Region 1, ...), not quotes, and the Census figure counts whole tracts within 1 km.
  Nessie's merchants are shared sandbox data, not a census of local businesses; the planner says so.
* Testing many ideas makes a lucky winner likely; that is why the latest season is held out and only the per-window winners are re-tested on it.

## Tests
```powershell
pip install pytest
python -m pytest tests -q
```
