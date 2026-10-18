import sqlite3

from algaebro import naming
from algaebro.store import Store, UserCtx, mask_handle


def test_masking_never_shows_more_than_the_last_four_digits():
    assert mask_handle("+15164710974").endswith("0974") and "5164" not in mask_handle("+15164710974")
    assert mask_handle("") == "unknown" and mask_handle("abc") == "unknown"


def test_an_old_database_is_upgraded_in_place(tmp_path):
    old = tmp_path / "old.db"
    c = sqlite3.connect(old)
    c.execute("CREATE TABLE users(user_id TEXT PRIMARY KEY, handle TEXT, space_id TEXT, agent_id TEXT, lat REAL, lon REAL, created_at TEXT, last_seen TEXT)")
    c.execute("INSERT INTO users(user_id, handle) VALUES('x', '+1555')")
    c.commit()
    c.close()
    st = Store(old)
    st.set_location("x", 1.0, 2.0, "pin")
    u = st.get_user("x")
    assert u["loc_source"] == "pin" and u["alerts_off"] in (0, None) and st.recent_feed() == []


def test_location_source_and_alerts_flag(store):
    store.upsert_user(UserCtx("u", "+1617", "s", "a"))
    store.set_location("u", 42.0, -71.0, "pin")
    store.set_alerts_off("u", True)
    u = store.get_user("u")
    assert (u["loc_source"], u["alerts_off"]) == ("pin", 1)


def test_feed_is_newest_first_and_never_stores_the_full_number(store):
    for i in range(3):
        store.log_feed("proximity", "+15164710974", "Z01", "Zone 1", 100 + i, f"text {i}", True)
    rows = store.recent_feed(2)
    assert [r["text"] for r in rows] == ["text 2", "text 1"]
    assert all("5164" not in r["user_label"] for r in rows)


def test_photo_rows(store):
    pid = store.add_photo("u", None, None, None, "/tmp/a.jpg", "att1")
    assert store.photo_attachment_known("att1") and len(store.unplaced_photos("u")) == 1
    store.place_photo(pid, "Z02", 1.0, 2.0)
    assert store.photos_for_zone("Z02") == [] and store.unplaced_photos("u") == [], "pending photos are not public"
    store.set_photo_status(pid, "verified")
    assert store.photos_for_zone("Z02")[0]["id"] == pid
    assert store.delete_user_photos("u") == ["/tmp/a.jpg"] and store.get_photo(pid) is None


# ---- OpenStreetMap naming ----------------------------------------------------------
PAYLOAD = {"elements": [
    {"type": "way", "center": {"lat": 42.3501, "lon": -71.0900}, "tags": {"name": "Massachusetts Avenue", "bridge": "yes", "bridge:name": "Harvard Bridge"}},
    {"type": "relation", "center": {"lat": 42.3560, "lon": -71.0730}, "tags": {"name": "Charles River Esplanade", "leisure": "park"}},
    {"type": "node", "lat": 42.3600, "lon": -71.0800, "tags": {"name": "Community Boating", "leisure": "marina"}},
    {"type": "way", "center": {"lat": 42.3, "lon": -71.0}, "tags": {"highway": "service"}},
    {"type": "way", "center": {"lat": 42.3501, "lon": -71.0900}, "tags": {"bridge": "yes", "bridge:name": "Harvard Bridge"}},
]}


def zone(zid, n, lat, lon):
    return {"id": zid, "number": n, "lat": lat, "lon": lon, "points": [[lon, lat]]}


def test_parse_prefers_the_bridge_name_and_drops_unnamed_or_duplicate_features():
    feats = naming.parse_features(PAYLOAD)
    assert [(f["name"], f["kind"]) for f in feats] == [("Harvard Bridge", "bridge"), ("Charles River Esplanade", "park"), ("Community Boating", "boating")]


def test_nearest_landmark_wins_and_far_zones_get_no_suggestion():
    feats = naming.parse_features(PAYLOAD)
    out = naming.suggest([zone("Z01", 1, 42.3503, -71.0902), zone("Z03", 3, 42.0, -71.0)], feats)
    assert out["Z01"]["name"] == "Harvard Bridge" and out["Z01"]["kind"] == "bridge" and out["Z01"]["distance_m"] < 100
    assert out["Z03"]["name"] is None


def test_two_zones_at_one_landmark_are_told_apart():
    feats = naming.parse_features(PAYLOAD)
    out = naming.suggest([zone("Z01", 1, 42.3503, -71.0900), zone("Z02", 2, 42.3499, -71.0900)], feats)
    assert out["Z01"]["name"] != out["Z02"]["name"] and out["Z01"]["name"].startswith("Harvard Bridge (")


def test_the_query_covers_the_study_area_and_asks_for_bridges():
    q = naming.build_query((-71.14, 42.34, -71.068, 42.38))
    assert "bridge" in q and "42.336" in q and "out center tags" in q


# ---- regression cases taken from the real OpenStreetMap output on the Charles ---------------------
def _north(m):
    return 42.35 + m / 111320.0


def _way(name, m, bridge=True, **tags):
    return {"type": "way", "center": {"lat": _north(m), "lon": -71.1}, "tags": {"name": name, **({"bridge": "yes"} if bridge else {}), **tags}}


def _zone_at(zid, n, m):
    return {"id": zid, "number": n, "lat": _north(m), "lon": -71.1, "points": [[-71.1, _north(m)]]}


def test_a_boathouse_beside_the_river_beats_a_highway_that_is_tagged_as_a_bridge():
    payload = {"elements": [_way("Storrow Drive", 108), {"type": "node", "lat": _north(-7), "lon": -71.1,
                                                         "tags": {"name": "W8 Resch Boathouse", "leisure": "marina"}}]}
    out = naming.suggest([_zone_at("Z05", 5, 0)], naming.parse_features(payload))
    assert out["Z05"]["name"] == "W8 Resch Boathouse" and "Storrow Drive (108 m)" in out["Z05"]["alternatives"]


def test_a_named_bridge_beats_a_nearby_street_and_overpasses_and_running_tracks_are_demoted():
    payload = {"elements": [_way("North Harvard Street", 1005), _way("Anderson Memorial Bridge", 1008),
                            _way("Grand Junction Running Track", 2063), _way("Boston University Bridge", 2106),
                            _way("Gloucester Street Overpass", 3081), _way("Harvard Bridge", 3192)]}
    zones_ = [_zone_at("Z01", 1, 1000), _zone_at("Z04", 4, 2000), _zone_at("Z07", 7, 3000)]
    out = naming.suggest(zones_, naming.parse_features(payload))
    assert [out[z]["name"] for z in ("Z01", "Z04", "Z07")] == ["Anderson Memorial Bridge", "Boston University Bridge", "Harvard Bridge"]


def test_a_street_on_a_bridge_is_still_used_when_nothing_better_is_close():
    out = naming.suggest([_zone_at("Z02", 2, 0)], naming.parse_features({"elements": [_way("River Street", 6)]}))
    assert out["Z02"]["name"] == "River Street" and out["Z02"]["kind"] == "road"


def test_alternatives_list_each_name_once():
    payload = {"elements": [_way("Harvard Bridge", 10 + i) for i in range(4)] + [_way("Longfellow Bridge", 90)]}
    out = naming.suggest([_zone_at("Z06", 6, 0)], naming.parse_features(payload))
    assert out["Z06"]["name"] == "Harvard Bridge" and out["Z06"]["alternatives"] == ["Longfellow Bridge (90 m)"]
