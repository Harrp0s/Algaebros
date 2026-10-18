from algaebro import geo


def test_plain_coordinates():
    assert geo.parse_location("I'm at 42.3550, -71.1100 now") == (42.355, -71.11)


def test_apple_maps_link():
    assert geo.parse_location("https://maps.apple.com/?ll=42.358,-71.095&q=Dropped%20Pin") == (42.358, -71.095)


def test_google_maps_link():
    assert geo.parse_location("https://www.google.com/maps/@42.3601,-71.0589,15z") == (42.3601, -71.0589)


def test_no_location_in_ordinary_text():
    assert geo.parse_location("subscribe 3") is None
    assert geo.parse_location("report 4 green scum, 2 ducks") is None
    assert geo.parse_location("") is None


def test_haversine_known_distance():
    # one degree of latitude is about 111.2 km
    assert abs(geo.haversine_m(42.0, -71.0, 43.0, -71.0) - 111195) < 300
