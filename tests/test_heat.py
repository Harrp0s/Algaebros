import io

import numpy as np
import pytest
from PIL import Image

from algaebro import heat, zones


@pytest.fixture(scope="module")
def grid(synth):
    labels, x, y = zones.load_grid()
    return zones.load_zone_records(), labels, x, y


def alpha_of(png):
    return np.array(Image.open(io.BytesIO(png)).convert("RGBA"))[..., 3]


def no_scene(*_):
    return None


def test_overlay_is_a_georeferenced_transparent_png_with_a_legend(replay_fc, grid):
    recs, labels, x, y = grid
    png, meta = heat.heat_overlay(replay_fc, recs, labels, x, y, scene_loader=no_scene)
    img = Image.open(io.BytesIO(png))
    assert img.mode == "RGBA" and img.size == (meta["width"], meta["height"]) and (alpha_of(png) > 200).any()
    (nw, ne, se, sw) = meta["coordinates"]                    # MapLibre wants north-west, north-east, south-east, south-west
    assert nw[0] < ne[0] and nw[1] > sw[1] and nw == [sw[0], nw[1]] and se == [ne[0], sw[1]]
    assert meta["legend"]["stops"][0]["color"].startswith("#") and meta["legend"]["high"] == replay_fc["thresholds"]["high"]
    assert meta["source"] == "interpolated" and "not a pixel-by-pixel measurement" in meta["note"]


def test_colour_ramp_is_orange_exactly_where_the_flag_starts(replay_fc):
    th = replay_fc["thresholds"]
    rgb = heat.colorize(np.array([[th["high"], th["elevated"] - 0.5, th["high"] + 1.0]]), th)[0]
    assert tuple(rgb[0]) == heat.ORANGE and tuple(rgb[1]) == heat.MINT and tuple(rgb[2]) == heat.RED


def test_areas_without_a_reading_stay_transparent_and_nothing_is_invented(replay_fc, grid):
    recs, labels, x, y = grid
    none = dict(replay_fc, zones=[{k: v for k, v in z.items() if k != "last_obs"} for z in replay_fc["zones"]])
    png, meta = heat.heat_overlay(none, recs, labels, x, y, scene_loader=no_scene)
    assert alpha_of(png).max() == 0 and meta["source"] == "interpolated"


def test_real_pixels_are_used_when_a_saved_scene_exists_and_clouds_stay_clear(replay_fc, grid):
    recs, labels, x, y = grid
    scene = np.where(labels > 0, 0.30, np.nan).astype("float32")
    scene[:, : scene.shape[1] // 2] = np.nan                  # the west half is cloud-covered in this scene
    png, meta = heat.heat_overlay(replay_fc, recs, labels, x, y, scene_loader=lambda a, l: ("2026-07-17", scene, 0.5))
    assert meta["source"] == "pixels" and meta["scene_date"] == "2026-07-17" and meta["coverage"] == 0.5
    a = alpha_of(png)
    cols = np.where((a > 200).any(axis=0))[0]
    mid_lon = (np.asarray(x)[labels.shape[1] // 2])      # sanity: coloured area sits in the east half only
    assert cols.min() > a.shape[1] * 0.3, "the cloud-covered west half must not be painted"


def test_prefer_controls_the_source(replay_fc, grid):
    recs, labels, x, y = grid
    scene = np.where(labels > 0, 0.1, np.nan).astype("float32")
    load = lambda a, l: ("2026-07-17", scene, 0.9)            # noqa: E731
    assert heat.heat_overlay(replay_fc, recs, labels, x, y, "interpolated", load)[1]["source"] == "interpolated"
    assert heat.heat_overlay(replay_fc, recs, labels, x, y, "auto", load)[1]["source"] == "pixels"
    assert heat.heat_overlay(replay_fc, recs, labels, x, y, "pixels", no_scene)[1]["source"] == "interpolated"
    assert heat.heat_overlay(replay_fc, recs, labels, x, y, "auto", lambda a, l: 1 / 0)[1]["source"] == "interpolated", "a broken loader must not break the map"


def test_without_a_pixel_grid_it_still_draws_from_the_stored_points(replay_fc, grid):
    recs, *_ = grid
    png, meta = heat.heat_overlay(replay_fc, recs, None, None, None)
    assert (alpha_of(png) > 0).any() and meta["source"] == "interpolated"


def test_interpolation_blends_smoothly_between_two_areas():
    fc = {"zones": [{"number": 1, "last_obs": 0.0}, {"number": 2, "last_obs": 0.2}]}
    centers = [{"number": 1, "lat": 42.35, "lon": -71.12}, {"number": 2, "lat": 42.35, "lon": -71.10}]
    lon = np.array([-71.12, -71.11, -71.10])
    lat = np.full(3, 42.35)
    v = heat.interpolate_values(fc, lon, lat, np.array([1, 1, 2]), centers)
    assert v[0] < 0.03 and v[2] > 0.17 and 0.07 < v[1] < 0.13 and v[0] < v[1] < v[2]


def test_blur_keeps_total_brightness_and_dilate_grows_a_mask():
    a = np.zeros((41, 41), dtype=np.float32)
    a[20, 20] = 1.0
    assert heat.gaussian_blur(a, 3.0).sum() == pytest.approx(1.0, abs=1e-3)
    m = heat.dilate(a, 2)
    assert m.sum() == 25 and m[18:23, 18:23].all()


def test_medoid_is_a_real_water_point_of_the_zone(grid):
    recs, *_ = grid
    lat, lon = heat.medoid(recs[0])
    assert [lon, lat] in [list(p) for p in recs[0]["points"]]
