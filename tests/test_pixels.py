import numpy as np
import pytest

from algaebro import config, pixels


@pytest.fixture()
def labels():
    lab = np.zeros((6, 8), dtype=np.int16)
    lab[1:5, 1:7] = 1
    return lab


def scene(labels, value=0.2, clouds=0.0, seed=0):
    arr = np.where(labels > 0, value, np.nan).astype("float16")
    if clouds:
        rng = np.random.default_rng(seed)
        arr[(labels > 0) & (rng.random(labels.shape) < clouds)] = np.nan
    return arr


def test_pixel_map_keeps_only_clear_open_water_inside_the_zones(labels):
    b04 = np.full(labels.shape, 0.05, dtype="float32")
    b05 = np.full(labels.shape, 0.08, dtype="float32")
    scl = np.full(labels.shape, config.SCL_WATER, dtype="uint8")
    scl[2, 2] = 9                                         # cloud
    b04[3, 3] = np.nan                                    # missing data
    out = pixels.pixel_map({"b04": b04, "b05": b05, "scl": scl}, labels)
    assert np.isnan(out[2, 2]) and np.isnan(out[3, 3]) and np.isnan(out[0, 0]), "cloud, missing data and land stay empty"
    assert np.isfinite(out[1, 1]) and out.dtype == np.float16
    assert pixels.coverage(out, labels) == pytest.approx((24 - 2) / 24)


def test_saved_scenes_round_trip_and_are_listed_in_date_order(labels, tmp_path):
    for d, i in [("2025-08-02", "B"), ("2025-07-01", "A")]:
        pixels.save_scene(d, i, scene(labels), tmp_path)
    assert [d for d, _ in pixels.list_scenes(tmp_path)] == ["2025-07-01", "2025-08-02"]
    assert pixels.list_scenes(tmp_path / "nope") == []


def test_a_scene_is_usable_only_after_the_processing_delay_and_never_from_the_future(labels, tmp_path):
    pixels.save_scene("2025-08-01", "A", scene(labels), tmp_path)
    assert pixels.best_scene_before("2025-08-01", labels, directory=tmp_path) is None, "not usable the day it was taken"
    assert pixels.best_scene_before("2025-07-30", labels, directory=tmp_path) is None, "a future scene must never be used"
    got = pixels.best_scene_before("2025-08-03", labels, directory=tmp_path)
    assert got is not None and got[0] == "2025-08-01" and got[2] == 1.0


def test_it_prefers_the_freshest_scene_with_enough_clear_pixels(labels, tmp_path):
    pixels.save_scene("2025-08-01", "A", scene(labels), tmp_path)
    pixels.save_scene("2025-08-06", "B", scene(labels, clouds=0.9, seed=3), tmp_path)    # newer but mostly cloud
    got = pixels.best_scene_before("2025-08-10", labels, min_coverage=0.4, directory=tmp_path)
    assert got[0] == "2025-08-01"


def test_old_scenes_and_scenes_of_a_different_grid_are_ignored(labels, tmp_path):
    pixels.save_scene("2025-05-01", "A", scene(labels), tmp_path)
    assert pixels.best_scene_before("2025-08-01", labels, max_age_days=45, directory=tmp_path) is None
    pixels.save_scene("2025-07-30", "B", np.zeros((3, 3), dtype="float16"), tmp_path)
    assert pixels.best_scene_before("2025-08-03", labels, directory=tmp_path) is None
