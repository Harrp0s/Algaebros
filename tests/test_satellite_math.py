from datetime import datetime, timezone

import numpy as np

from algaebro import config, satellite


class FakeItem:
    def __init__(self, props, dt):
        self.properties = props
        self.datetime = dt


def test_offset_from_processing_baseline():
    assert satellite.dn_offset(FakeItem({"s2:processing_baseline": "05.10"}, datetime(2023, 7, 1, tzinfo=timezone.utc))) == -1000
    assert satellite.dn_offset(FakeItem({"s2:processing_baseline": "03.00"}, datetime(2020, 7, 1, tzinfo=timezone.utc))) == 0


def test_offset_falls_back_to_date():
    assert satellite.dn_offset(FakeItem({}, datetime(2023, 7, 1, tzinfo=timezone.utc))) == -1000
    assert satellite.dn_offset(FakeItem({}, datetime(2021, 7, 1, tzinfo=timezone.utc))) == 0


def test_reflectance_masks_nodata_and_nonphysical():
    dn = np.array([0, 1000, 1500, 500], dtype="uint16")
    refl = satellite.to_reflectance(dn, -1000)
    assert np.isnan(refl[0])            # DN 0 is no-data
    assert np.isnan(refl[1])            # exactly zero after the offset: unusable
    assert abs(refl[2] - 0.05) < 1e-6
    assert np.isnan(refl[3])            # negative after the offset


def test_ndci_value_and_bad_inputs():
    v = satellite.ndci(np.array([0.02, np.nan, 0.0], "float32"), np.array([0.04, 0.04, 0.0], "float32"))
    assert abs(v[0] - (0.02 / 0.06)) < 1e-5
    assert np.isnan(v[1]) and np.isnan(v[2])


def test_zone_stats_needs_enough_clean_pixels():
    labels = np.zeros((10, 10), "int16")
    labels[:, :5] = 1      # 50 px
    labels[:, 5:] = 2      # 50 px
    index = np.full((10, 10), 0.1, "float32")
    valid = np.ones((10, 10), bool)
    valid[:, 5:] = False   # zone 2 is entirely cloudy
    rows = satellite.zone_stats(index, valid, labels, 2, min_pixels=20, min_frac=0.3)
    assert [r["zone"] for r in rows] == [1]
    assert abs(rows[0]["ndci"] - 0.1) < 1e-6


def test_cloudy_scene_is_skipped():
    scl = np.full((10, 10), 9, "uint8")  # cloud
    arrs = {"b04": np.full((10, 10), 0.02, "float32"), "b05": np.full((10, 10), 0.04, "float32"), "scl": scl}
    res = satellite.process_arrays(arrs, np.ones((10, 10), "int16"), 1)
    assert res["status"] == "skipped_cloudy" and res["rows"] == []


def test_clear_scene_gives_zone_medians():
    scl = np.full((10, 10), config.SCL_WATER, "uint8")
    arrs = {"b04": np.full((10, 10), 0.02, "float32"), "b05": np.full((10, 10), 0.04, "float32"), "scl": scl}
    labels = np.ones((10, 10), "int16")
    res = satellite.process_arrays(arrs, labels, 1)
    assert res["status"] == "ok"
    assert abs(res["rows"][0]["ndci"] - (0.02 / 0.06)) < 1e-4


def test_land_pixels_do_not_count_as_water():
    scl = np.full((10, 10), 4, "uint8")  # vegetation
    arrs = {"b04": np.full((10, 10), 0.02, "float32"), "b05": np.full((10, 10), 0.04, "float32"), "scl": scl}
    res = satellite.process_arrays(arrs, np.ones((10, 10), "int16"), 1)
    assert res["rows"] == []


def test_scl_nodata_becomes_zero_not_garbage():
    raw = np.array([[6.0, np.nan], [9.0, np.inf]], dtype="float64")
    out = satellite.scl_to_uint8(raw)
    assert out.dtype == np.uint8
    assert out.tolist() == [[6, 0], [9, 0]]
    water = out == config.SCL_WATER
    assert water.sum() == 1, "a no-data pixel must never be mistaken for water"
