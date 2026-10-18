import numpy as np

from algaebro import zones
import make_synthetic_data as gen


def test_zones_sit_on_river_and_ignore_pond_and_bridges():
    freq = gen.make_river()
    labels = zones.build_zone_labels(freq)
    assert labels.max() == 10
    assert (labels[freq < 0.6] == 0).all(), "a zone pixel is not on water"
    assert (labels[20:28, 20:28] == 0).all(), "the small pond must be ignored"
    assert (labels[:, 60:62] == 0).all(), "bridge columns are land"


def test_zones_numbered_west_to_east_and_all_nonempty():
    labels = zones.build_zone_labels(gen.make_river())
    centres = []
    for z in range(1, labels.max() + 1):
        rr, cc = np.nonzero(labels == z)
        assert len(rr) > 40
        centres.append(cc.mean())
    assert centres == sorted(centres)


def test_erosion_trims_bank_pixels():
    freq = gen.make_river()
    raw = (freq >= 0.6).sum()
    kept = (zones.build_zone_labels(freq) > 0).sum()
    assert kept < raw
