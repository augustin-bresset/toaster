from __future__ import annotations

import numpy as np

from toaster.core import LabelSchema


def test_colors_for_is_vectorized(schema):
    colors = schema.colors_for(np.array([1, 2, 0]))
    assert colors.tolist() == [[255, 0, 0], [0, 255, 0], [0, 0, 0]]
    assert colors.dtype == np.uint8


def test_out_of_range_labels_clamp_to_unlabeled(schema):
    colors = schema.colors_for(np.array([99, -1]))
    assert colors.tolist() == [[0, 0, 0], [0, 0, 0]]


def test_from_config_parses_hex_and_names():
    cfg = {
        "ignore_index": 0,
        "color_map": {0: "#000000", 1: "#ff0000", 2: [0, 0, 255]},
        "semantic_map": {1: "car", 2: "road"},
    }
    s = LabelSchema.from_config(cfg)
    assert s.unlabeled_id == 0
    assert s.get(1).name == "car"
    assert s.get(1).color == (255, 0, 0)
    assert s.get(2).color == (0, 0, 255)


def test_set_color_updates_lut(schema):
    assert schema.colors_for(np.array([1]))[0].tolist() == [255, 0, 0]
    schema.set_color(1, (10, 20, 30))
    assert schema.get(1).color == (10, 20, 30)
    # The cached LUT is invalidated, so colours_for reflects the change.
    assert schema.colors_for(np.array([1]))[0].tolist() == [10, 20, 30]


def test_yaml_round_trip(tmp_path):
    import yaml

    cfg = {"ignore_index": 0, "color_map": {0: "#010203", 1: "#0a0b0c"}}
    path = tmp_path / "schema.yaml"
    path.write_text(yaml.safe_dump(cfg))
    s = LabelSchema.from_yaml(path)
    assert s.get(1).color == (10, 11, 12)
