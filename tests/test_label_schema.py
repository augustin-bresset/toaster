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


def test_add_class_assigns_next_id_and_colours(schema):
    cls = schema.add_class("tree", color=(1, 2, 3))
    assert cls.id == 3  # one past the previous max (2)
    assert schema.get(3).name == "tree"
    # The LUT was invalidated, so the new class colours immediately.
    assert schema.colors_for(np.array([3]))[0].tolist() == [1, 2, 3]


def test_add_class_auto_colour_is_unused(schema):
    a = schema.add_class("x")
    b = schema.add_class("y")
    assert a.color != b.color  # auto-assigned colours differ
    assert a.color not in {(0, 0, 0), (255, 0, 0), (0, 255, 0)}  # avoids existing ones


def test_rename_class(schema):
    schema.rename(1, "car")
    assert schema.get(1).name == "car"
    assert schema.get(1).color == (255, 0, 0)  # colour untouched


def test_remove_class(schema):
    schema.remove(1)
    assert 1 not in schema
    assert [c.id for c in schema.classes] == [0, 2]
    # A now-orphaned label clamps to unlabeled rather than crashing.
    assert schema.colors_for(np.array([1]))[0].tolist() == [0, 0, 0]


def test_remove_unlabeled_is_refused(schema):
    import pytest

    with pytest.raises(ValueError, match="unlabeled"):
        schema.remove(0)


def test_to_yaml_round_trip(schema, tmp_path):
    schema.add_class("tree", color=(1, 2, 3))
    out = schema.to_yaml(tmp_path / "out.yaml")
    reloaded = LabelSchema.from_yaml(out)
    assert reloaded.unlabeled_id == schema.unlabeled_id
    assert {c.id: (c.name, c.color) for c in reloaded.classes} == {
        c.id: (c.name, c.color) for c in schema.classes
    }
