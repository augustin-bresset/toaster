"""Controller tests with a fake viewer — exercises the workflow without Qt/VTK."""

from __future__ import annotations

import numpy as np

from toaster.core import Grouping, Session
from toaster.interaction import InteractionController


class FakeViewer:
    """Records what the controller asks the renderer to do."""

    def __init__(self):
        self.last_highlight = None
        self.recolored = []
        self.point_cb = None
        self.box_cb = None
        self.cloud_colors = None
        self.visible_mask = None

    def set_cloud(self, xyz, colors):
        self.cloud_colors = colors

    def update_colors(self, indices, colors):
        self.recolored.append((np.asarray(indices), np.asarray(colors)))

    def highlight(self, indices):
        self.last_highlight = np.asarray(indices)

    def clear_highlight(self):
        self.last_highlight = np.empty(0, dtype=np.int64)

    def set_point_pick_callback(self, cb):
        self.point_cb = cb

    def set_box_pick_callback(self, cb):
        self.box_cb = cb

    def set_pick_mode(self, mode):
        self.pick_mode = mode

    def set_visible_mask(self, mask):
        self.visible_mask = None if mask is None else np.asarray(mask)

    def reset_camera(self):
        pass

    def render(self):
        pass


def _session(two_clusters, schema):
    two_clusters.ensure_labels(schema.unlabeled_id)
    return Session(two_clusters, schema)


def test_pick_without_grouping_selects_single_point(two_clusters, schema):
    session = _session(two_clusters, schema)
    viewer = FakeViewer()
    ctl = InteractionController(session, viewer)
    ctl.on_pick(7)
    assert session.selection.indices.tolist() == [7]
    assert viewer.last_highlight.tolist() == [7]


def test_pick_with_active_grouping_selects_whole_group(two_clusters, schema):
    session = _session(two_clusters, schema)
    # Group 0 = first 50 points, group 1 = the rest.
    gid = np.where(np.arange(two_clusters.n) < 50, 0, 1).astype(np.int32)
    session.add_grouping(Grouping(gid))
    ctl = InteractionController(session, FakeViewer())
    ctl.on_pick(3)  # a point in group 0
    assert session.selection.count == 50
    assert session.selection.indices.max() < 50


def test_shift_adds_ctrl_subtracts(two_clusters, schema):
    session = _session(two_clusters, schema)
    ctl = InteractionController(session, FakeViewer())
    ctl.on_pick(1)
    ctl.on_pick(2, frozenset({"shift"}))
    assert session.selection.indices.tolist() == [1, 2]
    ctl.on_pick(1, frozenset({"ctrl"}))
    assert session.selection.indices.tolist() == [2]


def test_assign_writes_labels_and_recolors(two_clusters, schema):
    session = _session(two_clusters, schema)
    viewer = FakeViewer()
    ctl = InteractionController(session, viewer)
    ctl.on_pick(5)
    ctl.assign(2)
    assert session.cloud.labels[5] == 2
    assert viewer.recolored  # the touched point was recoloured
    # Selection is cleared after assigning.
    assert session.selection.is_empty()


def _grouped_session(two_clusters, schema):
    """Session with a 2-group grouping active (group 0 = first 50, group 1 = rest)."""
    two_clusters.ensure_labels(schema.unlabeled_id)
    session = Session(two_clusters, schema)
    gid = np.where(np.arange(two_clusters.n) < 50, 0, 1).astype(np.int32)
    session.add_grouping(Grouping(gid, suggested_labels={1: 2}))
    return session


def test_select_group_selects_whole_segment(two_clusters, schema):
    session = _grouped_session(two_clusters, schema)
    ctl = InteractionController(session, FakeViewer())
    ctl.select_group(0)
    assert session.selection.count == 50
    assert session.selection.indices.max() < 50


def test_assign_group_labels_the_segment(two_clusters, schema):
    session = _grouped_session(two_clusters, schema)
    viewer = FakeViewer()
    ctl = InteractionController(session, viewer)
    ctl.set_active_class(1)
    n = ctl.assign_group(0)
    assert n == 50
    assert (session.cloud.labels[:50] == 1).all()
    assert (session.cloud.labels[50:] == 0).all()
    assert viewer.recolored  # labelled points are repainted in any view


def test_apply_suggested_single_and_all(two_clusters, schema):
    session = _grouped_session(two_clusters, schema)
    ctl = InteractionController(session, FakeViewer())
    # Only group 1 carries a suggestion (-> class 2).
    n = ctl.apply_suggested(1)
    assert n == 50
    assert (session.cloud.labels[50:] == 2).all()
    # Group 0 has no suggestion -> no-op.
    assert ctl.apply_suggested(0) == 0
    # "All suggested" applies every group that has one.
    session2 = _grouped_session(two_clusters, schema)
    ctl2 = InteractionController(session2, FakeViewer())
    assert ctl2.apply_suggested(None) == 50


def test_group_ops_noop_without_active_grouping(two_clusters, schema):
    two_clusters.ensure_labels(schema.unlabeled_id)
    session = Session(two_clusters, schema)  # no grouping
    ctl = InteractionController(session, FakeViewer())
    assert ctl.assign_group(0) == 0
    assert ctl.apply_suggested() == 0


def test_group_visibility_commands(two_clusters, schema):
    session = _grouped_session(two_clusters, schema)  # group 0 = first 50, group 1 = rest
    viewer = FakeViewer()
    ctl = InteractionController(session, viewer)

    # Hide group 0 -> its points masked off, group 1 still visible.
    ctl.set_group_visibility(0, False)
    assert viewer.visible_mask is not None
    assert viewer.visible_mask[:50].sum() == 0
    assert viewer.visible_mask[50:].all()
    assert next(s for s in ctl.snapshot().segments if s.id == 0).visible is False

    # Hide all -> every group masked off.
    ctl.hide_all_groups()
    assert viewer.visible_mask is not None and viewer.visible_mask.sum() == 0
    assert all(s.visible is False for s in ctl.snapshot().segments)

    # Show all -> mask cleared.
    ctl.show_all_groups()
    assert viewer.visible_mask is None
    assert all(s.visible for s in ctl.snapshot().segments)


def test_assign_visible_groups_labels_only_checked(two_clusters, schema):
    session = _grouped_session(two_clusters, schema)  # group 0 = first 50, group 1 = last 50
    ctl = InteractionController(session, FakeViewer())

    # Uncheck (hide) group 0, then assign -> only the visible group 1 is labelled.
    ctl.set_group_visibility(0, False)
    ctl.set_active_class(2)
    n = ctl.assign_visible_groups()
    labels = session.cloud.labels
    assert n == 50
    assert (labels[:50] == schema.unlabeled_id).all()  # hidden group untouched
    assert (labels[50:] == 2).all()  # checked group labelled

    # It is a single undoable batch.
    ctl.undo()
    assert (session.cloud.labels == schema.unlabeled_id).all()


def test_clear_grouping_discards_segmentation_keeps_labels(two_clusters, schema):
    session = _grouped_session(two_clusters, schema)
    ctl = InteractionController(session, FakeViewer())
    ctl.set_display_mode("grouping")
    # Label a segment first — its labels must survive the grouping being dropped.
    labelled = ctl.assign_group(0, 1)
    assert labelled == 50

    ctl.clear_grouping()
    assert session.active_grouping is None
    assert session.groupings == []
    assert ctl.display_mode == "labels"
    assert (session.cloud.labels[:50] == 1).all()  # labels kept
    assert ctl.snapshot().segments == []
    # Clearing again is a harmless no-op.
    ctl.clear_grouping()
    assert session.active_grouping is None


def test_class_editing(two_clusters, schema):
    from toaster.core import Selection

    two_clusters.ensure_labels(schema.unlabeled_id)
    session = Session(two_clusters, schema)
    ctl = InteractionController(session, FakeViewer())

    # Add a class -> it becomes the active brush.
    new_id = ctl.add_class("tree", (1, 2, 3))
    assert session.active_class == new_id
    assert session.schema.get(new_id).color == (1, 2, 3)

    # Rename and recolour an existing class.
    ctl.rename_class(1, "floor")
    assert session.schema.get(1).name == "floor"
    ctl.set_class_color(1, (9, 9, 9))
    assert session.schema.get(1).color == (9, 9, 9)

    # Label points class 1, then remove class 1 -> they fall back to unlabeled.
    session.annotation.assign(Selection.from_indices([0, 1, 2], two_clusters.n), 1)
    ctl.remove_class(1)
    assert 1 not in [c.id for c in session.schema.classes]
    assert (session.cloud.labels[:3] == schema.unlabeled_id).all()


def test_snapshot_is_a_flat_serializable_read_model(two_clusters, schema):
    from dataclasses import asdict

    session = _grouped_session(two_clusters, schema)  # grouping with suggested {1: 2}
    ctl = InteractionController(session, FakeViewer())
    ctl.set_active_class(1)
    snap = ctl.snapshot()

    assert [c.id for c in snap.classes] == [0, 1, 2]
    assert snap.active_class == 1
    assert snap.class_name(1) == "a"
    assert snap.active_grouping is not None and snap.active_grouping.n_groups == 2
    assert len(snap.segments) == 2
    seg1 = next(s for s in snap.segments if s.id == 1)
    assert seg1.count == 50
    assert seg1.suggested == 2
    assert snap.has_suggestions is True
    # No numpy / domain objects: a plain dict (i.e. wire-ready) round-trips.
    asdict(snap)


def test_run_segmenter_sets_active_grouping(two_clusters, schema):
    from toaster.segment import get_segmenter

    session = _session(two_clusters, schema)
    ctl = InteractionController(session, FakeViewer())
    ctl.run_segmenter(get_segmenter("dbscan", eps=0.5, min_samples=5))
    assert session.active_grouping is not None
    assert session.active_grouping.n_groups == 2
    assert ctl.display_mode == "grouping"
