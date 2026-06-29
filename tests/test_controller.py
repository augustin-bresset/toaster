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


def test_run_segmenter_sets_active_grouping(two_clusters, schema):
    from toaster.segment import get_segmenter

    session = _session(two_clusters, schema)
    ctl = InteractionController(session, FakeViewer())
    ctl.run_segmenter(get_segmenter("dbscan", eps=0.5, min_samples=5))
    assert session.active_grouping is not None
    assert session.active_grouping.n_groups == 2
    assert ctl.display_mode == "grouping"
