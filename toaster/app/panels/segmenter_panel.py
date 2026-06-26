"""Choose a segmenter, set its parameters, and run it (optionally on the selection)."""

from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from toaster.segment import available_segmenters, get_segmenter
from toaster.segment.base import Segmenter

__all__ = ["SegmenterPanel"]


class SegmenterPanel(QWidget):
    """Builds a configured :class:`~toaster.segment.base.Segmenter` and asks to run it."""

    run_requested = Signal(object, bool)  # (segmenter, scope_to_selection)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Segmenter"))

        self._choice = QComboBox()
        self._choice.addItems(available_segmenters())
        self._choice.currentTextChanged.connect(self._sync_params)
        layout.addWidget(self._choice)

        form = QFormLayout()
        self._eps = QDoubleSpinBox()
        self._eps.setRange(0.001, 1000.0)
        self._eps.setSingleStep(0.05)
        self._eps.setValue(0.5)
        self._min_samples = QSpinBox()
        self._min_samples.setRange(1, 10000)
        self._min_samples.setValue(10)
        self._min_cluster_size = QSpinBox()
        self._min_cluster_size.setRange(2, 100000)
        self._min_cluster_size.setValue(25)

        self._eps_label = QLabel("eps")
        self._min_samples_label = QLabel("min_samples")
        self._mcs_label = QLabel("min_cluster_size")
        form.addRow(self._eps_label, self._eps)
        form.addRow(self._min_samples_label, self._min_samples)
        form.addRow(self._mcs_label, self._min_cluster_size)
        layout.addLayout(form)

        self._scope = QCheckBox("Run on current selection only")
        self._scope.setChecked(True)
        layout.addWidget(self._scope)

        run = QPushButton("Run")
        run.clicked.connect(self._emit_run)
        layout.addWidget(run)
        layout.addStretch(1)

        self._sync_params(self._choice.currentText())

    def _sync_params(self, name: str) -> None:
        is_dbscan = name == "dbscan"
        for w in (self._eps, self._eps_label, self._min_samples, self._min_samples_label):
            w.setVisible(is_dbscan)
        for w in (self._min_cluster_size, self._mcs_label):
            w.setVisible(not is_dbscan)

    def _build_segmenter(self) -> Segmenter:
        name = self._choice.currentText()
        if name == "dbscan":
            return get_segmenter(
                "dbscan", eps=self._eps.value(), min_samples=self._min_samples.value()
            )
        if name == "hdbscan":
            return get_segmenter("hdbscan", min_cluster_size=self._min_cluster_size.value())
        return get_segmenter(name)

    def _emit_run(self) -> None:
        self.run_requested.emit(self._build_segmenter(), self._scope.isChecked())
