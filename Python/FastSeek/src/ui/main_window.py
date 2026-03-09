from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QLineEdit,
)

from src.core.frame_cache import FrameCache
from src.core.video_session import VideoSession
from src.core.decode_worker import DecodeWorker

class VideoViewer(QWidget):
    def __init__(self, video_path: Optional[str] = None) -> None:
        super().__init__()

        self.setWindowTitle("Fast Decord Frame Viewer")
        self.resize(1100, 800)

        self.session: Optional[VideoSession] = None
        self.cache: Optional[FrameCache] = None
        self.worker: Optional[DecodeWorker] = None

        self.current_index = 0
        self.last_requested_index = 0
        self.drag_pending_index: Optional[int] = None

        self.drag_timer = QTimer(self)
        self.drag_timer.setInterval(33)  # ~30 fps max
        self.drag_timer.timeout.connect(self._flush_drag_request)

        self.image_label = QLabel("Open a video")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 360)
        self.image_label.setStyleSheet("background: #111; color: #ddd;")

        self.info_label = QLabel("No video loaded")
        self.status_label = QLabel("Idle")

        self.open_button = QPushButton("Open")
        self.prev_button = QPushButton("<<")
        self.next_button = QPushButton(">>")

        self.frame_edit = QLineEdit()
        self.frame_edit.setPlaceholderText("Frame #")
        self.frame_edit.setFixedWidth(120)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setTracking(True)

        top_row = QHBoxLayout()
        top_row.addWidget(self.open_button)
        top_row.addWidget(self.prev_button)
        top_row.addWidget(self.next_button)
        top_row.addWidget(self.frame_edit)
        top_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(top_row)
        layout.addWidget(self.image_label, stretch=1)
        layout.addWidget(self.slider)
        layout.addWidget(self.info_label)
        layout.addWidget(self.status_label)

        self.open_button.clicked.connect(self.open_dialog)
        self.prev_button.clicked.connect(lambda: self.step(-1))
        self.next_button.clicked.connect(lambda: self.step(+1))
        self.frame_edit.returnPressed.connect(self.jump_to_frame)

        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderMoved.connect(self._on_slider_moved)
        self.slider.sliderReleased.connect(self._on_slider_released)

        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self.step(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self.step(+1))
        QShortcut(QKeySequence("Shift+Left"), self, activated=lambda: self.step(-10))
        QShortcut(QKeySequence("Shift+Right"), self, activated=lambda: self.step(+10))
        QShortcut(QKeySequence("Ctrl+Left"), self, activated=lambda: self.step(-100))
        QShortcut(QKeySequence("Ctrl+Right"), self, activated=lambda: self.step(+100))

        if video_path:
            self.load_video(video_path)

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            self.worker.stop()
        super().closeEvent(event)

    def open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open video",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.wmv *.webm);;All files (*)",
        )
        if path:
            self.load_video(path)

    def load_video(self, path: str) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

        self.session = VideoSession(path=path, preview_width=960)
        self.cache = FrameCache(capacity=200)
        self.worker = DecodeWorker(self.session, self.cache)

        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

        meta = self.session.get_metadata()
        self.slider.setMaximum(meta.frame_count - 1)
        self.current_index = 0
        self.last_requested_index = 0

        self.info_label.setText(
            f"{os.path.basename(meta.path)} | "
            f"{meta.width}x{meta.height} | "
            f"{meta.frame_count} frames | "
            f"{meta.fps:.3f} fps"
        )
        self.status_label.setText("Loaded. Requesting first frame...")

        self.request_frame(0, settled=True)

    def _build_prefetch_window(self, index: int) -> List[int]:
        if self.session is None:
            return []

        direction = 1 if index >= self.last_requested_index else -1
        frame_count = self.session.frame_count

        if direction >= 0:
            back_n = 5
            fwd_n = 20
        else:
            back_n = 20
            fwd_n = 5

        indices = []

        for i in range(index - back_n, index):
            if 0 <= i < frame_count:
                indices.append(i)

        for i in range(index + 1, index + 1 + fwd_n):
            if 0 <= i < frame_count:
                indices.append(i)

        return indices

    def request_frame(self, index: int, settled: bool) -> None:
        if self.session is None or self.worker is None:
            return

        index = max(0, min(index, self.session.frame_count - 1))

        if settled:
            prefetch = self._build_prefetch_window(index)
            self.worker.request_frame(index, prefetch_indices=prefetch)
        else:
            self.worker.request_drag_frame(index)

        self.last_requested_index = index
        self.status_label.setText(f"Requesting frame {index}...")

    def step(self, delta: int) -> None:
        if self.session is None:
            return
        target = max(0, min(self.current_index + delta, self.session.frame_count - 1))
        self.slider.setValue(target)
        self.request_frame(target, settled=True)

    def jump_to_frame(self) -> None:
        if self.session is None:
            return
        text = self.frame_edit.text().strip()
        if not text:
            return
        try:
            target = int(text)
        except ValueError:
            self.status_label.setText("Invalid frame number")
            return

        target = max(0, min(target, self.session.frame_count - 1))
        self.slider.setValue(target)
        self.request_frame(target, settled=True)

    def _on_slider_pressed(self) -> None:
        self.drag_pending_index = self.slider.value()

    def _on_slider_moved(self, value: int) -> None:
        self.drag_pending_index = value
        if not self.drag_timer.isActive():
            self.drag_timer.start()

    def _on_slider_released(self) -> None:
        self.drag_timer.stop()
        value = self.slider.value()
        self.drag_pending_index = None
        self.request_frame(value, settled=True)

    def _flush_drag_request(self) -> None:
        if self.drag_pending_index is None:
            return
        self.request_frame(self.drag_pending_index, settled=False)

    def _numpy_to_pixmap(self, frame_rgb: np.ndarray) -> QPixmap:
        frame_rgb = np.ascontiguousarray(frame_rgb)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(image)

    def _on_frame_ready(self, index: int, frame_rgb: np.ndarray, latency_ms: float, cache_hit: bool) -> None:
        self.current_index = index

        self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(False)

        pixmap = self._numpy_to_pixmap(frame_rgb)
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

        cache_size = len(self.cache) if self.cache is not None else 0
        source = "cache" if cache_hit else "decode"
        self.status_label.setText(
            f"Frame {index} | {source} | {latency_ms:.1f} ms | cache size: {cache_size}"
        )

    def _on_worker_error(self, message: str) -> None:
        self.status_label.setText(f"Worker error: {message}")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.image_label.pixmap() is not None:
            pixmap = self.image_label.pixmap()
            scaled = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled)