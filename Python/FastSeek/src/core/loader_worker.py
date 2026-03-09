from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from decord import VideoReader, cpu
from PySide6.QtCore import QObject, Signal, Slot


@dataclass
class LoadResult:
    path: str
    frame_count: int
    width: int
    height: int
    fps: float
    first_frame_rgb: np.ndarray


class LoaderWorker(QObject):
    progress_changed = Signal(int, str, str)
    load_finished = Signal(object)
    load_failed = Signal(str)
    finished = Signal()

    def __init__(self, path: str, preview_width: Optional[int] = 960) -> None:
        super().__init__()
        self.path = path
        self.preview_width = preview_width

    def _emit_progress(self, percent: int, stage_text: str) -> None:
        self.progress_changed.emit(percent, "Loading video...", stage_text)

    def _resize_for_preview(self, frame_rgb: np.ndarray) -> np.ndarray:
        if self.preview_width is None or self.preview_width <= 0:
            return frame_rgb

        h, w = frame_rgb.shape[:2]
        if w <= self.preview_width:
            return frame_rgb

        scale = self.preview_width / float(w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @Slot()
    def run(self) -> None:
        vr: Optional[VideoReader] = None
        try:
            self._emit_progress(10, "Opening file")
            vr = VideoReader(self.path, ctx=cpu(0))

            self._emit_progress(30, "Reading metadata")
            frame_count = len(vr)

            try:
                fps = float(vr.get_avg_fps())
            except Exception:
                fps = 0.0

            self._emit_progress(50, "Preparing preview")
            first = vr[0].asnumpy()  # RGB
            height, width = first.shape[:2]

            self._emit_progress(75, "Decoding first frame")
            first_frame_rgb = self._resize_for_preview(first)

            self._emit_progress(100, "Finalizing")
            result = LoadResult(
                path=self.path,
                frame_count=frame_count,
                width=width,
                height=height,
                fps=fps,
                first_frame_rgb=first_frame_rgb,
            )
            self.load_finished.emit(result)

        except Exception as exc:
            self.load_failed.emit(f"Could not load video: {exc}")
        finally:
            self.finished.emit()