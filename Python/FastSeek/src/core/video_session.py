from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import cv2
import numpy as np
from decord import VideoReader, cpu

@dataclass
class VideoMetadata:
    path: str
    frame_count: int
    width: int
    height: int
    fps: float


class VideoSession:
    """
    Synchronous Decord wrapper.
    Threading/caching stay outside this class.
    """

    def __init__(self, path: str, preview_width: Optional[int] = 640) -> None:
        self.path = path
        self.preview_width = preview_width

        self.vr = VideoReader(path, ctx=cpu(0))
        self.frame_count = len(self.vr)

        first = self.vr[0].asnumpy()
        self.height, self.width = first.shape[:2]

        try:
            self.fps = float(self.vr.get_avg_fps())
        except Exception:
            self.fps = 0.0

    def get_metadata(self) -> VideoMetadata:
        return VideoMetadata(
            path=self.path,
            frame_count=self.frame_count,
            width=self.width,
            height=self.height,
            fps=self.fps,
        )

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

    def get_frame(self, index: int) -> np.ndarray:
        index = max(0, min(index, self.frame_count - 1))
        frame = self.vr[index].asnumpy()  # RGB
        return self._resize_for_preview(frame)

    def get_batch(self, indices: Iterable[int]) -> List[np.ndarray]:
        cleaned = []
        for idx in indices:
            idx = max(0, min(int(idx), self.frame_count - 1))
            cleaned.append(idx)

        if not cleaned:
            return []

        batch = self.vr.get_batch(cleaned).asnumpy()  # RGB
        return [self._resize_for_preview(frame) for frame in batch]