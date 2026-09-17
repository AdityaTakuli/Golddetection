"""
The material stage as a single per-frame call.

Keeps the integration surface in GoldNormal.py small: build one
MaterialStage, feed it (frame, detections) every frame, read a verdict
when one is ready.

Ordering note -- the detector is localisation-only here. It answers "where
is the object", never "what is it made of". A YOLO trained on RGB crops
cannot answer the second question, because under uncontrolled light gold
and yellow plastic are genuinely the same pixels; the information is
absent from the input, and no amount of model capacity manufactures it.
The material call belongs to the physics, which is what this stage runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .colour import bgr_u8_to_linear_rgb, clipped_mask
from .config import PhysicsConfig
from .reference import ChipReading, ReferenceChips
from .specular import MaterialVerdict, Verdict
from .tracker import Detection, MaterialTracker, ObjectTrack

log = logging.getLogger("physics.pipeline")


@dataclass
class StageResult:
    tracks: List[ObjectTrack] = field(default_factory=list)
    chips: Optional[ChipReading] = None
    verdict: Optional[MaterialVerdict] = None
    frame_clip_fraction: float = 0.0

    @property
    def gold_confirmed(self) -> bool:
        return self.verdict is not None and self.verdict.state is Verdict.GOLD_LIKE


class MaterialStage:
    """Illuminant estimation + per-object sweep tracking, one call a frame."""

    def __init__(self, cfg: PhysicsConfig):
        self.cfg = cfg
        self.chips = ReferenceChips(cfg.chips)
        self.tracker = MaterialTracker(cfg.thresholds)
        self._warned_uncalibrated = False
        self._warned_clipping = False

    def reset(self) -> None:
        """Start a fresh capture event."""
        self.tracker.reset()
        self.chips.reset()
        self._warned_clipping = False

    def process(self, frame_bgr: np.ndarray, detections: Sequence[Detection]) -> StageResult:
        """Advance every tracked object by one frame.

        `frame_bgr` and the detection masks must be from the same frame --
        pairing a stale mask with fresh pixels samples background as
        though it were the object.
        """
        t = self.cfg.thresholds
        chips = self.chips.measure(frame_bgr, t.clip_level)

        if chips.note:
            log.debug("chips: %s", chips.note)
        if not chips.calibrated and not self._warned_uncalibrated:
            self._warned_uncalibrated = True
            log.warning(
                "No white reference patch configured -- chroma is measured against an "
                "assumed neutral illuminant and will drift with the lamp and ambient "
                "light. Run tools/calibrate_chips.py to fix this."
            )

        rgb_lin = bgr_u8_to_linear_rgb(frame_bgr)
        clip = clipped_mask(frame_bgr, t.clip_level)
        frame_clip = float(clip.mean())

        if frame_clip > 0.25 and not self._warned_clipping:
            self._warned_clipping = True
            log.warning(
                "%.0f%% of the frame is saturated -- lower the exposure. Clipped "
                "pixels carry no colour, so the material test cannot run on them.",
                frame_clip * 100,
            )

        tracks = self.tracker.update(rgb_lin, clip, detections, chips)
        return StageResult(
            tracks=tracks,
            chips=chips,
            verdict=self.tracker.best_verdict(),
            frame_clip_fraction=frame_clip,
        )

    # -- operator feedback ------------------------------------------------
    def hud_lines(self, result: StageResult) -> List[str]:
        """Short status lines for the on-screen HUD."""
        lines: List[str] = []
        if result.chips is not None and not result.chips.calibrated:
            lines.append("chips: NOT CALIBRATED")
        elif result.chips is not None and not result.chips.white_ok:
            lines.append(f"chips: {result.chips.note[:38]}")
        if result.frame_clip_fraction > 0.02:
            lines.append(f"clipped: {result.frame_clip_fraction:.0%}")
        for tr in result.tracks[:3]:
            lines.append(f"#{tr.track_id}: {tr.progress()}")
        return lines
