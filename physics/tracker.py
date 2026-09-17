"""
Per-object sweep tracking and the sweep state machine.

This replaces the threaded `sweep_worker` in SegmenterTester.py, which had
a data race that silently corrupted the measurement: the main thread wrote
`det.mask` while the worker read it against a *separately* latched
`latest_lab_frame`. Nothing crashed -- numpy assignment is by reference --
but the worker could pair a mask from frame N with pixels from frame N-2,
sampling background as though it were the object.

The fix is not a lock, it is removing the concurrency. The main loop
already holds both the frame and the masks at the same instant, so
accumulation happens synchronously, in-loop, on a guaranteed-consistent
pair. It is also cheaper: no thread per object, no shared-frame copies.

Sweep state machine (per object):

  BASELINE  collect N frames of ambient brightness
  WAITING   watch for frame p99 luminance to rise above baseline * rise
  SWEEPING  lamp is on the piece; track the peak
  DECIDED   peak has fallen away -> run the discriminator

If no excursion ever arrives the track times out and reports
INVALID_NO_SWEEP rather than guessing from ambient light alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import Thresholds
from .reference import ChipReading
from .specular import MaterialVerdict, SweepAccumulator, Verdict

log = logging.getLogger("physics.tracker")

Box = Tuple[int, int, int, int]


class SweepPhase(Enum):
    BASELINE = auto()
    WAITING = auto()
    SWEEPING = auto()
    DECIDED = auto()


@dataclass
class Detection:
    """One detector output for a frame. Localisation only -- the detector
    is deliberately not asked what material the object is."""

    box: Box
    mask: np.ndarray          # bool, full-frame
    confidence: float = 0.0
    class_name: str = ""


@dataclass
class ObjectTrack:
    track_id: int
    box: Box
    mask: np.ndarray
    confidence: float = 0.0
    class_name: str = ""
    phase: SweepPhase = SweepPhase.BASELINE
    verdict: MaterialVerdict = field(default_factory=MaterialVerdict)
    frames: int = 0
    last_seen: int = 0
    accumulator: Optional[SweepAccumulator] = None

    _baseline_samples: List[float] = field(default_factory=list, repr=False)
    baseline: float = 0.0
    peak: float = 0.0
    frames_above: int = 0

    @property
    def decided(self) -> bool:
        return self.phase is SweepPhase.DECIDED

    def progress(self) -> str:
        if self.phase is SweepPhase.BASELINE:
            return "measuring ambient"
        if self.phase is SweepPhase.WAITING:
            return "sweep the lamp across it"
        if self.phase is SweepPhase.SWEEPING:
            return "sweeping"
        return self.verdict.state.value


def _overlap_fraction(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area = max((ax2 - ax1) * (ay2 - ay1), 1)
    return inter / area


class MaterialTracker:
    """Associates detections across frames and runs each object's sweep."""

    def __init__(self, thresholds: Thresholds):
        self.t = thresholds
        self.tracks: List[ObjectTrack] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def update(
        self,
        rgb_lin: np.ndarray,
        clip: np.ndarray,
        detections: Sequence[Detection],
        chips: ChipReading,
    ) -> List[ObjectTrack]:
        """Advance every track by one frame.

        `rgb_lin`, `clip` and the detection masks must all come from the
        SAME frame -- that consistency is the whole reason this is
        synchronous. Do not cache a mask and pass it next frame.
        """
        for tr in self.tracks:
            tr.last_seen += 1

        for det in detections:
            tr = self._match(det.box)
            if tr is None:
                tr = ObjectTrack(
                    track_id=self._next_id,
                    box=det.box,
                    mask=det.mask,
                    confidence=det.confidence,
                    class_name=det.class_name,
                    accumulator=SweepAccumulator(self.t, seed=self._next_id),
                )
                self._next_id += 1
                self.tracks.append(tr)
                log.info("[track %d] new object (%s conf=%.2f)",
                         tr.track_id, det.class_name or "object", det.confidence)
            else:
                tr.box = det.box
                tr.mask = det.mask
                tr.confidence = det.confidence
                if det.class_name:
                    tr.class_name = det.class_name
            tr.last_seen = 0
            self._advance(tr, rgb_lin, clip, chips)

        ttl = self.t.track_ttl_frames
        self.tracks = [t for t in self.tracks if t.last_seen <= ttl or t.decided]
        return list(self.tracks)

    def _match(self, box: Box) -> Optional[ObjectTrack]:
        best, best_ov = None, self.t.match_overlap
        for tr in self.tracks:
            ov = _overlap_fraction(box, tr.box)
            if ov > best_ov:
                best, best_ov = tr, ov
        return best

    def _advance(self, tr: ObjectTrack, rgb_lin, clip, chips: ChipReading) -> None:
        if tr.decided:
            return

        tr.frames += 1
        p99 = tr.accumulator.add_frame(
            rgb_lin, tr.mask, clip, chips.illuminant_rgb, chips.calibrated
        )
        t = self.t

        if tr.phase is SweepPhase.BASELINE:
            if p99 > 0:
                tr._baseline_samples.append(p99)
            if len(tr._baseline_samples) >= t.baseline_frames:
                tr.baseline = float(np.median(tr._baseline_samples))
                tr.phase = SweepPhase.WAITING
                log.info("[track %d] baseline %.4f -- waiting for the lamp",
                         tr.track_id, tr.baseline)
            return

        if tr.phase is SweepPhase.WAITING:
            if tr.baseline > 0 and p99 >= tr.baseline * t.rise_factor:
                tr.phase = SweepPhase.SWEEPING
                tr.peak = p99
                tr.frames_above = 1
                log.info("[track %d] lamp detected (%.2fx baseline)",
                         tr.track_id, p99 / max(tr.baseline, 1e-6))
            elif tr.frames > t.timeout_frames:
                self._decide(tr, timed_out=True)
            return

        if tr.phase is SweepPhase.SWEEPING:
            tr.peak = max(tr.peak, p99)
            if p99 >= tr.baseline * t.rise_factor:
                tr.frames_above += 1
            fell = p99 <= tr.peak * t.fall_factor
            if fell and tr.frames_above >= t.min_sweep_frames:
                self._decide(tr)
            elif tr.frames > t.timeout_frames:
                self._decide(tr, timed_out=True)

    def _decide(self, tr: ObjectTrack, timed_out: bool = False) -> None:
        # The temporal excursion -- how much brighter the lamp actually made
        # the piece -- is what licenses a verdict at all. Curvature alone
        # produces a luminance range with no specular content, and every
        # material holds its chroma across pure diffuse shading.
        ratio = tr.peak / tr.baseline if tr.baseline > 0 else 0.0
        observed = (tr.phase is SweepPhase.SWEEPING
                    and tr.frames_above >= self.t.min_sweep_frames)
        tr.verdict = tr.accumulator.decide(sweep_ratio=ratio, sweep_observed=observed)
        if timed_out and tr.verdict.state is Verdict.INVALID_NO_SWEEP:
            tr.verdict.reason = (
                f"no usable sweep in {tr.frames} frames (peak {ratio:.2f}x ambient) -- "
                "sweep a lamp or phone torch across the piece"
            )
        tr.phase = SweepPhase.DECIDED
        log.info("[track %d] %s :: %s", tr.track_id,
                 tr.verdict.state.value, tr.verdict.reason)

    # -- convenience for the pipeline ------------------------------------
    def best_verdict(self) -> Optional[MaterialVerdict]:
        """The verdict to record for this capture event.

        A confirmed gold outweighs anything else; otherwise report the most
        informative non-pending result so the operator sees *why* nothing
        was confirmed rather than a bare negative.
        """
        decided = [t.verdict for t in self.tracks if t.decided]
        if not decided:
            return None
        for v in decided:
            if v.state is Verdict.GOLD_LIKE:
                return v
        order = [Verdict.DIELECTRIC, Verdict.NON_GOLD_METAL, Verdict.UNCERTAIN,
                 Verdict.INVALID_CLIPPED, Verdict.INVALID_NO_SWEEP]
        for state in order:
            for v in decided:
                if v.state is state:
                    return v
        return decided[0]
