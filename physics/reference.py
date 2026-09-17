"""
White / grey reference patch measurement.

One matte white patch permanently in frame converts every absolute colour
threshold in this system into a ratio. That matters more than it sounds:

  * it survives the lamp ageing, being swapped, or being a different colour
    temperature than the one you calibrated against;
  * it survives an IP/dome camera whose exposure and white balance you
    cannot reach over RTSP, because whatever the camera does to the object
    it also does to the patch;
  * it turns "is this pixel yellow" (meaningless) into "how far is this
    pixel's colour from the colour of the light" (the actual physics).

The grey patch is a second point on the response curve. It is optional and
used for a linearity sanity check -- if the measured white:grey ratio is far
from their nominal reflectance ratio, the camera is applying a tone curve
that the sRGB inversion in physics.colour does not model, and the slope
statistic will be biased.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .colour import bgr_u8_to_linear_rgb, clipped_mask, luminance
from .config import ChipConfig, Rect


@dataclass
class ChipReading:
    """One frame's view of the reference patches."""

    illuminant_rgb: np.ndarray       # linear RGB of the white patch
    calibrated: bool                 # False => assumed neutral, drifting
    white_ok: bool = False
    grey_ok: bool = False
    white_level: float = 0.0         # mean luminance of white patch
    grey_level: float = 0.0
    white_clip_fraction: float = 0.0
    linearity_error: Optional[float] = None
    note: str = ""

    @property
    def usable(self) -> bool:
        return self.calibrated and self.white_ok


def _rect_to_slice(rect: Rect, shape) -> tuple:
    h, w = shape[:2]
    x, y, rw, rh = rect
    x0 = int(round(x * w))
    y0 = int(round(y * h))
    x1 = int(round((x + rw) * w))
    y1 = int(round((y + rh) * h))
    x0, x1 = max(0, min(x0, x1)), min(w, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(h, max(y0, y1))
    return slice(y0, y1), slice(x0, x1)


def _trimmed_mean(px: np.ndarray, trim: float) -> np.ndarray:
    """Per-channel trimmed mean, robust to dust, glare and a stray shadow."""
    if px.size == 0:
        return np.zeros(3, dtype=np.float32)
    if trim <= 0:
        return px.mean(axis=0).astype(np.float32)
    lo = np.quantile(px, trim, axis=0)
    hi = np.quantile(px, 1.0 - trim, axis=0)
    keep = np.all((px >= lo) & (px <= hi), axis=1)
    sel = px[keep] if keep.any() else px
    return sel.mean(axis=0).astype(np.float32)


class ReferenceChips:
    """Measures the illuminant off the white patch, frame by frame."""

    NEUTRAL = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def __init__(self, cfg: ChipConfig):
        self.cfg = cfg
        self._smoothed: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._smoothed = None

    def measure(self, frame_bgr: np.ndarray, clip_level: int = 250) -> ChipReading:
        """Estimate the illuminant from this frame.

        Degrades gracefully: with no patch configured the illuminant is
        assumed neutral and `calibrated` is False, so the pipeline still
        runs -- it just cannot promise its thresholds will hold as the
        lighting changes.
        """
        if not self.cfg.configured:
            return ChipReading(
                illuminant_rgb=self.NEUTRAL.copy(),
                calibrated=False,
                note="no white patch configured; assuming neutral illuminant",
            )

        lin = bgr_u8_to_linear_rgb(frame_bgr)
        clip = clipped_mask(frame_bgr, clip_level)

        ws, wsx = _rect_to_slice(self.cfg.white_rect, frame_bgr.shape)
        wpx = lin[ws, wsx].reshape(-1, 3)
        wclip = float(clip[ws, wsx].mean()) if clip[ws, wsx].size else 1.0

        if wpx.size == 0:
            return ChipReading(self.NEUTRAL.copy(), False, note="white patch rect is empty")

        white = _trimmed_mean(wpx, self.cfg.trim_fraction)
        white_level = float(luminance(white))

        white_ok = True
        note = ""
        if wclip > self.cfg.max_clip_fraction:
            white_ok = False
            note = f"white patch clipping ({wclip:.1%}) -- lower exposure"
        elif white_level < self.cfg.min_mean_level:
            white_ok = False
            note = f"white patch too dark ({white_level:.3f}) -- raise exposure"

        # Normalise so the illuminant estimate is a pure colour, unit
        # luminance. Dividing by this leaves object radiance on its own
        # scale rather than rescaling it by the lamp's absolute power.
        est = white / max(white_level, 1e-6)

        if white_ok:
            a = float(np.clip(self.cfg.smoothing, 0.0, 0.999))
            self._smoothed = est if self._smoothed is None else a * self._smoothed + (1 - a) * est

        illum = self._smoothed if self._smoothed is not None else self.NEUTRAL.copy()

        grey_ok = False
        grey_level = 0.0
        lin_err = None
        if self.cfg.grey_rect is not None:
            gs, gsx = _rect_to_slice(self.cfg.grey_rect, frame_bgr.shape)
            gpx = lin[gs, gsx].reshape(-1, 3)
            if gpx.size:
                grey = _trimmed_mean(gpx, self.cfg.trim_fraction)
                grey_level = float(luminance(grey))
                gclip = float(clip[gs, gsx].mean())
                grey_ok = gclip <= self.cfg.max_clip_fraction and grey_level > 1e-4
                if grey_ok and white_level > 1e-4:
                    expected = self.cfg.grey_reflectance / max(self.cfg.white_reflectance, 1e-6)
                    lin_err = float(abs((grey_level / white_level) - expected) / max(expected, 1e-6))
                    if lin_err > 0.25 and not note:
                        note = (f"white:grey ratio off nominal by {lin_err:.0%} -- "
                                "camera tone curve is not sRGB; slope will be biased")

        return ChipReading(
            illuminant_rgb=np.asarray(illum, dtype=np.float32),
            calibrated=True,
            white_ok=white_ok,
            grey_ok=grey_ok,
            white_level=white_level,
            grey_level=grey_level,
            white_clip_fraction=wclip,
            linearity_error=lin_err,
            note=note,
        )
