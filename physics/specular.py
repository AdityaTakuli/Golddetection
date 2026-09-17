"""
The material discriminator.

Replaces the HSV yellow gate, which could never have worked: a yellow gate
cannot reject yellow plastic, because yellow plastic is yellow. The axis
that separates them is not hue but *how chromaticity behaves as luminance
rises* -- see the dichromatic model in physics.colour.

Method
------
Across a brightness sweep (a lamp or phone flash moved over the piece) we
accumulate per-pixel (luminance, chroma) pairs from inside the object mask,
with the illuminant divided out and saturated pixels discarded. Then:

  ratio = chroma(top luminance decile) / chroma(mid luminance band)
  slope = d(chroma) / d(normalised luminance), robustly fitted

  metal      -> chroma is invariant to brightness -> ratio ~ 1, slope ~ 0
  dielectric -> chroma collapses toward the illuminant -> ratio << 1

Both statistics are *shapes* of a distribution, not absolute levels, so
they survive exposure changes, bulb colour temperature and sensor drift in
a way a fixed Delta-b* threshold never could.

Hue is used only afterwards, to split a warm metal (gold, brass, bronze)
from a neutral one (silver, steel). It is never used to make the
metal/dielectric call.

Limits, stated plainly: this separates metal from non-metal and warm metal
from neutral metal. It does NOT separate gold from brass, gold from
gold-plated base metal, or resolve karat. Those need XRF or a density
measurement. A GOLD_LIKE verdict means "behaves optically like a warm
metal", nothing stronger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np

from .colour import (
    NEUTRAL,
    chroma_from_rg,
    chromaticity,
    luminance,
    von_kries,
)
from .config import Thresholds


class Verdict(Enum):
    PENDING = "PENDING"                    # not enough data yet
    SWEEPING = "SWEEPING"                  # sweep in progress
    GOLD_LIKE = "GOLD_LIKE"                # warm metal
    NON_GOLD_METAL = "NON_GOLD_METAL"      # metal, but neutral/cool (silver, steel)
    DIELECTRIC = "DIELECTRIC"              # plastic, paint, fabric, resin
    UNCERTAIN = "UNCERTAIN"                # statistics inconclusive
    INVALID_CLIPPED = "INVALID_CLIPPED"    # highlights blown; measurement void
    INVALID_NO_SWEEP = "INVALID_NO_SWEEP"  # never got a brightness excursion


# Verdicts that should never be reported as a gold confirmation.
NON_CONFIRMING = {
    Verdict.PENDING, Verdict.SWEEPING, Verdict.DIELECTRIC, Verdict.UNCERTAIN,
    Verdict.INVALID_CLIPPED, Verdict.INVALID_NO_SWEEP, Verdict.NON_GOLD_METAL,
}


@dataclass
class MaterialVerdict:
    state: Verdict = Verdict.PENDING
    chroma_ratio: float = 0.0
    slope: float = 0.0
    hue_deg: float = 0.0
    chroma_top: float = 0.0
    chroma_mid: float = 0.0
    dynamic_range: float = 0.0
    peak_clip_fraction: float = 0.0
    n_samples: int = 0
    n_frames: int = 0
    calibrated: bool = False
    reason: str = ""

    @property
    def is_gold(self) -> bool:
        return self.state is Verdict.GOLD_LIKE

    @property
    def is_final(self) -> bool:
        return self.state not in (Verdict.PENDING, Verdict.SWEEPING)

    def as_dict(self) -> dict:
        return {
            "state": self.state.value,
            "chroma_ratio": round(self.chroma_ratio, 5),
            "slope": round(self.slope, 5),
            "hue_deg": round(self.hue_deg, 2),
            "chroma_top": round(self.chroma_top, 5),
            "chroma_mid": round(self.chroma_mid, 5),
            "dynamic_range": round(self.dynamic_range, 3),
            "peak_clip_fraction": round(self.peak_clip_fraction, 4),
            "n_samples": self.n_samples,
            "n_frames": self.n_frames,
            "calibrated": self.calibrated,
            "reason": self.reason,
        }


def _theil_sen(x: np.ndarray, y: np.ndarray) -> float:
    """Median of pairwise slopes. Robust to a couple of bad bins."""
    n = len(x)
    if n < 2:
        return 0.0
    i, j = np.triu_indices(n, k=1)
    dx = x[j] - x[i]
    ok = np.abs(dx) > 1e-9
    if not ok.any():
        return 0.0
    return float(np.median((y[j] - y[i])[ok] / dx[ok]))


class SweepAccumulator:
    """Collects (luminance, chroma, hue-vector) samples over a sweep.

    Pixels are randomly subsampled per frame rather than kept whole: that
    bounds memory while leaving the joint (luminance, chroma) distribution
    -- which is all the statistics below actually need -- intact.
    """

    def __init__(self, thresholds: Thresholds, seed: int = 0):
        self.t = thresholds
        self._rng = np.random.default_rng(seed)
        self._lum: List[np.ndarray] = []
        self._chroma: List[np.ndarray] = []
        self._rg: List[np.ndarray] = []
        self.n_frames = 0
        self.peak_clip_fraction = 0.0
        self.calibrated = False
        self._total = 0

    def add_frame(
        self,
        rgb_lin: np.ndarray,
        mask: np.ndarray,
        clip: np.ndarray,
        illuminant_rgb: np.ndarray,
        calibrated: bool,
    ) -> float:
        """Add one frame's masked pixels. Returns this frame's p99 luminance.

        That return value is what drives the sweep state machine -- it is
        how we notice the lamp arriving and leaving.
        """
        self.calibrated = self.calibrated or calibrated
        sel = mask & ~clip
        n_mask = int(mask.sum())
        if n_mask:
            self.peak_clip_fraction = max(
                self.peak_clip_fraction, float((mask & clip).sum()) / n_mask
            )
        n = int(sel.sum())
        if n < 32:
            return 0.0

        px = von_kries(rgb_lin[sel], illuminant_rgb)
        if n > self.t.pixels_per_frame:
            idx = self._rng.choice(n, self.t.pixels_per_frame, replace=False)
            px = px[idx]

        lum = luminance(px)
        rg = chromaticity(px)

        if self._total < self.t.max_samples:
            self._lum.append(lum)
            self._chroma.append(chroma_from_rg(rg))
            self._rg.append(rg)
            self._total += len(lum)

        self.n_frames += 1
        return float(np.percentile(lum, 99)) if len(lum) else 0.0

    # -- statistics -------------------------------------------------------
    def decide(self) -> MaterialVerdict:
        t = self.t
        v = MaterialVerdict(n_frames=self.n_frames, calibrated=self.calibrated,
                            peak_clip_fraction=self.peak_clip_fraction)

        if not self._lum:
            v.state = Verdict.INVALID_NO_SWEEP
            v.reason = "no usable pixels collected"
            return v

        lum = np.concatenate(self._lum)
        chroma = np.concatenate(self._chroma)
        rg = np.concatenate(self._rg)
        v.n_samples = int(len(lum))

        if v.n_samples < t.min_samples:
            v.state = Verdict.INVALID_NO_SWEEP
            v.reason = f"only {v.n_samples} samples (need {t.min_samples})"
            return v

        # Saturated highlights destroy chroma for every material alike, so
        # a blown measurement is void rather than merely uncertain.
        if self.peak_clip_fraction > t.max_clip_fraction:
            v.state = Verdict.INVALID_CLIPPED
            v.reason = (f"{self.peak_clip_fraction:.1%} of the object clipped at peak "
                        f"(limit {t.max_clip_fraction:.1%}) -- lower exposure")
            return v

        y99 = float(np.percentile(lum, 99.0))
        y50 = float(np.percentile(lum, 50.0))
        if y99 <= 1e-6:
            v.state = Verdict.INVALID_NO_SWEEP
            v.reason = "object is black"
            return v

        v.dynamic_range = y99 / max(y50, 1e-6)
        if v.dynamic_range < t.min_dynamic_range:
            v.state = Verdict.INVALID_NO_SWEEP
            v.reason = (f"luminance range {v.dynamic_range:.2f}x below "
                        f"{t.min_dynamic_range}x -- sweep the lamp across the piece")
            return v

        yn = lum / y99

        top = yn >= 0.90
        mid = (yn >= 0.40) & (yn <= 0.60)
        if top.sum() < 50 or mid.sum() < 50:
            v.state = Verdict.UNCERTAIN
            v.reason = "too few samples in the top or mid luminance band"
            return v

        v.chroma_top = float(np.median(chroma[top]))
        v.chroma_mid = float(np.median(chroma[mid]))

        # Circular mean of the top decile's hue -- the metal's own colour.
        d = rg[top] - NEUTRAL
        v.hue_deg = float(np.degrees(np.arctan2(d[:, 1].mean(), d[:, 0].mean())))

        # Robust slope over the upper half, binned by luminance quantile.
        band = yn >= 0.40
        edges = np.quantile(yn[band], np.linspace(0.0, 1.0, 11))
        xs, ys = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = band & (yn >= a) & (yn <= b)
            if m.sum() >= 30:
                xs.append(float(np.median(yn[m])))
                ys.append(float(np.median(chroma[m])))
        v.slope = _theil_sen(np.asarray(xs), np.asarray(ys)) if len(xs) >= 3 else 0.0

        # An object with no colour at any brightness carries no information
        # for this test -- the ratio would be noise over noise.
        if v.chroma_mid < t.min_gold_chroma and v.chroma_top < t.min_gold_chroma:
            v.state = Verdict.NON_GOLD_METAL if v.slope >= t.metal_slope_min else Verdict.DIELECTRIC
            v.reason = f"achromatic object (chroma {v.chroma_mid:.4f}) -- not gold either way"
            return v

        v.chroma_ratio = v.chroma_top / max(v.chroma_mid, 1e-6)

        if v.chroma_ratio >= t.metal_ratio_min and v.slope >= t.metal_slope_min:
            warm = t.gold_hue_min <= v.hue_deg <= t.gold_hue_max
            if warm and v.chroma_top >= t.min_gold_chroma:
                v.state = Verdict.GOLD_LIKE
                v.reason = (f"chroma held into highlight (ratio {v.chroma_ratio:.2f}, "
                            f"slope {v.slope:+.3f}), warm hue {v.hue_deg:.0f}deg")
            else:
                v.state = Verdict.NON_GOLD_METAL
                v.reason = (f"metal-like (ratio {v.chroma_ratio:.2f}) but hue "
                            f"{v.hue_deg:.0f}deg outside the gold window")
        elif v.chroma_ratio <= t.dielectric_ratio_max:
            v.state = Verdict.DIELECTRIC
            v.reason = (f"chroma collapsed toward the illuminant in the highlight "
                        f"(ratio {v.chroma_ratio:.2f}) -- dielectric, not metal")
        else:
            v.state = Verdict.UNCERTAIN
            v.reason = (f"ratio {v.chroma_ratio:.2f} sits between the dielectric "
                        f"({t.dielectric_ratio_max}) and metal ({t.metal_ratio_min}) bands")
        return v

    def raw_samples(self):
        """(luminance, chroma, rg) arrays -- for feature extraction / datasets."""
        if not self._lum:
            return (np.empty(0, np.float32),) * 2 + (np.empty((0, 2), np.float32),)
        return (np.concatenate(self._lum), np.concatenate(self._chroma),
                np.concatenate(self._rg))
