"""
Feature extraction for Model B (the material classifier).

Model B does NOT take an RGB crop. An RGB crop is exactly the input that
cannot separate gold from yellow plastic -- that information is not in the
pixels under uncontrolled light, so no architecture recovers it. What it
takes is the *shape of the chroma-vs-luminance distribution* measured over
a controlled brightness excursion, which is where the discriminating
physics actually lives.

Start with gradient-boosted trees over these features, not a CNN:

  * they win at the sample counts you will realistically collect
    (thousands of physical objects, not millions of images);
  * they run on a Pi with no GPU;
  * they tell you *why* they decided, which a pawn-counter audit trail
    needs and a CNN will not give you.

Move to a CNN over the raw multi-illumination stack only past ~20k
samples. Until then this is both more accurate and more defensible.

Train with:
    import lightgbm, pandas as pd
    df = pd.read_csv("dataset/features.csv")
    X, y = df[FEATURE_NAMES], df["label"]
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .colour import NEUTRAL
from .specular import MaterialVerdict, SweepAccumulator, _theil_sen

_DECILES = list(range(10, 101, 10))

FEATURE_NAMES: List[str] = (
    [f"chroma_p{p}" for p in _DECILES]
    + [f"lum_p{p}" for p in (5, 25, 50, 75, 90, 99)]
    + [
        "chroma_slope", "chroma_ratio", "chroma_top", "chroma_mid", "chroma_low",
        "chroma_std", "chroma_iqr",
        "r_slope", "g_slope", "r_top", "g_top", "r_mid", "g_mid",
        "hue_top", "hue_mid", "hue_spread",
        "dynamic_range", "sweep_ratio", "peak_clip_fraction",
        "lum_skew", "lum_kurtosis", "highlight_fraction",
        "n_samples", "n_frames", "calibrated",
    ]
)


def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else 0.0


def _binned_slope(yn: np.ndarray, values: np.ndarray, lo: float = 0.40) -> float:
    """Robust slope of `values` against normalised luminance above `lo`."""
    band = yn >= lo
    if band.sum() < 60:
        return 0.0
    edges = np.quantile(yn[band], np.linspace(0.0, 1.0, 11))
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = band & (yn >= a) & (yn <= b)
        if m.sum() >= 25:
            xs.append(float(np.median(yn[m])))
            ys.append(float(np.median(values[m])))
    return _safe(_theil_sen(np.asarray(xs), np.asarray(ys))) if len(xs) >= 3 else 0.0


def extract_features(
    accumulator: SweepAccumulator,
    verdict: Optional[MaterialVerdict] = None,
) -> Dict[str, float]:
    """Physics feature vector for one captured sweep.

    Returns zeros for every feature when the sweep carries too little data
    to characterise -- a row of zeros is honest, an imputed guess is not.
    """
    feats: Dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}

    lum, chroma, rg = accumulator.raw_samples()
    if len(lum) < 200:
        return feats

    y99 = float(np.percentile(lum, 99.0))
    if y99 <= 1e-6:
        return feats
    yn = lum / y99

    for p in _DECILES:
        lo, hi = (p - 10) / 100.0, p / 100.0
        m = (yn >= lo) & (yn <= hi)
        feats[f"chroma_p{p}"] = _safe(np.median(chroma[m])) if m.sum() >= 20 else 0.0

    for p in (5, 25, 50, 75, 90, 99):
        feats[f"lum_p{p}"] = _safe(np.percentile(lum, p))

    top = yn >= 0.90
    mid = (yn >= 0.40) & (yn <= 0.60)
    low = yn <= 0.25

    if top.sum() >= 20:
        feats["chroma_top"] = _safe(np.median(chroma[top]))
        d = rg[top] - NEUTRAL
        feats["r_top"] = _safe(np.median(rg[top][:, 0]))
        feats["g_top"] = _safe(np.median(rg[top][:, 1]))
        feats["hue_top"] = _safe(np.degrees(np.arctan2(d[:, 1].mean(), d[:, 0].mean())))
        # Angular spread of the highlight's hue. A metal's highlight holds
        # one colour; a whitening dielectric's hue wanders as chroma -> 0.
        ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
        feats["hue_spread"] = _safe(np.percentile(ang, 84) - np.percentile(ang, 16))
    if mid.sum() >= 20:
        feats["chroma_mid"] = _safe(np.median(chroma[mid]))
        dm = rg[mid] - NEUTRAL
        feats["r_mid"] = _safe(np.median(rg[mid][:, 0]))
        feats["g_mid"] = _safe(np.median(rg[mid][:, 1]))
        feats["hue_mid"] = _safe(np.degrees(np.arctan2(dm[:, 1].mean(), dm[:, 0].mean())))
    if low.sum() >= 20:
        feats["chroma_low"] = _safe(np.median(chroma[low]))

    feats["chroma_ratio"] = _safe(feats["chroma_top"] / max(feats["chroma_mid"], 1e-6))
    feats["chroma_std"] = _safe(np.std(chroma))
    feats["chroma_iqr"] = _safe(np.percentile(chroma, 75) - np.percentile(chroma, 25))
    feats["chroma_slope"] = _binned_slope(yn, chroma)
    feats["r_slope"] = _binned_slope(yn, rg[:, 0])
    feats["g_slope"] = _binned_slope(yn, rg[:, 1])

    y50 = float(np.percentile(lum, 50.0))
    feats["dynamic_range"] = _safe(y99 / max(y50, 1e-6))
    feats["highlight_fraction"] = _safe(float((yn >= 0.9).mean()))

    mu, sd = float(lum.mean()), float(lum.std())
    if sd > 1e-9:
        z = (lum - mu) / sd
        feats["lum_skew"] = _safe(np.mean(z ** 3))
        feats["lum_kurtosis"] = _safe(np.mean(z ** 4) - 3.0)

    feats["peak_clip_fraction"] = _safe(accumulator.peak_clip_fraction)
    feats["n_samples"] = float(len(lum))
    feats["n_frames"] = float(accumulator.n_frames)
    feats["calibrated"] = 1.0 if accumulator.calibrated else 0.0
    feats["sweep_ratio"] = _safe(verdict.sweep_ratio) if verdict else 0.0

    return feats


def feature_row(feats: Dict[str, float]) -> List[float]:
    """Ordered values matching FEATURE_NAMES."""
    return [feats.get(name, 0.0) for name in FEATURE_NAMES]
