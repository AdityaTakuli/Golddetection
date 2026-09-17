"""
Configuration for the material-discrimination stage.

Every threshold here is a *starting point*, not a calibrated value. The
defaults were chosen from the physics (see physics.colour) and from
synthetic dichromatic samples -- they have not been fitted to real gold.
Run tools/capture_dataset.py on your own pieces, then tighten them.
Treating these numbers as production-ready is the fastest way to ship a
confident wrong answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "physics.json"

# Normalised rect: (x, y, w, h) as fractions of frame width/height, so a
# calibration survives a resolution change.
Rect = Tuple[float, float, float, float]


@dataclass
class CameraConfig:
    """How to open a camera and how to nail its controls down.

    `source` is either a V4L2 index (0, 1, ...) or a stream URL
    ("rtsp://user:pass@host:554/Streaming/Channels/101"). Dome/IP cameras
    take the URL path; USB webcams take the index.
    """

    source: Union[int, str] = 0
    name: str = "primary"

    width: Optional[int] = 1280
    height: Optional[int] = 720
    fps: Optional[int] = 30

    # Rotate the incoming frame by this many degrees (0/90/180/270).
    # The existing rig mounts both USB cameras upside-down.
    rotate: int = 0

    # --- control locking -------------------------------------------------
    # Auto anything is fatal to this measurement. Auto-exposure applies a
    # global gain that moves the instant the lamp arrives -- exactly when
    # the signal appears -- and auto white balance actively cancels the
    # colour cast that IS the gold signal.
    lock_controls: bool = True
    auto_exposure: bool = False
    auto_white_balance: bool = False
    autofocus: bool = False

    exposure_absolute: Optional[int] = 150   # V4L2 units (100us steps typically)
    gain: Optional[int] = 0
    white_balance_temperature: Optional[int] = 4600
    focus_absolute: Optional[int] = 0
    brightness: Optional[int] = None
    contrast: Optional[int] = None
    saturation: Optional[int] = None

    # Explicit /dev/videoN for v4l2-ctl. Derived from `source` when it is
    # an int and this is left unset.
    v4l2_device: Optional[str] = None

    # --- stream tuning ---------------------------------------------------
    # RTSP needs a small buffer or you analyse frames that are seconds old.
    stream_buffer_size: int = 1
    stream_reconnect_delay_s: float = 2.0
    stream_max_reconnects: int = 0  # 0 = retry forever

    @property
    def is_stream(self) -> bool:
        return isinstance(self.source, str) and "://" in self.source


@dataclass
class ChipConfig:
    """Where the white and grey reference patches live in frame.

    The white patch is what makes every threshold below survive a bulb
    change, ambient drift or an IP camera whose exposure you cannot reach.
    Without it the system still runs, but chroma is measured against an
    assumed neutral illuminant and will drift.

    Use PTFE if you can. Most white plastics and papers contain optical
    brighteners that absorb UV and re-emit blue -- a reference that shifts
    colour with the lamp's UV content is worse than no reference.
    """

    white_rect: Optional[Rect] = None
    grey_rect: Optional[Rect] = None

    # Nominal reflectance of the patches, used for the linearity check.
    white_reflectance: float = 0.95
    grey_reflectance: float = 0.18

    # Reject a reading if this fraction of patch pixels is on the shoulder.
    max_clip_fraction: float = 0.02
    # Reject a reading if the patch is too dark to carry colour information.
    min_mean_level: float = 0.02
    # Fraction trimmed from each tail before averaging the patch.
    trim_fraction: float = 0.15

    # Exponential smoothing on the illuminant estimate. The lamp does not
    # change colour frame to frame; the estimate is noisy, so smooth it.
    smoothing: float = 0.85

    @property
    def configured(self) -> bool:
        return self.white_rect is not None


@dataclass
class Thresholds:
    """Decision thresholds for the material verdict. Calibrate these."""

    # --- sample validity -------------------------------------------------
    clip_level: int = 250
    # Above this fraction of clipped pixels in the analysis band the
    # measurement is void -- not "uncertain", void. Saturated pixels drive
    # chroma to zero for every material.
    max_clip_fraction: float = 0.05
    min_samples: int = 4000
    # Ratio of p99 to p50 luminance within the pooled samples.
    min_dynamic_range: float = 1.6
    # Ratio of peak to baseline frame luminance, i.e. how much brighter the
    # lamp actually made the object. This is the one that matters: pooled
    # dynamic range can come from curvature alone, and a dielectric lit
    # flatly shows CONSTANT chroma across its shading -- indistinguishable
    # from metal. Only a real specular excursion separates them.
    min_sweep_ratio: float = 1.5

    # --- the discriminator ----------------------------------------------
    # chroma(top luminance decile) / chroma(median band).
    # Metal holds its colour into the highlight -> ratio near 1.
    # Dielectric whitens -> ratio well below 1.
    metal_ratio_min: float = 0.80
    dielectric_ratio_max: float = 0.55
    # d(chroma)/d(normalised luminance) over the upper band.
    metal_slope_min: float = -0.05

    # --- warm metal vs neutral metal ------------------------------------
    # Hue angle in the chromaticity plane. Gold (R>G>B) lands near 0-20 deg;
    # a saturated yellow sits nearer 45. Wide by default -- narrow it once
    # you have measured your own stock.
    gold_hue_min: float = -15.0
    gold_hue_max: float = 60.0
    min_gold_chroma: float = 0.015

    # --- sweep state machine --------------------------------------------
    baseline_frames: int = 12
    rise_factor: float = 1.25    # frame p99 must exceed baseline * this
    fall_factor: float = 0.85    # ...then drop below peak * this
    min_sweep_frames: int = 8
    timeout_frames: int = 300
    # Random pixels sampled from the mask per frame. Bounds memory while
    # keeping the joint (luminance, chroma) distribution intact.
    pixels_per_frame: int = 2000
    max_samples: int = 400_000

    # Track association IoU-ish overlap fraction.
    match_overlap: float = 0.35
    track_ttl_frames: int = 8


@dataclass
class PhysicsConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    context_camera: Optional[CameraConfig] = None
    chips: ChipConfig = field(default_factory=ChipConfig)
    thresholds: Thresholds = field(default_factory=Thresholds)

    @classmethod
    def load(cls, path: Union[str, Path, None] = None) -> "PhysicsConfig":
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text())
        ctx = raw.get("context_camera")
        return cls(
            camera=CameraConfig(**raw.get("camera", {})),
            context_camera=CameraConfig(**ctx) if ctx else None,
            chips=ChipConfig(**raw.get("chips", {})),
            thresholds=Thresholds(**raw.get("thresholds", {})),
        )

    def save(self, path: Union[str, Path, None] = None) -> Path:
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, default=str))
        return p
