"""
Camera sourcing and control locking.

Two jobs, kept separate on purpose:

  CameraSource   -- open a frame source, whatever it is: a USB index, an
                    RTSP/HTTP stream from a dome camera over Ethernet, or
                    a caller-supplied opener for hardware with its own SDK.
  CameraControl  -- nail down exposure, gain, white balance and focus.

Why the locking matters more than it looks: auto-exposure is a *global*
gain. When the lamp arrives it pulls that gain down, compressing the very
brightness excursion the discriminator measures -- and it does so at
exactly the moment the signal appears. Reading only masked pixels does not
escape it, because the gain multiplies those pixels too. Auto white
balance is worse: it exists to cancel colour casts, and a gold highlight's
warm tint IS a colour cast. Left on, it erases the signal by design.

Dome/IP cameras over RTSP expose no V4L2 controls, so nothing here can set
them. Lock those in the camera's own web UI or via ONVIF, once. This module
detects the symptoms (clipping, drifting illuminant) and says so rather
than pretending it has control it does not have.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .config import CameraConfig

log = logging.getLogger("physics.camera")

# V4L2 control names moved around between kernel versions. Try each alias
# in order and use whichever the driver actually exposes.
_CTRL_ALIASES = {
    "auto_exposure":  ("auto_exposure", "exposure_auto"),
    "exposure_time":  ("exposure_time_absolute", "exposure_absolute"),
    "auto_wb":        ("white_balance_automatic", "white_balance_temperature_auto"),
    "wb_temperature": ("white_balance_temperature",),
    "auto_focus":     ("focus_automatic_continuous", "focus_auto"),
    "focus":          ("focus_absolute",),
    "gain":           ("gain",),
    "brightness":     ("brightness",),
    "contrast":       ("contrast",),
    "saturation":     ("saturation",),
}

# For both auto_exposure spellings: 1 = manual, 3 = aperture priority (auto).
_EXPOSURE_MANUAL = 1
_EXPOSURE_AUTO = 3


@dataclass
class LockReport:
    """What we asked for, what actually took, and what could not be set."""

    applied: dict = field(default_factory=dict)
    verified: dict = field(default_factory=dict)
    failed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    method: str = "none"

    @property
    def ok(self) -> bool:
        return not self.failed and self.method != "none"

    def summary(self) -> str:
        if self.method == "none":
            return "camera controls NOT locked -- " + ("; ".join(self.warnings) or "no method available")
        bits = [f"{k}={v}" for k, v in sorted(self.verified.items())]
        s = f"locked via {self.method}: " + (", ".join(bits) if bits else "(nothing verified)")
        if self.failed:
            s += f" | could not set: {', '.join(self.failed)}"
        return s


class CameraControl:
    """Locks V4L2 camera controls. No-op (with warnings) for streams."""

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.device = self._resolve_device()
        self._available: Optional[List[str]] = None

    def _resolve_device(self) -> Optional[str]:
        if self.cfg.is_stream:
            return None
        if self.cfg.v4l2_device:
            return self.cfg.v4l2_device
        if isinstance(self.cfg.source, int):
            return f"/dev/video{self.cfg.source}"
        return None

    @staticmethod
    def _have_v4l2() -> bool:
        return shutil.which("v4l2-ctl") is not None

    def _list_controls(self) -> List[str]:
        if self._available is not None:
            return self._available
        self._available = []
        if not (self.device and self._have_v4l2()):
            return self._available
        try:
            out = subprocess.run(
                ["v4l2-ctl", "-d", self.device, "--list-ctrls"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            self._available = re.findall(r"^\s*(\w+)\s+0x[0-9a-f]+", out, re.MULTILINE)
        except (subprocess.SubprocessError, OSError) as exc:
            log.warning("v4l2-ctl --list-ctrls failed on %s: %s", self.device, exc)
        return self._available

    def _pick(self, key: str) -> Optional[str]:
        avail = self._list_controls()
        for name in _CTRL_ALIASES.get(key, ()):
            if name in avail:
                return name
        return None

    def _set(self, name: str, value) -> bool:
        try:
            r = subprocess.run(
                ["v4l2-ctl", "-d", self.device, f"--set-ctrl={name}={value}"],
                capture_output=True, text=True, timeout=5,
            )
            return r.returncode == 0
        except (subprocess.SubprocessError, OSError):
            return False

    def _get(self, name: str) -> Optional[int]:
        try:
            r = subprocess.run(
                ["v4l2-ctl", "-d", self.device, f"--get-ctrl={name}"],
                capture_output=True, text=True, timeout=5,
            )
            m = re.search(r":\s*(-?\d+)", r.stdout)
            return int(m.group(1)) if m else None
        except (subprocess.SubprocessError, OSError):
            return None

    def lock(self, cap: Optional[cv2.VideoCapture] = None) -> LockReport:
        """Disable every automatic and pin the manual values.

        Order matters: the autos go off first, because a driver will
        silently ignore (or immediately overwrite) a manual value set while
        its automatic counterpart is still running.
        """
        rep = LockReport()
        cfg = self.cfg

        if not cfg.lock_controls:
            rep.warnings.append("lock_controls disabled in config")
            return rep

        if cfg.is_stream:
            rep.warnings.append(
                "stream source: exposure/gain/white balance cannot be set over RTSP. "
                "Lock them in the camera's web UI or via ONVIF, once -- otherwise the "
                "material verdict will drift and highlights may clip unrecoverably."
            )
            return rep

        if self.device and self._have_v4l2() and self._list_controls():
            rep.method = "v4l2-ctl"
            self._lock_v4l2(rep)
        elif cap is not None:
            rep.method = "cv2"
            self._lock_cv2(cap, rep)
            rep.warnings.append(
                "v4l2-ctl unavailable; used OpenCV properties, which many UVC "
                "drivers accept and then ignore. Install v4l2-utils for a real lock."
            )
        else:
            rep.warnings.append("no v4l2-ctl and no capture handle -- nothing locked")
        return rep

    def _lock_v4l2(self, rep: LockReport) -> None:
        cfg = self.cfg

        # 1. autos off, before anything manual.
        for key, off_value in (
            ("auto_exposure", _EXPOSURE_AUTO if cfg.auto_exposure else _EXPOSURE_MANUAL),
            ("auto_wb", int(bool(cfg.auto_white_balance))),
            ("auto_focus", int(bool(cfg.autofocus))),
        ):
            name = self._pick(key)
            if name is None:
                continue
            if self._set(name, off_value):
                rep.applied[name] = off_value
            else:
                rep.failed.append(name)

        # 2. manual values.
        for key, value in (
            ("exposure_time", cfg.exposure_absolute),
            ("gain", cfg.gain),
            ("wb_temperature", cfg.white_balance_temperature),
            ("focus", cfg.focus_absolute),
            ("brightness", cfg.brightness),
            ("contrast", cfg.contrast),
            ("saturation", cfg.saturation),
        ):
            if value is None:
                continue
            name = self._pick(key)
            if name is None:
                rep.warnings.append(f"driver has no control for {key}")
                continue
            if self._set(name, value):
                rep.applied[name] = value
            else:
                rep.failed.append(name)

        # 3. read back -- a control that accepted a write can still clamp it.
        for name, want in rep.applied.items():
            got = self._get(name)
            if got is not None:
                rep.verified[name] = got
                if got != want:
                    rep.warnings.append(f"{name} clamped to {got} (asked {want})")

    def _lock_cv2(self, cap: cv2.VideoCapture, rep: LockReport) -> None:
        cfg = self.cfg
        # 0.25/0.75 is the widely-used UVC convention for manual/auto here.
        pairs = [
            (cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if cfg.auto_exposure else 0.25),
            (cv2.CAP_PROP_AUTO_WB, 1 if cfg.auto_white_balance else 0),
            (cv2.CAP_PROP_AUTOFOCUS, 1 if cfg.autofocus else 0),
        ]
        if cfg.exposure_absolute is not None:
            pairs.append((cv2.CAP_PROP_EXPOSURE, cfg.exposure_absolute))
        if cfg.gain is not None:
            pairs.append((cv2.CAP_PROP_GAIN, cfg.gain))
        if cfg.white_balance_temperature is not None:
            pairs.append((cv2.CAP_PROP_WB_TEMPERATURE, cfg.white_balance_temperature))
        if cfg.focus_absolute is not None:
            pairs.append((cv2.CAP_PROP_FOCUS, cfg.focus_absolute))
        for prop, value in pairs:
            if cap.set(prop, float(value)):
                rep.applied[str(prop)] = value
                rep.verified[str(prop)] = cap.get(prop)
            else:
                rep.failed.append(str(prop))

    def set_exposure(self, value: int) -> bool:
        """Set exposure only. Used by the clipping search."""
        if self.cfg.is_stream or not self.device or not self._have_v4l2():
            return False
        name = self._pick("exposure_time")
        return bool(name) and self._set(name, value)


_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class CameraSource:
    """A frame source: USB index, RTSP/HTTP stream, or your own opener.

    `opener` exists so hardware with its own connection logic -- a dome
    camera SDK, an authenticated stream builder -- drops straight in
    without this module needing to know about it:

        CameraSource(cfg, opener=lambda c: my_dome_connect(c.source))
    """

    def __init__(self, cfg: CameraConfig, opener: Optional[Callable[[CameraConfig], cv2.VideoCapture]] = None):
        self.cfg = cfg
        self._opener = opener
        self.cap: Optional[cv2.VideoCapture] = None
        self.control = CameraControl(cfg)
        self.lock_report: Optional[LockReport] = None
        self._reconnects = 0
        self._consecutive_failures = 0

    # -- lifecycle --------------------------------------------------------
    def open(self, lock: bool = True) -> "CameraSource":
        self.cap = self._open_capture()
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera source {self.cfg.source!r}")
        if not self.cfg.is_stream:
            if self.cfg.width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
            if self.cfg.height:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            if self.cfg.fps:
                self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        if lock:
            self.lock_report = self.control.lock(self.cap)
            log.info("[%s] %s", self.cfg.name, self.lock_report.summary())
            for w in self.lock_report.warnings:
                log.warning("[%s] %s", self.cfg.name, w)
        return self

    def _open_capture(self) -> cv2.VideoCapture:
        if self._opener is not None:
            return self._opener(self.cfg)
        if self.cfg.is_stream:
            cap = cv2.VideoCapture(self.cfg.source, cv2.CAP_FFMPEG)
            # Without this the decoder queues frames and you analyse a scene
            # that is seconds stale -- fatal when a human is sweeping a lamp.
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, self.cfg.stream_buffer_size)
            except cv2.error:
                pass
            return cap
        return cv2.VideoCapture(self.cfg.source)

    def _reconnect(self) -> bool:
        limit = self.cfg.stream_max_reconnects
        if limit and self._reconnects >= limit:
            return False
        self._reconnects += 1
        log.warning("[%s] stream lost; reconnect #%d in %.1fs",
                    self.cfg.name, self._reconnects, self.cfg.stream_reconnect_delay_s)
        try:
            if self.cap:
                self.cap.release()
        except cv2.error:
            pass
        time.sleep(self.cfg.stream_reconnect_delay_s)
        try:
            self.cap = self._open_capture()
            return bool(self.cap and self.cap.isOpened())
        except cv2.error:
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Grab one frame, applying configured rotation.

        Streams drop out; a dropped stream reconnects rather than killing
        the run. USB read failures are passed straight through.
        """
        if self.cap is None:
            return False, None
        ok, frame = self.cap.read()
        if ok and frame is not None:
            self._consecutive_failures = 0
            rot = _ROTATIONS.get(self.cfg.rotate % 360)
            if rot is not None:
                frame = cv2.rotate(frame, rot)
            return True, frame

        self._consecutive_failures += 1
        if self.cfg.is_stream and self._consecutive_failures >= 5:
            self._consecutive_failures = 0
            if self._reconnect():
                return False, None
        return False, None

    def release(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except cv2.error:
                pass
            self.cap = None

    def __enter__(self) -> "CameraSource":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()


def find_exposure_without_clipping(
    read_frame: Callable[[], Optional[np.ndarray]],
    set_exposure: Callable[[int], bool],
    candidates: List[int],
    clip_level: int = 250,
    max_clip_fraction: float = 0.002,
    settle_frames: int = 5,
) -> Optional[int]:
    """Walk exposure down until the frame's highlights stop clipping.

    This is the "stop clipping highlights" step, and it is not cosmetic:
    once a channel saturates, chroma goes to zero for gold and plastic
    alike, so the discriminator is measuring nothing. Better a dark frame
    than a blown one -- darkness is recoverable, saturation is not.

    `candidates` should be descending. Returns the first exposure whose
    clipped fraction is under the limit, else None.
    """
    from .colour import clipped_mask

    for exposure in candidates:
        if not set_exposure(exposure):
            return None
        for _ in range(settle_frames):   # drivers apply changes lazily
            read_frame()
        frame = read_frame()
        if frame is None:
            continue
        frac = float(clipped_mask(frame, clip_level).mean())
        log.info("exposure %d -> %.3f%% clipped", exposure, frac * 100)
        if frac <= max_clip_fraction:
            return exposure
    return None
