#!/usr/bin/env python3
"""
Calibrate the reference patches and the exposure.

Two jobs, both one-time per rig:

  1. Tell the system where the white (and optionally grey) reference patch
     sits in frame. Every chroma threshold then becomes a ratio against the
     light's own colour, which is what makes the material verdict survive a
     bulb change, ambient drift, or a dome camera whose white balance you
     cannot reach over RTSP.

  2. Find an exposure at which the object's highlights do not saturate.
     This matters more than it sounds: once a channel clips, chroma goes to
     zero for gold and plastic alike, so the discriminator is measuring
     nothing. A dark frame is recoverable; a blown one is not.

Patch material: use PTFE if you can get it. Most white plastics, papers and
paints contain optical brighteners that absorb UV and re-emit blue, so they
shift colour with the lamp's UV content -- a reference that moves is worse
than no reference.

Usage
-----
    python3 tools/calibrate_chips.py                      # interactive
    python3 tools/calibrate_chips.py --camera rtsp://...  # dome camera
    python3 tools/calibrate_chips.py --white 0.02,0.70,0.08,0.12 --no-gui
    python3 tools/calibrate_chips.py --find-exposure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physics.camera import CameraSource, find_exposure_without_clipping  # noqa: E402
from physics.colour import clipped_mask  # noqa: E402
from physics.config import PhysicsConfig  # noqa: E402
from physics.reference import ReferenceChips  # noqa: E402


def _parse_rect(text):
    parts = [float(p) for p in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("rect must be x,y,w,h as fractions of the frame")
    return tuple(parts)


def _select_rect(window, frame, prompt):
    print(f"\n{prompt}\n  drag a box, then ENTER/SPACE to accept, C to skip")
    disp = frame.copy()
    cv2.putText(disp, prompt[:60], (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)
    x, y, w, h = cv2.selectROI(window, disp, showCrosshair=True, fromCenter=False)
    if w == 0 or h == 0:
        return None
    H, W = frame.shape[:2]
    return (x / W, y / H, w / W, h / H)


def _grab(source, warmup=10):
    frame = None
    for _ in range(warmup):
        ok, f = source.read()
        if ok:
            frame = f
    if frame is None:
        raise RuntimeError("could not read a frame from the camera")
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--camera", default=None, help="USB index or stream URL")
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=None)
    ap.add_argument("--white", type=_parse_rect, default=None,
                    help="white patch rect as x,y,w,h fractions (skips the GUI)")
    ap.add_argument("--grey", type=_parse_rect, default=None)
    ap.add_argument("--no-gui", action="store_true", help="for SSH / headless rigs")
    ap.add_argument("--find-exposure", action="store_true",
                    help="search downward for an exposure with no clipping")
    ap.add_argument("--exposure-candidates", default="600,500,400,300,250,200,150,120,100,80,60,40,20",
                    help="descending V4L2 exposure values to try")
    args = ap.parse_args()

    cfg = PhysicsConfig.load(args.config)
    if args.camera is not None:
        cfg.camera.source = int(args.camera) if args.camera.isdigit() else args.camera
    if args.rotate is not None:
        cfg.camera.rotate = args.rotate

    source = CameraSource(cfg.camera).open(lock=True)
    if source.lock_report:
        print(f"\n[camera] {source.lock_report.summary()}")
        for w in source.lock_report.warnings:
            print(f"[camera] WARNING: {w}")

    try:
        frame = _grab(source)
        print(f"[camera] frame {frame.shape[1]}x{frame.shape[0]}")

        if args.find_exposure:
            if cfg.camera.is_stream:
                print("\n[exposure] Cannot set exposure on a stream. Lock it in the "
                      "camera's own web UI, aiming for no saturated pixels on the "
                      "brightest part of the piece under your lamp.")
            else:
                cands = [int(c) for c in args.exposure_candidates.split(",")]
                chosen = find_exposure_without_clipping(
                    lambda: (source.read() or (False, None))[1],
                    source.control.set_exposure,
                    cands,
                )
                if chosen is None:
                    print("\n[exposure] No candidate avoided clipping. Dim the lamp, "
                          "move it further away, or add a diffuser.")
                else:
                    cfg.camera.exposure_absolute = chosen
                    print(f"\n[exposure] Using {chosen}")
                frame = _grab(source)

        # -- patch rects --
        white = args.white if args.white is not None else cfg.chips.white_rect
        grey = args.grey if args.grey is not None else cfg.chips.grey_rect

        if not args.no_gui and args.white is None:
            win = "calibrate -- reference patches"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            white = _select_rect(win, frame, "Select the WHITE reference patch") or white
            grey = _select_rect(win, frame, "Select the GREY patch (optional -- C to skip)") or grey
            cv2.destroyWindow(win)

        cfg.chips.white_rect = white
        cfg.chips.grey_rect = grey

        if white is None:
            print("\nNo white patch set. The system will still run, but chroma is "
                  "measured against an assumed neutral illuminant and thresholds "
                  "will drift as the lighting changes.")
        else:
            reading = ReferenceChips(cfg.chips).measure(frame, cfg.thresholds.clip_level)
            e = reading.illuminant_rgb
            print(f"\n[chips] illuminant RGB = [{e[0]:.3f} {e[1]:.3f} {e[2]:.3f}]")
            print(f"[chips] white level {reading.white_level:.3f}  "
                  f"clipped {reading.white_clip_fraction:.1%}  ok={reading.white_ok}")
            if reading.linearity_error is not None:
                print(f"[chips] white:grey linearity error {reading.linearity_error:.0%}")
            if reading.note:
                print(f"[chips] NOTE: {reading.note}")

        frac = float(clipped_mask(frame, cfg.thresholds.clip_level).mean())
        print(f"[frame] {frac:.2%} of pixels saturated"
              + ("  <-- too high, lower the exposure" if frac > 0.02 else ""))

        path = cfg.save(args.config)
        print(f"\nSaved {path}")
    finally:
        source.release()


if __name__ == "__main__":
    main()
