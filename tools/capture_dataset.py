#!/usr/bin/env python3
"""
Collect labelled sweeps for training Model B (the material classifier).

Workflow per sample
-------------------
  1. Put one piece on the platform.
  2. Press its label key (see below). The capture arms and takes a baseline.
  3. Sweep your lamp or phone torch across the piece, once, steadily.
  4. The sample is written when the sweep completes.

Keys:  g gold   p plastic   b brass   s silver   o other
       n  next pose of the SAME physical object
       u  undo the last sample
       r  reset the current capture      q  quit

IMPORTANT -- object_id and data leakage
---------------------------------------
Every sample records an `object_id` identifying the *physical piece*. When
you train, split on that, not on rows:

    from sklearn.model_selection import GroupKFold
    GroupKFold().split(X, y, groups=df.object_id)

Ten poses of one ring are ten views of one object, not ten independent
samples. Split them randomly and the same ring lands in train and test, and
your accuracy will look excellent and mean nothing. Aim for 200-500
distinct physical objects, 10-20 poses each, and make sure the plastic
class contains the yellow pieces that actually fool your current system --
not easy negatives.

Output
------
    dataset/features.csv           one row per sample, ready for LightGBM
    dataset/<label>/<sample_id>/   frames, mask, metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physics.camera import CameraSource  # noqa: E402
from physics.config import PhysicsConfig  # noqa: E402
from physics.features import FEATURE_NAMES, extract_features, feature_row  # noqa: E402
from physics.pipeline import MaterialStage  # noqa: E402
from physics.specular import Verdict  # noqa: E402
from physics.tracker import Detection  # noqa: E402

LABEL_KEYS = {
    ord("g"): "gold",
    ord("p"): "plastic",
    ord("b"): "brass",
    ord("s"): "silver",
    ord("o"): "other",
}

META_COLUMNS = [
    "sample_id", "object_id", "label", "captured_at", "pose",
    "verdict", "verdict_reason", "ornament_type", "detector_confidence",
    "calibrated", "camera_source", "notes",
]


class DatasetWriter:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.root / "features.csv"
        self._written = []
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as fh:
                csv.writer(fh).writerow(META_COLUMNS + FEATURE_NAMES)

    def write(self, meta: dict, feats: dict, frames, mask) -> Path:
        sample_dir = self.root / meta["label"] / meta["sample_id"]
        sample_dir.mkdir(parents=True, exist_ok=True)

        for i, f in enumerate(frames):
            cv2.imwrite(str(sample_dir / f"frame_{i:03d}.jpg"), f,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if mask is not None:
            cv2.imwrite(str(sample_dir / "mask.png"), (mask.astype(np.uint8) * 255))
        (sample_dir / "metrics.json").write_text(
            json.dumps({"meta": meta, "features": feats}, indent=2))

        with self.csv_path.open("a", newline="") as fh:
            csv.writer(fh).writerow(
                [meta.get(c, "") for c in META_COLUMNS] + feature_row(feats))
        self._written.append(sample_dir)
        return sample_dir

    def undo(self):
        """Drop the last sample -- a mislabelled row poisons training."""
        if not self._written:
            return None
        last = self._written.pop()
        shutil.rmtree(last, ignore_errors=True)
        rows = list(csv.reader(self.csv_path.open()))
        if len(rows) > 1:
            with self.csv_path.open("w", newline="") as fh:
                csv.writer(fh).writerows(rows[:-1])
        return last

    def counts(self) -> dict:
        if not self.csv_path.exists():
            return {}
        out = {}
        with self.csv_path.open() as fh:
            for row in csv.DictReader(fh):
                out[row["label"]] = out.get(row["label"], 0) + 1
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=None)
    ap.add_argument("--model", default="weights/GoldSegmentationbest1.pt")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--label", default=None, choices=sorted(set(LABEL_KEYS.values())),
                    help="preset the label so a whole batch needs no keypresses")
    ap.add_argument("--object-id", default=None,
                    help="identifier for the physical piece; poses of one piece "
                         "MUST share it or your train/test split will leak")
    ap.add_argument("--notes", default="")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    if args.headless and not args.label:
        ap.error("--headless needs --label (no keyboard to label with)")

    cfg = PhysicsConfig.load(args.config)
    if args.camera is not None:
        cfg.camera.source = int(args.camera) if args.camera.isdigit() else args.camera
    if args.rotate is not None:
        cfg.camera.rotate = args.rotate

    from ultralytics import YOLO
    model = YOLO(args.model)
    names = getattr(model, "names", {}) or {}

    source = CameraSource(cfg.camera).open(lock=True)
    if source.lock_report:
        print(f"[camera] {source.lock_report.summary()}")
        for w in source.lock_report.warnings:
            print(f"[camera] WARNING: {w}")
    if not cfg.chips.configured:
        print("[chips] WARNING: no white reference patch. Samples will be "
              "recorded with calibrated=0 and are not comparable across "
              "lighting changes. Run tools/calibrate_chips.py first.")

    stage = MaterialStage(cfg)
    writer = DatasetWriter(Path(args.out))
    print(f"[dataset] {args.out} currently holds {writer.counts()}")

    label = args.label
    object_id = args.object_id or uuid.uuid4().hex[:8]
    pose = 0
    armed = label is not None
    frames_kept: list = []

    win = "capture dataset"
    if not args.headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        print("\nKeys: g gold  p plastic  b brass  s silver  o other | "
              "n next pose  u undo  r reset  q quit\n")

    try:
        while True:
            ok, frame = source.read()
            if not ok:
                time.sleep(0.02)
                continue

            results = model(frame, conf=0.2, verbose=False)[0]
            h, w = frame.shape[:2]
            raw = results.masks.data.cpu().numpy() if getattr(results, "masks", None) is not None else None

            dets = []
            for i, box in enumerate(results.boxes):
                coords = tuple(map(int, box.xyxy[0]))
                if raw is not None and i < len(raw):
                    m = cv2.resize((raw[i] * 255).astype(np.uint8), (w, h),
                                   interpolation=cv2.INTER_NEAREST) > 127
                else:
                    m = np.zeros((h, w), bool)
                    x1, y1, x2, y2 = coords
                    m[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)] = True
                if int(m.sum()) < 100:
                    continue
                cls_id = int(box.cls.item()) if box.cls is not None else -1
                dets.append(Detection(coords, m, float(box.conf.item()),
                                      str(names.get(cls_id, ""))))

            result = stage.process(frame, dets) if armed else None

            if armed and result is not None:
                if len(frames_kept) < 24 and stage.tracker.tracks:
                    frames_kept.append(frame.copy())

                done = [t for t in stage.tracker.tracks if t.decided]
                if done:
                    track = done[0]
                    verdict = track.verdict
                    if verdict.state is Verdict.INVALID_NO_SWEEP:
                        print(f"  -> discarded: {verdict.reason}")
                    else:
                        feats = extract_features(track.accumulator, verdict)
                        meta = {
                            "sample_id": datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6],
                            "object_id": object_id,
                            "label": label,
                            "captured_at": datetime.now().isoformat(timespec="seconds"),
                            "pose": pose,
                            "verdict": verdict.state.value,
                            "verdict_reason": verdict.reason,
                            "ornament_type": track.class_name,
                            "detector_confidence": round(track.confidence, 4),
                            "calibrated": int(verdict.calibrated),
                            "camera_source": str(cfg.camera.source),
                            "notes": args.notes,
                        }
                        path = writer.write(meta, feats, frames_kept, track.mask)
                        agree = "agrees" if (
                            (label == "gold") == (verdict.state is Verdict.GOLD_LIKE)
                        ) else "DISAGREES with label"
                        print(f"  saved {label}/{meta['sample_id']}  "
                              f"physics={verdict.state.value} ({agree})  -> {path}")
                        print(f"  totals: {writer.counts()}")
                        pose += 1

                    stage.reset()
                    frames_kept = []
                    armed = args.label is not None and args.headless

            if not args.headless:
                disp = frame.copy()
                for d in dets:
                    cv2.rectangle(disp, d.box[:2], d.box[2:], (0, 255, 0), 2)
                status = (f"{label} pose {pose} obj {object_id}" if armed
                          else "press a label key to arm")
                cv2.putText(disp, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
                if armed and stage.tracker.tracks:
                    cv2.putText(disp, stage.tracker.tracks[0].progress(), (10, 58),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow(win, disp)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key in LABEL_KEYS:
                    label = LABEL_KEYS[key]
                    object_id = uuid.uuid4().hex[:8]
                    pose = 0
                    armed = True
                    stage.reset()
                    frames_kept = []
                    print(f"[armed] {label} (new object {object_id}) -- sweep the lamp now")
                elif key == ord("n") and label:
                    armed = True
                    stage.reset()
                    frames_kept = []
                    print(f"[armed] {label} pose {pose} (same object {object_id})")
                elif key == ord("u"):
                    removed = writer.undo()
                    print(f"[undo] removed {removed}" if removed else "[undo] nothing to remove")
                elif key == ord("r"):
                    stage.reset()
                    frames_kept = []
                    print("[reset]")
    finally:
        source.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print(f"\nFinal counts: {writer.counts()}")
        print(f"Features CSV: {writer.csv_path}")


if __name__ == "__main__":
    main()
