"""Tests for Model B feature extraction and dataset capture."""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from physics.config import Thresholds  # noqa: E402
from physics.features import FEATURE_NAMES, extract_features, feature_row  # noqa: E402
from physics.specular import SweepAccumulator  # noqa: E402
from tools.capture_dataset import META_COLUMNS, DatasetWriter  # noqa: E402

GOLD = np.array([0.95, 0.65, 0.25], dtype=np.float32)
BODY = np.array([0.80, 0.70, 0.15], dtype=np.float32)


def _accumulate(kind, seed=2, frames=50):
    t = Thresholds(min_samples=500)
    acc = SweepAccumulator(t, seed=seed)
    rng = np.random.default_rng(seed)
    for f in range(frames):
        drive = 0.15 + 0.8 * np.sin(np.pi * f / frames) ** 2
        shade = rng.uniform(0.25, 1.0, (48, 48, 1)).astype(np.float32)
        px = shade * drive * GOLD if kind == "metal" else 0.2 * BODY + shade * drive * 0.5
        acc.add_frame(np.clip(px, 0, None), np.ones((48, 48), bool),
                      np.zeros((48, 48), bool), np.float32([1, 1, 1]), True)
    return acc


def test_every_feature_is_finite():
    acc = _accumulate("metal")
    feats = extract_features(acc, acc.decide(sweep_ratio=5.0, sweep_observed=True))
    row = feature_row(feats)
    assert len(row) == len(FEATURE_NAMES)
    assert all(np.isfinite(v) for v in row)


def test_features_separate_metal_from_dielectric():
    out = {}
    for kind in ("metal", "dielectric"):
        acc = _accumulate(kind)
        out[kind] = extract_features(acc, acc.decide(sweep_ratio=5.0, sweep_observed=True))
    assert out["metal"]["chroma_ratio"] > out["dielectric"]["chroma_ratio"] + 0.3
    assert out["metal"]["chroma_slope"] > out["dielectric"]["chroma_slope"]


def test_sparse_sweep_returns_zeros_not_garbage():
    """Too little data must yield an honest zero row, never an imputed guess."""
    acc = SweepAccumulator(Thresholds(min_samples=500))
    feats = extract_features(acc)
    assert set(feats) == set(FEATURE_NAMES)
    assert all(v == 0.0 for v in feats.values())


def test_dataset_writer_roundtrip(tmp_path):
    w = DatasetWriter(tmp_path / "ds")
    meta = {c: "x" for c in META_COLUMNS}
    meta.update(label="gold", sample_id="s1", object_id="obj1")
    feats = {n: 1.0 for n in FEATURE_NAMES}
    path = w.write(meta, feats, [np.zeros((8, 8, 3), np.uint8)], np.ones((8, 8), bool))

    assert path.exists()
    assert (path / "metrics.json").exists()
    assert (path / "mask.png").exists()
    assert w.counts() == {"gold": 1}

    rows = list(csv.reader(w.csv_path.open()))
    assert len(rows[0]) == len(rows[1]) == len(META_COLUMNS) + len(FEATURE_NAMES)


def test_dataset_writer_undo_removes_row_and_files(tmp_path):
    """A mislabelled sample poisons training, so undo must be complete."""
    w = DatasetWriter(tmp_path / "ds")
    meta = {c: "x" for c in META_COLUMNS}
    meta.update(label="plastic", sample_id="s1", object_id="obj1")
    path = w.write(meta, {n: 0.0 for n in FEATURE_NAMES}, [], None)
    assert w.counts() == {"plastic": 1}

    w.undo()
    assert w.counts() == {}
    assert not path.exists()
    assert len(list(csv.reader(w.csv_path.open()))) == 1  # header survives


def test_object_id_is_recorded_for_grouped_splitting(tmp_path):
    """Poses of one physical piece share an object_id; splitting on rows
    instead of groups leaks the same ring into train and test."""
    w = DatasetWriter(tmp_path / "ds")
    for pose in range(3):
        meta = {c: "x" for c in META_COLUMNS}
        meta.update(label="gold", sample_id=f"s{pose}", object_id="ring-7", pose=pose)
        w.write(meta, {n: 0.0 for n in FEATURE_NAMES}, [], None)
    rows = list(csv.DictReader(w.csv_path.open()))
    assert {r["object_id"] for r in rows} == {"ring-7"}
    assert len(rows) == 3
