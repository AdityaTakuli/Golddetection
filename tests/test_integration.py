"""
End-to-end test of the live pipeline with stubbed models and camera.

Drives DualCameraSystem.run() over a scripted sweep and asserts the event
lands in the database with a material verdict. Catches the class of
breakage unit tests miss: a renamed field, a camera the loop cannot read,
a verdict that never reaches the DB row.

ultralytics, easyocr and torch are heavyweight and unavailable in CI, so
they are stubbed at import time. cv2 is real (headless build).
"""

import sys
import sqlite3
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GOLD = np.array([0.95, 0.65, 0.25], dtype=np.float32)


# ---------------------------------------------------------------- stubs
class _Box:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = [np.array(xyxy, dtype=np.float32)]
        self.conf = types.SimpleNamespace(item=lambda: conf)
        self.cls = types.SimpleNamespace(item=lambda: cls)


class _Masks:
    def __init__(self, arr, polys):
        self.data = types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(numpy=lambda: arr))
        self.xy = polys


class _Result:
    def __init__(self, boxes, masks):
        self.boxes = boxes
        self.masks = masks


class _FakeYOLO:
    """Detects one ornament in a fixed box; no people."""

    names = {0: "Ring"}

    def __init__(self, path, *a, **k):
        self.path = str(path)
        self.is_person_model = "26n-seg" in self.path

    def __call__(self, frame, *a, **k):
        if self.is_person_model:
            return [_Result([], None)]
        h, w = frame.shape[:2]
        m = np.zeros((1, h, w), dtype=np.float32)
        m[0, h // 4:3 * h // 4, w // 4:3 * w // 4] = 1.0
        box = _Box([w // 4, h // 4, 3 * w // 4, 3 * h // 4], 0.91, 0)
        return [_Result([box], _Masks(m, []))]


class _FakeReader:
    def __init__(self, *a, **k):
        pass

    def readtext(self, frame):
        return [((0, 0), "12.50 g", 0.99)]


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics",
                        types.SimpleNamespace(YOLO=_FakeYOLO))
    monkeypatch.setitem(sys.modules, "easyocr",
                        types.SimpleNamespace(Reader=_FakeReader))
    monkeypatch.chdir(tmp_path)

    import importlib
    for mod in [m for m in sys.modules if m.startswith("GoldNormal")]:
        del sys.modules[mod]
    gn = importlib.import_module("GoldNormal")
    importlib.reload(gn)
    monkeypatch.setattr(gn, "PROJECT_ROOT", tmp_path)
    return gn


class _ScriptedCamera:
    """Replays a lamp sweep across a gold-coloured object."""

    def __init__(self, frames=80, size=96, material="metal"):
        self.frames, self.size, self.material = frames, size, material
        self.i = 0
        self._rng = np.random.default_rng(11)

    def isOpened(self):
        return True

    def set(self, *a):
        return True

    def get(self, *a):
        return 0.0

    def release(self):
        pass

    def read(self):
        if self.i >= self.frames:
            return False, None
        f, n = self.i, self.size
        self.i += 1
        drive = 0.12 + 0.85 * np.sin(np.pi * max(f - 8, 0) / max(self.frames - 8, 1)) ** 2
        shade = self._rng.uniform(0.3, 1.0, (n, n, 1)).astype(np.float32)
        if self.material == "metal":
            lin = shade * drive * GOLD
        else:
            # keep the specular term below saturation: a blown highlight
            # is correctly reported INVALID_CLIPPED, which would not
            # exercise the dielectric rejection path we want here
            lin = 0.20 * np.array([0.80, 0.70, 0.15], np.float32) + shade * drive * 0.5
        srgb = np.clip(lin, 0, 1) ** (1 / 2.2)
        return True, (np.clip(srgb[..., ::-1], 0, 1) * 255).astype(np.uint8)


def _run(gn, tmp_path, material):
    from physics.config import PhysicsConfig
    cfg = PhysicsConfig()
    cfg.camera.lock_controls = False
    cfg.thresholds.baseline_frames = 5
    cfg.thresholds.min_samples = 500
    cfg.thresholds.min_sweep_frames = 4
    cfg.thresholds.timeout_frames = 70

    system = gn.DualCameraSystem(
        gold_model_path="weights/fake-gold.pt",
        seg_model_path="weights/yolo26n-seg.onnx",
        config=cfg,
        primary_opener=lambda c: _ScriptedCamera(material=material),
        show_window=False,
        max_frames=110,
    )
    system.recorder.TAIL_SECS = 0     # close the event without waiting
    system.run()
    return sqlite3.connect(tmp_path / "runs" / "gold.db")


def test_pipeline_records_event_with_gold_verdict(pipeline, tmp_path):
    conn = _run(pipeline, tmp_path, "metal")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM detections").fetchall()
    assert rows, "no detection event was written"
    row = rows[-1]
    assert row["material_verdict"] == "GOLD_LIKE", row["material_reason"]
    assert row["detection_confidence"] == pytest.approx(0.91, abs=1e-3)
    assert row["material_metrics"]


def test_pipeline_rejects_yellow_dielectric(pipeline, tmp_path):
    conn = _run(pipeline, tmp_path, "dielectric")
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM detections").fetchall()[-1]
    assert row["material_verdict"] == "DIELECTRIC", row["material_reason"]


def test_db_migration_adds_verdict_columns_to_old_database(pipeline, tmp_path):
    """A database created before the physics stage must upgrade in place."""
    db_dir = tmp_path / "runs"
    db_dir.mkdir(parents=True, exist_ok=True)
    old = sqlite3.connect(db_dir / "legacy.db")
    old.execute(
        "CREATE TABLE detections (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "event_id TEXT UNIQUE, captured_at TEXT, duration_sec REAL, "
        "c270_video_path TEXT NOT NULL, lenovo_video_path TEXT, weight TEXT, "
        "image_c270 TEXT, image_lenovo TEXT, detection_confidence REAL, "
        "bbox_json TEXT, sync_offset_ms INTEGER, "
        "processing_status TEXT NOT NULL DEFAULT 'pending', processing_error TEXT, "
        "queued_at TEXT, processed_at TEXT)")
    old.execute("INSERT INTO detections (event_id, c270_video_path) VALUES ('e1','/v.mp4')")
    old.commit(); old.close()

    pipeline.DB(path="runs/legacy.db")

    conn = sqlite3.connect(db_dir / "legacy.db")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(detections)")}
    assert {"material_verdict", "material_reason", "material_metrics"} <= cols
    assert conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 1
