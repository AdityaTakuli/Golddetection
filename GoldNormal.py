"""
GoldNormal.py -- Dual-camera gold detection system (v2)
========================================================

Architecture:  record first, process second.

Main thread (C270):
  detect gold -> start dual recording -> stop after 10s tail ->
  insert raw event as 'pending' -> enqueue for background processing

Lenovo thread:
  continuous read + rotate 180 -> frame buffer under lock

PostProcessWorker (single daemon thread, FIFO):
  pick pending event -> extract gold crop from C270 video ->
  match Lenovo frame -> run OCR on clean frame -> update DB row

Both cameras are mounted upside-down: rotate 180 before anything.
"""

import cv2
import time
import uuid
import queue
import threading
import logging
import sqlite3
from pathlib import Path
from datetime import datetime

import json as _json
import numpy as np
import easyocr
from ultralytics import YOLO

from physics.config import PhysicsConfig
from physics.camera import CameraSource
from physics.pipeline import MaterialStage
from physics.tracker import Detection as MaterialDetection
from physics.specular import Verdict


# ---------------------------------------------------------------------------
#  LOGGING
# ---------------------------------------------------------------------------
Path("runs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("runs/detection.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("GoldNormal")

# Project root (absolute) -- all paths relative to this
PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
#  DATABASE
# ---------------------------------------------------------------------------
class DB:
    """Thread-safe SQLite wrapper with pending/done/failed workflow."""

    def __init__(self, path="runs/gold.db"):
        abs_path = str((PROJECT_ROOT / path).resolve())
        (PROJECT_ROOT / path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(abs_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init()

    def _init(self):
        with self._lock:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id            TEXT UNIQUE,
                    captured_at         TEXT,
                    duration_sec        REAL,

                    c270_video_path     TEXT NOT NULL,
                    lenovo_video_path   TEXT,

                    weight              TEXT,
                    image_c270          TEXT,
                    image_lenovo        TEXT,

                    detection_confidence REAL,
                    bbox_json           TEXT,
                    sync_offset_ms      INTEGER,

                    processing_status   TEXT NOT NULL DEFAULT 'pending',
                    processing_error    TEXT,

                    -- Material verdict from the physics stage. Kept separate
                    -- from processing_status: a row can be fully processed and
                    -- still carry a non-confirming verdict.
                    material_verdict    TEXT,
                    material_reason     TEXT,
                    material_metrics    TEXT,

                    queued_at           TEXT,
                    processed_at        TEXT
                )
            """)
            self._migrate()
            self.conn.commit()

    def _migrate(self):
        """Add columns to databases created before the physics stage existed.

        Called with the lock already held.
        """
        have = {r["name"] for r in self.conn.execute("PRAGMA table_info(detections)")}
        for col in ("material_verdict", "material_reason", "material_metrics"):
            if col not in have:
                self.conn.execute(f"ALTER TABLE detections ADD COLUMN {col} TEXT")
                log.info("DB migrated: added column %s", col)

    def insert_raw_event(self, event_id, captured_at, duration_sec,
                         c270_video_path, lenovo_video_path):
        """Insert immediately after both writers are released."""
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO detections
                       (event_id, captured_at, duration_sec,
                        c270_video_path, lenovo_video_path,
                        processing_status, queued_at)
                   VALUES (?, ?, ?, ?, ?, 'pending', ?)""",
                (event_id, captured_at, duration_sec,
                 c270_video_path, lenovo_video_path, now),
            )
            self.conn.commit()
            return cur.lastrowid

    def update_processed(self, event_id, weight, image_c270, image_lenovo,
                         detection_confidence=None, bbox_json=None,
                         sync_offset_ms=None, material_verdict=None,
                         material_reason=None, material_metrics=None):
        """Called by PostProcessWorker after extraction.
        Status is 'done' if all data present, 'partial' if some missing."""
        now = datetime.now().isoformat(timespec="seconds")
        # Determine status: partial if any key field is missing
        if image_c270 and weight and weight != "None":
            status = "done"
        else:
            status = "partial"
        with self._lock:
            self.conn.execute(
                """UPDATE detections
                   SET weight=?, image_c270=?, image_lenovo=?,
                       detection_confidence=?, bbox_json=?,
                       sync_offset_ms=?,
                       material_verdict=?, material_reason=?, material_metrics=?,
                       processing_status=?, processed_at=?
                   WHERE event_id=?""",
                (weight, image_c270, image_lenovo,
                 detection_confidence, bbox_json, sync_offset_ms,
                 material_verdict, material_reason, material_metrics,
                 status, now, event_id),
            )
            self.conn.commit()

    def update_failed(self, event_id, error_msg):
        """Called by PostProcessWorker on failure."""
        with self._lock:
            self.conn.execute(
                """UPDATE detections
                   SET processing_status='failed', processing_error=?
                   WHERE event_id=?""",
                (error_msg, event_id),
            )
            self.conn.commit()

    def update_processing(self, event_id):
        """Mark row as currently being processed."""
        with self._lock:
            self.conn.execute(
                """UPDATE detections SET processing_status='processing'
                   WHERE event_id=?""",
                (event_id,),
            )
            self.conn.commit()

    def get_pending_events(self):
        """Return event_ids of rows still pending or failed (for recovery)."""
        with self._lock:
            rows = self.conn.execute(
                """SELECT event_id FROM detections
                   WHERE processing_status IN ('pending', 'failed')
                   ORDER BY queued_at ASC"""
            ).fetchall()
            return [r["event_id"] for r in rows]

    def get_event(self, event_id):
        """Return a single event row as dict."""
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM detections WHERE event_id=?", (event_id,)
            ).fetchone()
            return dict(row) if row else None


# ---------------------------------------------------------------------------
#  LENOVO CAMERA  (daemon thread, passive frame buffer)
# ---------------------------------------------------------------------------
class LenovoCamera:
    """
    Runs in a daemon thread. Main loop never waits for it.
    capture_latest() returns the most recent rotated frame instantly.
    """

    def __init__(self, cfg, opener=None):
        """cfg is a physics.config.CameraConfig, so this accepts a USB index,
        an RTSP/HTTP dome camera over Ethernet, or a custom opener.
        Rotation now comes from cfg.rotate rather than being hardcoded."""
        self.cfg = cfg
        self._source = CameraSource(cfg, opener=opener)
        try:
            self._source.open(lock=True)
            self._available = True
        except RuntimeError as exc:
            log.warning("Context cam (%s) unavailable: %s", cfg.source, exc)
            self._available = False
        self._frame = None
        self._lock  = threading.Lock()
        self._running = False

    @property
    def available(self):
        return self._available

    def start(self):
        if not self._available:
            return
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True, name="LenovoCam")
        t.start()
        log.info("Lenovo camera thread started.")

    def _loop(self):
        while self._running:
            ret, frame = self._source.read()   # rotation applied by CameraSource
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.03)

    def capture_latest(self):
        """Return a copy of the latest frame, or None."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        if self._available:
            self._source.release()


# ---------------------------------------------------------------------------
#  PERSON SEGMENTATION  (YOLO)
# ---------------------------------------------------------------------------
class YOLOSegmentation:
    def __init__(self, model_path: str, classes=None):
        self.model   = YOLO(model_path)
        self.classes = classes or [0]

    def run(self, frame):
        """Return raw results on the frame."""
        return self.model(frame, classes=self.classes, conf=0.2)[0]

    def draw(self, frame, results):
        """Overlay masks + boxes on display_frame."""
        if results.masks is not None:
            for pts in results.masks.xy:
                c = np.array(pts, dtype=np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [c], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.polylines(frame, [c], True, (0, 255, 0), 2)
        if results.boxes is not None:
            for box in results.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        return frame


# ---------------------------------------------------------------------------
#  GOLD DETECTOR
# ---------------------------------------------------------------------------
def rasterise_person_mask(person_masks, frame_shape):
    """Burn every person polygon into one mask, once per frame.

    Previously this was rebuilt from scratch inside the per-box overlap
    test, so a frame with N gold boxes allocated 2N full-frame arrays. At
    1080p/30fps that is hundreds of MB/s of churn to answer a question that
    only needs a lookup.
    """
    if not person_masks:
        return None
    h, w = frame_shape[:2]
    m = np.zeros((h, w), dtype=np.uint8)
    for pts in person_masks:
        cv2.fillPoly(m, [np.array(pts, dtype=np.int32)], 255)
    return m


def _overlaps_person(box_coords, person_bin):
    """True if the box touches any person pixel. Tests the box region only."""
    if person_bin is None:
        return False
    x1, y1, x2, y2 = box_coords
    h, w = person_bin.shape[:2]
    x1, y1 = max(int(x1), 0), max(int(y1), 0)
    x2, y2 = min(int(x2), w), min(int(y2), h)
    if x2 <= x1 or y2 <= y1:
        return False
    return bool(person_bin[y1:y2, x1:x2].any())


class GoldDetector:
    """Localises ornaments. Deliberately does NOT decide what they are made of.

    A detector trained on RGB crops cannot separate gold from yellow
    plastic: under uncontrolled light they are the same pixels, so the
    information is absent from its input. Scaling the model does not
    create it. Material is the physics stage's call (see physics.specular);
    this model's job is a tight mask and an ornament type.
    """

    def __init__(self, model_path: str, conf: float = 0.2):
        self.model = YOLO(model_path)
        self.conf = conf
        self.names = getattr(self.model, "names", {}) or {}

    def detect(self, frame, person_bin=None):
        """Run inference and drop anything overlapping a person.

        Returns physics Detections carrying a per-object mask, so the
        material stage samples only object pixels. Segmentation masks are
        used when the model provides them; a detect-only model falls back
        to the filled box, which is looser but still works.
        """
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        h, w = frame.shape[:2]

        raw_masks = None
        if getattr(results, "masks", None) is not None:
            raw_masks = results.masks.data.cpu().numpy()

        out = []
        for i, box in enumerate(results.boxes):
            coords = tuple(map(int, box.xyxy[0]))
            if _overlaps_person(coords, person_bin):
                continue

            if raw_masks is not None and i < len(raw_masks):
                m = cv2.resize((raw_masks[i] * 255).astype(np.uint8), (w, h),
                               interpolation=cv2.INTER_NEAREST) > 127
            else:
                m = np.zeros((h, w), dtype=bool)
                x1, y1, x2, y2 = coords
                m[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)] = True

            if int(m.sum()) < 100:
                continue

            cls_id = int(box.cls.item()) if box.cls is not None else -1
            out.append(MaterialDetection(
                box=coords,
                mask=m,
                confidence=float(box.conf.item()) if box.conf is not None else 0.0,
                class_name=str(self.names.get(cls_id, "")),
            ))
        return out

    @staticmethod
    def draw_boxes(frame, gold_list):
        """Draw gold boxes on a display_frame copy only."""
        for cx1, cy1, cx2, cy2 in gold_list:
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
        return frame


# ---------------------------------------------------------------------------
#  DUAL VIDEO RECORDER  (10-second tail, two synchronized writers)
# ---------------------------------------------------------------------------
class DualVideoRecorder:
    TAIL_SECS = 10

    def __init__(self, out_dir: str = "runs/recordings"):
        self.out_dir = (PROJECT_ROOT / out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # active event state
        self._c270_writer   = None
        self._lenovo_writer = None
        self._event_id      = None
        self._captured_at   = None
        self._t0            = 0.0
        self._last_gold_t   = 0.0
        self.recording      = False

    @property
    def event_id(self):
        return self._event_id

    @property
    def c270_path(self):
        return str(self.out_dir / f"{self._event_id}_c270.mp4") if self._event_id else None

    @property
    def lenovo_path(self):
        return str(self.out_dir / f"{self._event_id}_lenovo.mp4") if self._event_id else None

    def feed(self, c270_frame, lenovo_frame, gold_detected, gold_list=None):
        """
        Feed both camera frames each loop iteration.

        Returns
        -------
        recording     bool
        just_stopped  bool
        completed     dict | None   -- event info when recording just ended
        """
        now = time.time()

        if gold_detected:
            self._last_gold_t = now
            if not self.recording:
                self._start(c270_frame, lenovo_frame)

        just_stopped = False
        completed    = None

        if self.recording:
            # Draw green gold boxes on the recorded C270 video
            rec_frame = c270_frame.copy()
            if gold_list:
                for (x1, y1, x2, y2) in gold_list:
                    cv2.rectangle(rec_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            self._c270_writer.write(rec_frame)
            if self._lenovo_writer is not None and lenovo_frame is not None:
                self._lenovo_writer.write(lenovo_frame)

            tail_expired = (not gold_detected and
                            (now - self._last_gold_t) > self.TAIL_SECS)
            if tail_expired:
                completed = self._stop()
                just_stopped = True

        return self.recording, just_stopped, completed

    def force_stop(self):
        """Call on shutdown. Returns completed event dict or None."""
        if self.recording:
            return self._stop()
        return None

    def _start(self, c270_frame, lenovo_frame):
        self._event_id   = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self._captured_at = datetime.now().isoformat(timespec="seconds")
        self._t0          = time.time()

        # C270 writer
        h, w = c270_frame.shape[:2]
        self._c270_writer = cv2.VideoWriter(
            self.c270_path, self._fourcc, 20.0, (w, h),
        )

        # Lenovo writer (if frame available)
        if lenovo_frame is not None:
            lh, lw = lenovo_frame.shape[:2]
            self._lenovo_writer = cv2.VideoWriter(
                self.lenovo_path, self._fourcc, 20.0, (lw, lh),
            )
        else:
            self._lenovo_writer = None

        self.recording = True
        log.info("REC started  event=%s", self._event_id)

    def _stop(self):
        duration = time.time() - self._t0

        if self._c270_writer:
            self._c270_writer.release()
            self._c270_writer = None
        lenovo_path = None
        if self._lenovo_writer:
            self._lenovo_writer.release()
            self._lenovo_writer = None
            lenovo_path = self.lenovo_path

        self.recording = False
        log.info("REC stopped  event=%s  dur=%.1fs", self._event_id, duration)

        completed = {
            "event_id":          self._event_id,
            "captured_at":       self._captured_at,
            "duration_sec":      round(duration, 1),
            "c270_video_path":   self.c270_path,
            "lenovo_video_path": lenovo_path,
        }
        self._event_id = None
        return completed

# ---------------------------------------------------------------------------
#  OCR  (weight reader)
# ---------------------------------------------------------------------------
class OCRReader:
    def __init__(self):
        log.info("Initialising EasyOCR...")
        try:
            self.reader = easyocr.Reader(["en"], gpu=True)
            log.info("EasyOCR ready.")
        except Exception as exc:
            log.warning("EasyOCR init failed: %s", exc)
            self.reader = None

    def read(self, frame) -> str:
        """Return first digit-containing string found in frame, else 'None'."""
        if self.reader is None:
            return "None"
        try:
            for _, text, _ in self.reader.readtext(frame):
                if any(ch.isdigit() for ch in text):
                    return text
        except Exception as exc:
            log.warning("OCR error: %s", exc)
        return "None"


# ---------------------------------------------------------------------------
#  UTILITIES
# ---------------------------------------------------------------------------
def get_screen_resolution():
    try:
        import tkinter as tk
        r = tk.Tk(); r.withdraw()
        res = r.winfo_screenwidth(), r.winfo_screenheight()
        r.destroy();  return res
    except Exception:
        pass
    try:
        import subprocess
        for line in subprocess.check_output(["xrandr"]).decode().splitlines():
            if "*" in line:
                w, h = map(int, line.split()[0].split("x"))
                return w, h
    except Exception:
        pass
    return None


def resize_fit(frame, max_w, max_h):
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h)
    return cv2.resize(frame, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_LINEAR)


VERDICT_COLOURS = {
    "GOLD_LIKE":        (0, 215, 255),
    "NON_GOLD_METAL":   (200, 200, 200),
    "DIELECTRIC":       (30, 30, 200),
    "UNCERTAIN":        (80, 160, 220),
    "INVALID_CLIPPED":  (0, 140, 255),
    "INVALID_NO_SWEEP": (120, 120, 120),
}


def draw_hud(frame, gold_detected, recording, rec_start,
             last_weight, save_flash, verdict=None, physics_lines=None):
    """Draw status panel on display_frame only."""
    h, w = frame.shape[:2]
    px   = w - 240
    font = cv2.FONT_HERSHEY_SIMPLEX

    def put(text, y, color):
        cv2.putText(frame, text, (px, y), font, 0.6, color, 2)

    put("Gold: YES" if gold_detected else "Gold: NO",
        30,  (0, 255, 0) if gold_detected else (0, 0, 255))
    put("REC:  ON " if recording else "REC:  OFF",
        65,  (0, 255, 0) if recording else (0, 0, 255))

    if recording and rec_start:
        e = int(time.time() - rec_start)
        put(f"Dur:  {e // 60:02d}:{e % 60:02d}", 100, (255, 255, 255))
    else:
        put("Dur:  00:00", 100, (255, 255, 255))

    wt = last_weight if last_weight != "None" else "--"
    put(f"Wt:   {wt}", 135, (255, 255, 0))

    y = 170
    if verdict is not None:
        put(f"Mat:  {verdict.state.value}", y,
            VERDICT_COLOURS.get(verdict.state.value, (255, 255, 255)))
        y += 30

    # Operator guidance -- the sweep only works if a human knows to do it.
    for line in (physics_lines or [])[:4]:
        cv2.putText(frame, line[:34], (px, y), font, 0.45, (200, 200, 200), 1)
        y += 22

    if save_flash:
        put("Event saved!", y, (0, 255, 255))

    return frame


# ---------------------------------------------------------------------------
#  MAIN SYSTEM
# ---------------------------------------------------------------------------
class DualCameraSystem:
    """
    Orchestrates C270 (detection) + Lenovo (context) with
    dual recording. Snapshots and OCR captured inline.
    """

    FLASH_SECS = 2.0
    MAX_READ_FAILURES = 100   # ~3s of dead camera before giving up

    def __init__(
        self,
        gold_model_path: str,
        seg_model_path:  str,
        config:          PhysicsConfig = None,
        primary_opener=None,
        context_opener=None,
        show_window:     bool = True,
        max_frames:      int = 0,
    ):
        """Cameras come from PhysicsConfig, so either can be a USB index or an
        RTSP/HTTP dome camera. `primary_opener`/`context_opener` let existing
        connection code supply its own VideoCapture."""
        self.cfg = config or PhysicsConfig.load()
        # A kiosk on a Pi has no X display, and neither does a CI run.
        self.show_window = show_window
        self.max_frames = max_frames   # 0 = run until stopped

        # -- primary (detection) camera --
        self.source = CameraSource(self.cfg.camera, opener=primary_opener).open(lock=True)
        log.info("Primary camera opened: %s", self.cfg.camera.source)
        if self.source.lock_report and not self.source.lock_report.ok:
            log.warning(
                "Primary camera controls are not locked. Auto exposure and auto "
                "white balance will corrupt the material verdict -- auto-exposure "
                "moves the gain the instant the lamp arrives, and AWB exists to "
                "cancel exactly the colour cast the gold signal consists of."
            )

        # -- context camera (optional) --
        ctx = self.cfg.context_camera
        self.lenovo = LenovoCamera(ctx, opener=context_opener) if ctx else None
        if self.lenovo is not None:
            self.lenovo.start()

        # -- detection models (main thread) --
        self.seg  = YOLOSegmentation(seg_model_path)
        self.gold = GoldDetector(gold_model_path)
        self.ocr  = OCRReader()

        # -- physics: the stage that actually decides material --
        self.material = MaterialStage(self.cfg)

        # -- subsystems --
        self.recorder = DualVideoRecorder()
        self.db       = DB()

        # -- per-event state --
        self._rec_start      = None
        self._prev_recording = False
        self._save_flash_t   = 0.0
        self._last_ocr_t     = 0.0
        self._last_weight    = "None"
        self._stop           = False
        self._read_failures  = 0

        # Snapshot data captured when gold first appears
        self._pending          = None
        self._ocr_settle_start = 0.0
        self._ocr_settled      = False

        Path(PROJECT_ROOT / "runs" / "images").mkdir(parents=True, exist_ok=True)

    # -- inline helpers ---
    def _save_snapshot(self, frame, prefix, event_id, crop_box=None):
        """Save a frame (optionally cropped to gold box) and return absolute path."""
        path = str(PROJECT_ROOT / "runs" / "images" / f"{event_id}_{prefix}.jpg")
        if crop_box:
            x1, y1, x2, y2 = crop_box
            h, w = frame.shape[:2]
            px1, py1 = max(0, x1 - 20), max(0, y1 - 20)
            px2, py2 = min(w, x2 + 20), min(h, y2 + 20)
            cv2.imwrite(path, frame[py1:py2, px1:px2])
        else:
            cv2.imwrite(path, frame)
        return path

    def _union_box(self, gold_list):
        """Compute union bounding rectangle of all gold boxes."""
        if not gold_list:
            return None
        x1 = min(b[0] for b in gold_list)
        y1 = min(b[1] for b in gold_list)
        x2 = max(b[2] for b in gold_list)
        y2 = max(b[3] for b in gold_list)
        return (x1, y1, x2, y2)

    def _on_gold_first_seen(self, clean_frame, detections, event_id):
        """
        Called once when an ornament first appears. Captures snapshots + OCR
        using the already-loaded models. No background worker needed.

        Note this is only the *localisation* event -- it does not mean gold.
        The material verdict arrives later, once a sweep has been observed.
        """
        ts = datetime.now().isoformat(timespec="seconds")

        union = self._union_box([d.box for d in detections])
        img_c270 = self._save_snapshot(clean_frame, "c270_crop", event_id, crop_box=union)

        # Confidence comes from the detections we already have. The previous
        # version re-ran the model here purely to recover a number it had
        # just discarded.
        best_conf = round(max((d.confidence for d in detections), default=0.0), 4) or None
        bbox_dict = None
        if union:
            bbox_dict = {"x1": union[0], "y1": union[1],
                         "x2": union[2], "y2": union[3]}

        # Lenovo frame
        lenovo_frm = self.lenovo.capture_latest() if self.lenovo else None
        img_lenovo = self._save_snapshot(lenovo_frm, "lenovo_frame", event_id) \
            if lenovo_frm is not None else None

        # OCR weight -- initial read; will be updated after 5s settle
        weight = self.ocr.read(clean_frame)

        self._pending = {
            "captured_at":          ts,
            "weight":               weight,
            "image_c270":           img_c270,
            "image_lenovo":         img_lenovo,
            "detection_confidence": best_conf,
            "bbox_json":            _json.dumps(bbox_dict) if bbox_dict else None,
            "material":             None,
            "ornament_type":        next((d.class_name for d in detections
                                          if d.class_name), ""),
        }
        # Start 5-second OCR settle timer
        self._ocr_settle_start = time.time()
        self._ocr_settled      = False
        log.info("Snapshots saved  C270=%s  Lenovo=%s  weight=%s  conf=%s",
                 img_c270, img_lenovo, weight, best_conf)

    def _on_recording_done(self, completed):
        """
        Called after both writers are released.
        Writes the complete DB row in one shot.
        """
        if self._pending is None:
            log.warning("Recording done but no pending data -- writing raw row.")
            row_id = self.db.insert_raw_event(**completed)
            log.info("DB row #%d inserted (pending) event=%s", row_id, completed["event_id"])
            self._save_flash_t = time.time()
            return

        ev = self._pending

        # First insert the raw event
        row_id = self.db.insert_raw_event(**completed)

        # processing_status is about pipeline completeness, not about
        # whether the piece is gold -- those are different questions and
        # collapsing them would let a DIELECTRIC verdict read as success.
        if ev["image_c270"] and ev["weight"] and ev["weight"] != "None":
            status = "done"
        else:
            status = "partial"

        verdict = ev.get("material")
        if verdict is None:
            v_state = Verdict.INVALID_NO_SWEEP.value
            v_reason = "capture ended before a sweep was observed"
            v_metrics = None
        else:
            v_state = verdict.state.value
            v_reason = verdict.reason
            v_metrics = _json.dumps(verdict.as_dict())

        # Then immediately update with the inline-captured data
        self.db.update_processed(
            completed["event_id"],
            weight=ev["weight"],
            image_c270=ev["image_c270"],
            image_lenovo=ev["image_lenovo"],
            detection_confidence=ev["detection_confidence"],
            bbox_json=ev["bbox_json"],
            sync_offset_ms=0,  # inline capture = same moment
            material_verdict=v_state,
            material_reason=v_reason,
            material_metrics=v_metrics,
        )

        log.info("DB row #%d written (%s)  event=%s  weight=%s  material=%s",
                 row_id, status, completed["event_id"], ev["weight"], v_state)
        if v_state != Verdict.GOLD_LIKE.value:
            log.warning("Event %s NOT confirmed as gold: %s",
                        completed["event_id"], v_reason)

        self._pending      = None
        self._save_flash_t = time.time()

    # -- main loop --
    def run(self):
        win = "Gold Detection"
        disp_w, disp_h = 1280, 720
        if self.show_window:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            res = get_screen_resolution()
            disp_w, disp_h = res if res else (1280, 720)
            cv2.resizeWindow(win, disp_w, disp_h)
            log.info("Main loop running -- press Q to quit.")
        else:
            log.info("Main loop running headless.")

        frames_done = 0
        while not self._stop:
            if self.max_frames and frames_done >= self.max_frames:
                break
            frames_done += 1
            ret, frame = self.source.read()
            if not ret:
                self._read_failures += 1
                # A stream reconnects itself; a USB camera that has stopped
                # answering is unplugged or wedged, and spinning on it
                # forever just fills the log.
                if (not self.cfg.camera.is_stream
                        and self._read_failures >= self.MAX_READ_FAILURES):
                    log.error("Primary camera gave %d consecutive read failures "
                              "-- stopping.", self._read_failures)
                    break
                if self._read_failures % 30 == 1:
                    log.warning("Primary camera read failed (%d) -- retrying...",
                                self._read_failures)
                time.sleep(0.03)
                continue
            self._read_failures = 0

            # -- 1. clean_frame = unannotated source of truth --
            # (rotation already applied by CameraSource per cfg.rotate)
            clean_frame = frame.copy()

            # -- 2. person segmentation (runs on clean_frame) --
            seg_results  = self.seg.run(clean_frame)
            person_masks = seg_results.masks.xy if seg_results.masks else None
            person_bin   = rasterise_person_mask(person_masks, clean_frame.shape)

            # -- 3. localise ornaments (material is decided later, by physics) --
            detections    = self.gold.detect(clean_frame, person_bin)
            gold_list     = [d.box for d in detections]
            gold_detected = len(detections) > 0

            # -- 4. physics stage: the actual material test --
            # Same frame, same masks, same instant -- the consistency the
            # old threaded sweep worker could not guarantee.
            material = self.material.process(clean_frame, detections)

            # -- 5. get latest context frame --
            lenovo_frame = self.lenovo.capture_latest() if self.lenovo else None

            # -- 6. dual recording (writes clean frames only) --
            recording, just_stopped, completed = self.recorder.feed(
                clean_frame, lenovo_frame, gold_detected, gold_list
            )

            # -- 7. handle state transitions --
            if recording and not self._prev_recording:
                # Gold just appeared -> first frame of recording
                self._rec_start = time.time()
                event_id = self.recorder.event_id
                self.material.reset()   # new capture event, new sweep
                self._on_gold_first_seen(clean_frame, detections, event_id)

            elif not recording and self._prev_recording:
                self._rec_start = None

            if just_stopped and completed:
                self._on_recording_done(completed)

            if self._pending is not None and material.verdict is not None:
                self._pending["material"] = material.verdict

            self._prev_recording = recording

            # -- 8. OCR update while gold visible (5-second settle) --
            now = time.time()
            if gold_detected and self._pending and not self._ocr_settled:
                if (now - self._last_ocr_t) >= 1.0:
                    self._last_ocr_t = now
                    w = self.ocr.read(clean_frame)
                    if w != "None":
                        self._last_weight = w
                        self._pending["weight"] = w

                # After 5 seconds, lock in the weight
                if (now - self._ocr_settle_start) >= 5.0:
                    self._ocr_settled = True
                    log.info("OCR settled: weight=%s", self._pending.get("weight"))

            # -- 9. display_frame = annotated copy for HUD --
            display_frame = clean_frame.copy()
            display_frame = self.seg.draw(display_frame, seg_results)
            display_frame = GoldDetector.draw_boxes(display_frame, gold_list)

            save_flash = (now - self._save_flash_t) < self.FLASH_SECS

            display_frame = draw_hud(
                display_frame, gold_detected, recording,
                self._rec_start, self._last_weight, save_flash,
                verdict=material.verdict,
                physics_lines=self.material.hud_lines(material),
            )

            if self.show_window:
                cv2.imshow(win, resize_fit(display_frame, disp_w, disp_h))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        # -- shutdown --
        log.info("Shutting down...")
        completed = self.recorder.force_stop()
        if completed:
            self._on_recording_done(completed)

        self.source.release()
        if self.lenovo:
            self.lenovo.stop()
        if self.show_window:
            cv2.destroyAllWindows()
        log.info("Done.")


# ---------------------------------------------------------------------------
#  ENTRY POINT
# ---------------------------------------------------------------------------
def _build_arg_parser():
    import argparse
    ap = argparse.ArgumentParser(
        description="Gold detection with physics-based material verification")
    ap.add_argument("--config", default=None,
                    help="path to physics config JSON (default config/physics.json)")
    ap.add_argument("--camera", default=None,
                    help="override primary camera: a USB index (0) or a stream "
                         "URL (rtsp://user:pass@host:554/...)")
    ap.add_argument("--context-camera", default=None,
                    help="override context camera; same forms as --camera")
    ap.add_argument("--rotate", type=int, default=None, choices=[0, 90, 180, 270],
                    help="rotate primary camera frames")
    ap.add_argument("--gold-model", default="weights/GoldSegmentationbest1.pt")
    ap.add_argument("--seg-model", default="weights/yolo26n-seg.onnx")
    ap.add_argument("--headless", action="store_true",
                    help="run without a display window (kiosk / SSH / Pi)")
    ap.add_argument("--no-lock", action="store_true",
                    help="skip camera control locking (NOT recommended: auto "
                         "exposure and auto white balance corrupt the verdict)")
    return ap


def _as_source(text):
    """"0" -> 0 (USB index); anything with a scheme stays a URL."""
    return int(text) if text.isdigit() else text


if __name__ == "__main__":
    import signal

    args = _build_arg_parser().parse_args()
    cfg = PhysicsConfig.load(args.config)

    if args.camera is not None:
        cfg.camera.source = _as_source(args.camera)
    if args.rotate is not None:
        cfg.camera.rotate = args.rotate
    if args.context_camera is not None:
        from physics.config import CameraConfig
        cfg.context_camera = cfg.context_camera or CameraConfig(name="context")
        cfg.context_camera.source = _as_source(args.context_camera)
    if args.no_lock:
        cfg.camera.lock_controls = False
        if cfg.context_camera:
            cfg.context_camera.lock_controls = False

    if not cfg.chips.configured:
        log.warning(
            "No reference patch configured. The system will run, but chroma is "
            "measured against an assumed neutral illuminant and thresholds will "
            "drift. Run:  python3 tools/calibrate_chips.py"
        )

    system = DualCameraSystem(
        gold_model_path = args.gold_model,
        seg_model_path  = args.seg_model,
        config          = cfg,
        show_window     = not args.headless,
    )

    _stop_flag = False
    def _sigint_handler(signum, frame_arg):
        global _stop_flag
        if _stop_flag:
            raise SystemExit(1)
        log.info("Ctrl+C received -- shutting down cleanly...")
        _stop_flag = True
        system._stop = True
    signal.signal(signal.SIGINT, _sigint_handler)

    system.run()
