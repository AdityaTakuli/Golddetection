# Gold Detection and Segmentation System

Real-time gold/jewellery detection system using YOLO models with person segmentation to suppress false positives, OCR weight reading, auto-recording, and a SQLite database for logging every detection event.

## How It Works

1. **Gold Detection** — Custom YOLO11n model detects gold objects in a defined ROI
2. **Person Segmentation** — YOLO26-seg masks out people so gold worn by a person is ignored
3. **Weight Reading** — EasyOCR reads the weight from the scale display
4. **Auto Recording** — Video is recorded automatically when gold is detected, stops 10s after last detection
5. **Database Logging** — Every detection is logged to SQLite with video path, weight, timestamp, and extracted image

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Run the detection system
python3 GoldNormal.py
```

Press **Q** to quit.

### GUI Viewer (browse detections)

```bash
pip install customtkinter pillow
python3 viewer/app.py
```

The viewer reads from the database and auto-refreshes every 5 seconds. It can run alongside the detection system.

## Requirements

### Python Dependencies

```bash
pip install ultralytics opencv-python easyocr numpy
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Optional: Database Viewer

To visually browse the detection database:

```bash
sudo apt install sqlitebrowser
sqlitebrowser runs/jewellery_detections.db
```

## Material verification (the physics stage)

The detector finds *where* an ornament is. It deliberately does **not**
decide what it is made of, because that question cannot be answered from an
RGB crop: under uncontrolled light, gold and yellow plastic are genuinely
the same pixels. The information is absent from the input, so no amount of
model capacity recovers it.

The material call is a physical measurement instead, based on the
dichromatic reflection model:

| | diffuse term | highlight behaviour |
|---|---|---|
| **Metal** (gold) | none | every pixel is a scalar multiple of one spectral vector, so chromaticity is **invariant** to brightness |
| **Dielectric** (plastic, paint, resin) | present | the specular term carries the *illuminant's* colour, so chromaticity **collapses toward white** |

So the discriminator is not "how yellow is it" — a yellow gate can never
reject yellow plastic — but *how chromaticity behaves as luminance rises*:

```
ratio = chroma(top luminance decile) / chroma(mid band)
slope = d(chroma) / d(normalised luminance)

metal      -> ratio ~ 1, slope ~ 0
dielectric -> ratio << 1, slope negative
```

Both are *shapes* of a distribution rather than absolute levels, so they
survive exposure changes, bulb colour temperature and sensor drift.

### It only works during a sweep

A brightness excursion has to be driven across the piece — a desk lamp or a
phone torch is fine to start with. Without one the test **refuses to rule**
(`INVALID_NO_SWEEP`) rather than guessing, because a dielectric under flat
light holds its chroma across shading exactly as a metal does. This is
enforced, and there is a regression test for it.

### Verdicts

| Verdict | Meaning |
|---|---|
| `GOLD_LIKE` | behaves optically like a warm metal |
| `NON_GOLD_METAL` | metal, but neutral/cool hue (silver, steel) |
| `DIELECTRIC` | plastic, paint, fabric, resin |
| `UNCERTAIN` | statistics between the bands |
| `INVALID_CLIPPED` | highlights saturated; measurement void |
| `INVALID_NO_SWEEP` | no brightness excursion observed |

### What this does NOT do

`GOLD_LIKE` means "behaves optically like a warm metal", nothing stronger.
It does **not** separate gold from **brass** (near-identical reflectance
curve), from **gold-plated** base metal (the surface really is gold), and it
does not resolve **karat**. Those need XRF, or a density measurement —
and since the rig already has a weighing scale, a hydrostatic cradle
(weigh in air, weigh submerged) gives specific gravity almost for free.
Note that tungsten matches gold's density, so density and XRF are
complementary rather than redundant.

## Setup

### 1. Lock the camera and calibrate (do this first)

Auto-exposure and auto white balance **destroy this measurement**.
Auto-exposure is a global gain that moves the instant the lamp arrives —
exactly when the signal appears — and AWB exists to cancel colour casts,
which is precisely what the gold signal is. Reading only masked pixels does
not escape either, because both act on those pixels too.

```bash
sudo apt install v4l-utils          # needed for a real control lock
python3 tools/calibrate_chips.py --find-exposure
```

This locks exposure/gain/WB/focus, searches downward for an exposure whose
highlights do not clip, and lets you mark the white (and optional grey)
reference patch.

**Reference patch material:** use PTFE if you can. Most white plastics,
papers and paints contain optical brighteners that absorb UV and re-emit
blue, so they shift colour with the lamp's UV content — a reference that
moves is worse than none.

Once a white patch is set, every chroma threshold becomes a *ratio* against
the light's own colour, which is what makes the verdict survive a bulb
change, ambient drift, or a dome camera whose settings you cannot reach.

### 2. Run

```bash
python3 GoldNormal.py                                  # USB camera
python3 GoldNormal.py --camera rtsp://user:pass@host:554/Streaming/Channels/101
python3 GoldNormal.py --headless                       # kiosk / SSH / no X
```

Either camera can be a USB index or an RTSP/HTTP dome camera over Ethernet.
Exposure cannot be set over RTSP — lock it in the camera's own web UI, once;
the pipeline warns rather than pretending it has control it does not have.

### 3. Collect data for Model B

```bash
python3 tools/capture_dataset.py
```

Keys: `g` gold, `p` plastic, `b` brass, `s` silver, `o` other, `n` next pose,
`u` undo, `q` quit. Writes `dataset/features.csv` (41 physics features per
sample) ready for LightGBM.

**Split on `object_id`, not on rows.** Ten poses of one ring are ten views of
one object. Split them randomly and the same ring lands in train and test,
and your accuracy will look excellent and mean nothing:

```python
from sklearn.model_selection import GroupKFold
GroupKFold().split(X, y, groups=df.object_id)
```

Aim for 200–500 distinct physical pieces, 10–20 poses each, and make sure
the plastic class contains the yellow pieces that actually fool the system
today — not easy negatives.

### Which model to train

- **Model A (have it):** YOLO11n-seg for *localisation only*. Its class
  labels also give you the ornament type for free.
- **Model B (the one that matters):** gradient-boosted trees over the 41
  physics features. Not a CNN — GBTs win at the sample counts you will
  realistically collect, run on a Pi with no GPU, and tell you *why* they
  decided, which an audit trail needs. Move to a CNN only past ~20k samples.
- **Do not train:** a bigger YOLO on more RGB gold photos. Scaling n→s→m→l
  on the same data cannot learn a distinction the data does not contain.

## Calibrate the thresholds

Every threshold in `config/physics.json` is a **starting point derived from
the physics and validated on synthetic samples — not fitted to real gold.**
Collect data with `capture_dataset.py`, compare the `verdict` column against
your labels, and tighten them. Treating the defaults as production-ready is
the fastest way to ship a confident wrong answer.

## Tests

```bash
python3 -m pytest tests/ -q          # 30 tests
```

## Known gaps

- **Weight comes from OCR of the scale display.** `OCRReader.read` returns
  the first digit-containing string *anywhere* in the frame — a timestamp
  or price tag qualifies — and the row is then marked `done`. Reading the
  scale over RS-232/USB serial with `pyserial` is far more reliable, harder
  to tamper with, and removes the EasyOCR dependency entirely.
- **No audit trail.** Nothing hashes or signs the image/weight/timestamp
  tuple yet.
- **No hand/glove gate.** The "capture only when no hands are present" rule
  is not implemented; MediaPipe is not wired in.
- **Person segmentation runs every frame** at conf=0.2 to answer a question
  that only matters at trigger time.
- **Illumination is manual.** A lamp swept by hand is not repeatable. Fixed
  LEDs under software control, synchronised to capture, would make the
  measurement reproducible — required for a real audit trail.
- `database/` is orphaned: it writes `runs/jewellery_detections.db` /
  `gold_detections`, while the live path and viewer use `runs/gold.db` /
  `detections`.

## Project Structure

```
Golddetection/
├── GoldNormal.py                  ← Main script (run this)
├── physics/                       ← Material verification stage
│   ├── colour.py                  ← linear radiance, chromaticity, von Kries
│   ├── config.py                  ← config + thresholds (calibrate these)
│   ├── reference.py               ← white/grey patch illuminant estimation
│   ├── specular.py                ← the discriminator + verdicts
│   ├── tracker.py                 ← per-object sweep state machine
│   ├── features.py                ← 41 features for Model B
│   ├── camera.py                  ← USB/RTSP sources + control locking
│   └── pipeline.py                ← one call per frame
├── tools/
│   ├── calibrate_chips.py         ← run this first
│   └── capture_dataset.py         ← collect Model B training data
├── tests/                         ← pytest suite
├── database/                      ← Database management package
│   ├── __init__.py
│   ├── db_manager.py              ← SQLite DB with deduplication
│   ├── image_extractor.py         ← Extracts best gold frame from videos
│   └── post_processor.py          ← Background thread for image extraction
├── viewer/                        ← GUI viewer application
│   ├── app.py                     ← Run this to launch viewer
│   ├── db_reader.py               ← Read-only DB queries
│   ├── file_opener.py             ← Cross-platform file opener
│   └── widgets/                   ← UI components
│       ├── sidebar.py
│       ├── topbar.py
│       ├── table_view.py
│       └── detail_panel.py
├── weights/                       ← All YOLO model files
│   ├── Yolo11n.engine             ← Gold detection (TensorRT)
│   ├── yolo26n-seg.onnx           ← Person segmentation (ONNX)
│   └── ...                        ← Other model formats (.pt, .onnx)
├── runs/
│   ├── recordings/                ← Auto-saved .mp4 clips
│   ├── images/                    ← Snapshots and extracted gold frames
│   ├── jewellery_detections.db    ← SQLite database (auto-created)
│   └── detection.log              ← Event log
├── test_db.py                     ← Database test suite
├── data.yaml                      ← Dataset config (Roboflow)
├── requirements.txt
└── ReadME.md
```

## Models

| Model | Format | Purpose |
|---|---|---|
| `Yolo11n.engine` | TensorRT | Gold/jewellery detection (primary) |
| `yolo26n-seg.onnx` | ONNX | Person segmentation |

All models are stored in the `weights/` directory.

### Detection Classes (from `data.yaml`)

Bangles · Chain · Earrings · Gold Bar · Gold Coin · Ring

## Database

Detection events are stored in `runs/jewellery_detections.db` (SQLite, auto-created on first run).

### Checking the Database

**Command line:**
```bash
sqlite3 -header -column runs/jewellery_detections.db "SELECT * FROM gold_detections;"
```

**GUI viewer:**
```bash
sudo apt install sqlitebrowser
sqlitebrowser runs/jewellery_detections.db
```

**Python:**
```python
from database.db_manager import JewelleryDBManager
db = JewelleryDBManager()
for row in db.get_all_detections():
    print(row)
```

### Running Tests

```bash
python3 test_db.py
# Expected output: ALL TESTS PASSED
```

## Dependencies

- [PyTorch](https://pytorch.org/)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [OpenCV](https://opencv.org/)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [NumPy](https://numpy.org/)
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) (viewer GUI)
- [Pillow](https://pillow.readthedocs.io/) (image thumbnails in viewer)
- SQLite3 (built-in with Python)
