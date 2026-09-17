"""
Tests for the material-discrimination stage.

The important one is `test_flat_lit_plastic_is_not_gold`. Diffuse body
reflection holds its chromaticity across shading, so under flat light a
yellow dielectric looks exactly like a metal to this statistic. The method
is only valid once a real specular excursion has been driven onto the
piece. If that guard regresses, the system reports confident false gold --
which at a pawn counter means a loan against plastic.
"""

import numpy as np
import pytest

from physics import colour
from physics.camera import CameraControl, find_exposure_without_clipping
from physics.config import CameraConfig, ChipConfig, PhysicsConfig, Thresholds
from physics.reference import ReferenceChips
from physics.specular import SweepAccumulator, Verdict
from physics.tracker import Detection, MaterialTracker

GOLD = np.array([0.95, 0.65, 0.25], dtype=np.float32)
YELLOW_BODY = np.array([0.80, 0.70, 0.15], dtype=np.float32)
NEUTRAL = np.array([1.0, 1.0, 1.0], dtype=np.float32)


def _thresholds(**kw):
    base = dict(min_samples=800, baseline_frames=6, min_sweep_frames=5, timeout_frames=60)
    base.update(kw)
    return Thresholds(**base)


def _sweep(kind, *, lamp=True, frames=90, clip=False, seed=3):
    """Simulate a sweep. `kind` is 'metal' or 'dielectric'."""
    t = _thresholds()
    tracker = MaterialTracker(t)
    rng = np.random.default_rng(seed)
    from physics.reference import ChipReading
    chips = ChipReading(NEUTRAL.copy(), True, True)
    n = 48
    for f in range(frames):
        drive = (0.15 + 0.85 * np.sin(np.pi * max(f - 10, 0) / max(frames - 10, 1)) ** 2) if lamp else 0.3
        shade = rng.uniform(0.2, 1.0, (n, n, 1)).astype(np.float32)
        if kind == "metal":
            px = shade * drive * GOLD
        else:
            px = 0.25 * YELLOW_BODY + shade * drive * 0.9
        px = np.clip(px, 0, None)
        mask = np.zeros((n, n), bool)
        mask[8:40, 8:40] = True
        clipmap = np.zeros((n, n), bool)
        if clip:
            clipmap[8:40, 8:40] = shade[8:40, 8:40, 0] > 0.5
        tracker.update(px, clipmap, [Detection((8, 8, 40, 40), mask, 0.9, "Ring")], chips)
    return tracker


# -- the physics ---------------------------------------------------------
def test_metal_chromaticity_is_invariant_to_brightness():
    """No diffuse term means every pixel is a scalar multiple of one vector."""
    ramp = np.linspace(0.05, 1.0, 64)[:, None] * GOLD
    chroma = colour.chroma_from_rg(colour.chromaticity(ramp))
    assert float(chroma.max() - chroma.min()) < 1e-5


def test_dielectric_chromaticity_collapses_toward_illuminant():
    """The specular term carries the illuminant's colour, so highlights whiten."""
    ms = np.linspace(0.0, 1.5, 64)[:, None]
    px = 0.3 * YELLOW_BODY + ms * NEUTRAL
    chroma = colour.chroma_from_rg(colour.chromaticity(px))
    assert chroma[0] > 0.15
    assert chroma[-1] < 0.03


def test_srgb_linearisation_roundtrips():
    x = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    assert np.allclose(colour.linear_to_srgb(colour.srgb_to_linear(x)), x, atol=1e-5)


# -- verdicts ------------------------------------------------------------
def test_swept_metal_reads_gold_like():
    v = _sweep("metal", lamp=True).best_verdict()
    assert v.state is Verdict.GOLD_LIKE
    assert v.chroma_ratio > 0.8


def test_swept_yellow_dielectric_is_rejected():
    v = _sweep("dielectric", lamp=True).best_verdict()
    assert v.state is Verdict.DIELECTRIC
    assert v.chroma_ratio < 0.6


def test_flat_lit_plastic_is_not_gold():
    """REGRESSION GUARD -- see module docstring.

    Without a specular excursion this must refuse to rule, not guess.
    A yellow dielectric under flat light presents constant chroma across
    its shading, identical in this statistic to a metal.
    """
    v = _sweep("dielectric", lamp=False).best_verdict()
    assert v.state is Verdict.INVALID_NO_SWEEP
    assert v.state is not Verdict.GOLD_LIKE


def test_flat_lit_metal_also_refuses_rather_than_guessing():
    v = _sweep("metal", lamp=False).best_verdict()
    assert v.state is Verdict.INVALID_NO_SWEEP


def test_decide_fails_closed_when_caller_omits_sweep_evidence():
    """A forgetful caller must get a refusal, never a false confirmation."""
    assert SweepAccumulator(_thresholds()).decide().state is Verdict.INVALID_NO_SWEEP


def test_clipped_highlights_void_the_measurement():
    """Saturated pixels drive chroma to zero for every material alike."""
    v = _sweep("metal", lamp=True, clip=True).best_verdict()
    assert v.state is Verdict.INVALID_CLIPPED


def test_hue_window_alone_would_not_have_saved_us():
    """The simulated yellow dielectric sits inside any plausible gold hue
    window -- which is exactly why the old HSV gate could not work."""
    v = _sweep("dielectric", lamp=True).best_verdict()
    t = Thresholds()
    assert t.gold_hue_min <= v.hue_deg <= t.gold_hue_max


# -- tracker -------------------------------------------------------------
def test_tracker_associates_one_object_across_frames():
    tracker = _sweep("metal", lamp=True)
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].frames > 10


def test_tracker_assigns_distinct_ids_to_separated_objects():
    t = _thresholds()
    tracker = MaterialTracker(t)
    from physics.reference import ChipReading
    chips = ChipReading(NEUTRAL.copy(), True, True)
    px = np.full((64, 64, 3), 0.4, np.float32)
    clip = np.zeros((64, 64), bool)
    m1 = np.zeros((64, 64), bool); m1[2:20, 2:20] = True
    m2 = np.zeros((64, 64), bool); m2[40:60, 40:60] = True
    tracker.update(px, clip, [Detection((2, 2, 20, 20), m1), Detection((40, 40, 60, 60), m2)], chips)
    assert {t_.track_id for t_ in tracker.tracks} == {1, 2}


# -- camera --------------------------------------------------------------
def test_stream_source_is_detected():
    assert CameraConfig(source="rtsp://10.0.0.5:554/Streaming/Channels/101").is_stream
    assert not CameraConfig(source=0).is_stream


def test_stream_control_lock_reports_failure_rather_than_pretending():
    rep = CameraControl(CameraConfig(source="rtsp://host/stream")).lock()
    assert not rep.ok
    assert any("RTSP" in w for w in rep.warnings)


def test_usb_source_resolves_device_node():
    assert CameraControl(CameraConfig(source=2)).device == "/dev/video2"


def test_exposure_search_picks_first_unclipped_candidate():
    frames = {
        400: np.full((8, 8, 3), 255, np.uint8),
        200: np.full((8, 8, 3), 252, np.uint8),
        100: np.full((8, 8, 3), 180, np.uint8),
    }
    state = {"e": 400}
    chosen = find_exposure_without_clipping(
        lambda: frames[state["e"]],
        lambda v: (state.__setitem__("e", v), True)[1],
        [400, 200, 100], settle_frames=1,
    )
    assert chosen == 100


# -- reference chips -----------------------------------------------------
def test_uncalibrated_chips_degrade_gracefully():
    r = ReferenceChips(ChipConfig()).measure(np.full((64, 64, 3), 128, np.uint8))
    assert not r.calibrated
    assert np.allclose(r.illuminant_rgb, NEUTRAL)


def test_white_patch_estimates_a_warm_illuminant():
    frame = np.zeros((64, 64, 3), np.uint8)
    frame[:, :] = (180, 200, 230)  # BGR -> warm white
    r = ReferenceChips(ChipConfig(white_rect=(0.1, 0.1, 0.5, 0.5))).measure(frame)
    assert r.white_ok
    assert r.illuminant_rgb[0] > r.illuminant_rgb[2]  # R > B


def test_clipped_white_patch_is_rejected():
    frame = np.full((64, 64, 3), 255, np.uint8)
    r = ReferenceChips(ChipConfig(white_rect=(0.1, 0.1, 0.5, 0.5))).measure(frame)
    assert not r.white_ok
    assert "clipping" in r.note


# -- config --------------------------------------------------------------
def test_config_roundtrips(tmp_path):
    cfg = PhysicsConfig()
    cfg.camera.source = "rtsp://10.0.0.5:554/s"
    cfg.chips.white_rect = (0.1, 0.2, 0.05, 0.05)
    path = cfg.save(tmp_path / "physics.json")
    back = PhysicsConfig.load(path)
    assert back.camera.source == cfg.camera.source
    assert tuple(back.chips.white_rect) == cfg.chips.white_rect


def test_missing_config_returns_defaults(tmp_path):
    assert PhysicsConfig.load(tmp_path / "nope.json").camera.source == 0
