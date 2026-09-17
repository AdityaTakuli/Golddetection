"""
Colour science primitives for the material-discrimination stage.

Everything downstream assumes *linear* radiance, not sRGB code values.
The dichromatic reflection model (Shafer) is linear in radiance:

    I = m_d * D * E   +   m_s * S * E          (dielectric)
    I =                   m_s * M * E          (metal)

where D is the body reflectance, S the interface reflectance (spectrally
flat for a dielectric, so S*E has the illuminant's colour) and M the
metal's own spectrally selective reflectance.

The consequence we exploit:

  * dielectric -- as a pixel gets brighter it is increasingly dominated by
    the m_s term, so its chromaticity migrates toward the *illuminant's*
    chromaticity. Yellow plastic whitens in its highlight.
  * metal -- there is no m_d term, so every pixel is a scalar multiple of
    the same spectral vector. Chromaticity is *invariant* to brightness.
    Gold stays gold in its highlight.

So the discriminator is not "how yellow is it" but "does its chromaticity
move toward white as it brightens". That is what `chroma` measures here,
once the illuminant has been divided out (see physics.reference).

Channel order is RGB throughout. OpenCV hands us BGR; convert at the edge
with `bgr_u8_to_linear_rgb` and never think about it again.
"""

from __future__ import annotations

import numpy as np

# Equal-energy point in the 2D chromaticity plane. After von Kries
# normalisation by the white reference, the illuminant sits here, so
# distance from this point is "how far from the light's own colour".
NEUTRAL = 1.0 / 3.0

# An 8-bit channel at or above this is assumed to be on the sensor's
# shoulder and carrying no reliable colour information. 250 rather than
# 255 because most ISPs soft-knee the top few codes.
DEFAULT_CLIP_LEVEL = 250

# Rec.709 linear luminance weights.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

_EPS = 1e-6


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Invert the sRGB transfer function. Input/output in [0, 1] float."""
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Forward sRGB transfer function, for display only."""
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055).astype(np.float32)


def bgr_u8_to_linear_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """OpenCV BGR uint8 frame -> float32 linear RGB in [0, 1].

    This is an *approximate* linearisation: it assumes the camera emits
    sRGB-encoded values. Real ISPs apply their own tone curve, so for
    quantitative work you should replace this with a measured response
    curve (see tools/calibrate_chips.py --measure-response). It is still
    dramatically better than treating code values as radiance.
    """
    rgb = np.asarray(frame_bgr, dtype=np.uint8)[..., ::-1]
    return srgb_to_linear(rgb.astype(np.float32) / 255.0)


def clipped_mask(frame_bgr: np.ndarray, clip_level: int = DEFAULT_CLIP_LEVEL) -> np.ndarray:
    """Boolean map of pixels where ANY channel is on the sensor shoulder.

    Must be computed on the original 8-bit data, before linearisation --
    once a channel saturates, chroma collapses toward zero for *every*
    material and gold becomes indistinguishable from plastic. Those pixels
    carry no signal and have to be excluded, not modelled.
    """
    return np.any(np.asarray(frame_bgr, dtype=np.uint8) >= clip_level, axis=-1)


def luminance(rgb_lin: np.ndarray) -> np.ndarray:
    """Rec.709 relative luminance of linear RGB."""
    return np.tensordot(np.asarray(rgb_lin, dtype=np.float32), _LUMA, axes=([-1], [0]))


def chromaticity(rgb_lin: np.ndarray) -> np.ndarray:
    """Linear RGB -> 2D chromaticity (r, g), each = channel / (R+G+B).

    Intensity is divided out, so a metal -- whose pixels are all scalar
    multiples of one spectral vector -- collapses to a single point here
    no matter how hard you light it.
    """
    rgb = np.asarray(rgb_lin, dtype=np.float32)
    total = np.sum(rgb, axis=-1, keepdims=True)
    return (rgb[..., :2] / np.maximum(total, _EPS)).astype(np.float32)


def chroma_from_rg(rg: np.ndarray) -> np.ndarray:
    """Distance from the neutral point in the chromaticity plane.

    With the illuminant normalised out this is "how far this pixel's colour
    is from the colour of the light". Dielectric highlights drive it to
    zero; metal highlights hold it constant.
    """
    d = np.asarray(rg, dtype=np.float32) - NEUTRAL
    return np.hypot(d[..., 0], d[..., 1]).astype(np.float32)


def hue_angle_from_rg(rg: np.ndarray) -> np.ndarray:
    """Hue angle in degrees within the chromaticity plane, [-180, 180].

    Roughly: ~0 deg points red, ~45 deg yellow, ~90 deg green. Gold sits in
    the warm wedge between. Used only to separate a *warm* metal from a
    neutral one (silver, steel) -- never to make the metal/dielectric call,
    because yellow plastic is warm too.
    """
    d = np.asarray(rg, dtype=np.float32) - NEUTRAL
    return np.degrees(np.arctan2(d[..., 1], d[..., 0])).astype(np.float32)


def von_kries(rgb_lin: np.ndarray, illuminant_rgb: np.ndarray) -> np.ndarray:
    """Divide out the illuminant, channel-wise.

    After this a spectrally flat surface reads neutral regardless of the
    lamp's colour temperature, so chroma thresholds stop drifting when the
    bulb ages or the ambient light changes.
    """
    e = np.asarray(illuminant_rgb, dtype=np.float32)
    e = np.maximum(e, _EPS)
    return (np.asarray(rgb_lin, dtype=np.float32) / e).astype(np.float32)


def angular_distance_deg(a: float, b: float) -> float:
    """Smallest signed-magnitude angle between two degree values."""
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)
