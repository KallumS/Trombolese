"""One-dimensional duct acoustics: losses, radiation, and transfer matrices.

The bore is treated as a chain of cylindrical and conical frusta. Each is
described by a 2x2 transfer matrix relating pressure and volume flow at its
input to those at its output:

    [p_in ]   [A  B] [p_out]
    [U_in ] = [C  D] [U_out]

Chaining segments is then a matrix product, and the input impedance of the
whole bore loaded by a radiation impedance Z_L is

    Z_in = (A Z_L + B) / (C Z_L + D)

Every routine here is vectorised over frequency: pass an array of frequencies
and you get back arrays of shape (n_freq, 2, 2) or (n_freq,).

Modelling approximations
------------------------
Three approximations are baked in. They are all standard, but they bound how
far the numbers here should be trusted:

1. **Plane/spherical wave propagation only.** No higher-order transverse
   modes. Valid below the first cross-mode cutoff, roughly 1.84 c / (2 pi a);
   for a 7 mm bore radius that is about 14 kHz, so the assumption is safe over
   the whole range we care about, but it degrades inside the bell flare where
   `a` is large.
2. **Kirchhoff wide-tube boundary-layer losses.** Viscous and thermal losses
   are folded into a complex wavenumber, valid when the tube radius is much
   larger than the boundary-layer thickness (true above ~50 Hz for these
   bores). The small associated correction to the characteristic impedance is
   neglected, which is second order.
3. **No tone holes, no wall vibration, no nonlinear (brassy) propagation.**
   At trombone playing dynamics, nonlinear steepening genuinely matters for
   timbre. It does not much move the resonance *frequencies*, which is what
   this stage is for, so it is deferred to the waveguide implementation.
"""

from __future__ import annotations

import numpy as np

from .constants import AIR_20C, Air

__all__ = [
    "wavenumber",
    "radiation_impedance",
    "cylinder_matrix",
    "cone_matrix",
    "invert",
    "chain",
    "input_impedance",
]

# Below this relative change in radius across a segment, the cone's apex is so
# far away that the spherical-wave formulation loses precision (and can
# overflow). Such a segment is indistinguishable from a cylinder anyway.
_CONE_FLATNESS_TOL = 1e-9


def wavenumber(
    freqs: np.ndarray, radius: float | np.ndarray, air: Air = AIR_20C
) -> np.ndarray:
    """Complex wavenumber including viscothermal boundary-layer losses.

    Uses the Kirchhoff wide-tube approximation, in which the attenuation
    coefficient and the phase-velocity correction are equal in magnitude:

        alpha = (1 / (a c)) sqrt(omega / 2) [ sqrt(nu) + (gamma - 1) sqrt(nu / Pr) ]
        k     = (omega / c + alpha) - j alpha

    The sign convention is exp(-jkx) for a wave travelling in +x, so a
    negative imaginary part means decay. Losses scale as 1/a, which is why the
    narrow throat of the instrument contributes far more damping per metre
    than the bell does.

    Parameters
    ----------
    freqs:
        Frequencies in Hz. Must be strictly positive.
    radius:
        Duct radius in metres. Broadcasts against ``freqs``.
    air:
        Air properties.
    """
    freqs = np.asarray(freqs, dtype=float)
    if np.any(freqs <= 0.0):
        raise ValueError("frequencies must be strictly positive")

    omega = 2.0 * np.pi * freqs
    c = air.speed_of_sound
    nu = air.kinematic_viscosity

    boundary_layer = np.sqrt(nu) + (air.heat_capacity_ratio - 1.0) * np.sqrt(
        nu / air.prandtl_number
    )
    alpha = np.sqrt(omega / 2.0) * boundary_layer / (np.asarray(radius) * c)

    return (omega / c + alpha) - 1j * alpha


def radiation_impedance(
    freqs: np.ndarray, radius: float, air: Air = AIR_20C
) -> np.ndarray:
    """Radiation impedance of an unflanged open end, in acoustic ohms.

    Built from the Levine-Schwinger low-frequency limit for an unflanged pipe,

        resistance  x = (ka)^2 / 4        reactance  y = 0.6133 ka

    (a radiation resistance rising as frequency squared, and an end correction
    of 0.6133 a), extended across the band as

        z = (x + j y) / (1 + x)

    which reduces to ``x + jy`` when ka << 1 and tends to 1 when ka >> 1 -- at
    high frequency the mouth stops reflecting and the bore looks like an
    infinite pipe. ``Re(z) = x / (1 + x)`` is positive everywhere, so the
    termination stays passive and cannot inject energy.

    Dividing by ``1 + x`` rather than by ``1 + x + jy`` matters. The latter is
    the more obvious Pade form, but at low ka the reactance dominates the
    resistance (``y^2 / x = 1.5``), so it leaks ``y^2`` into the real part and
    inflates the radiation resistance roughly 2.5-fold -- which would show up
    as badly over-damped pedal notes. Dividing by a purely real quantity leaves
    the low-frequency resistance and reactance both exact.

    This remains an engineering fit rather than the exact Levine-Schwinger
    solution: right in the low-frequency limit, right in the high-frequency
    limit, and qualitatively right between them, which is what
    resonance-frequency work needs. A waveguide implementation will want a
    proper reflection-function fit instead.
    """
    freqs = np.asarray(freqs, dtype=float)
    area = np.pi * radius**2
    z_char = air.density * air.speed_of_sound / area

    ka = 2.0 * np.pi * freqs * radius / air.speed_of_sound
    resistance = 0.25 * ka**2
    reactance = 0.6133 * ka

    return z_char * (resistance + 1j * reactance) / (1.0 + resistance)


def cylinder_matrix(
    freqs: np.ndarray, radius: float, length: float, air: Air = AIR_20C
) -> np.ndarray:
    """Transfer matrix of a cylindrical segment, shape (n_freq, 2, 2).

    The classic plane-wave result, with the characteristic impedance written
    as ``Zc = rho omega / (S k)`` so that it stays consistent with the complex
    wavenumber returned by :func:`wavenumber` (it reduces to ``rho c / S`` in
    the lossless limit).
    """
    freqs = np.asarray(freqs, dtype=float)
    k = wavenumber(freqs, radius, air)
    area = np.pi * radius**2
    z_char = air.density * 2.0 * np.pi * freqs / (area * k)

    kl = k * length
    cos_kl = np.cos(kl)
    sin_kl = np.sin(kl)

    matrix = np.empty(freqs.shape + (2, 2), dtype=complex)
    matrix[..., 0, 0] = cos_kl
    matrix[..., 0, 1] = 1j * z_char * sin_kl
    matrix[..., 1, 0] = 1j * sin_kl / z_char
    matrix[..., 1, 1] = cos_kl
    return matrix


def cone_matrix(
    freqs: np.ndarray,
    radius_in: float,
    radius_out: float,
    length: float,
    air: Air = AIR_20C,
) -> np.ndarray:
    """Transfer matrix of a conical frustum, shape (n_freq, 2, 2).

    Rather than transcribing the closed-form conical matrix, this builds the
    two independent spherical-wave solutions and composes them. Inside a cone
    of half-angle theta with its apex at the origin, where ``S(x) = Omega x^2``:

        p(x) = (A e^{-jkx} + B e^{+jkx}) / x
        U(x) = (Omega / (rho omega)) [ A e^{-jkx} (kx - j) - B e^{+jkx} (kx + j) ]

    Collecting these into a basis matrix ``V(x)`` mapping ``[A, B]`` to
    ``[p, U]`` gives the segment matrix directly as ``V(x1) V(x2)^-1``. The two
    signed distances from the apex, ``x1`` and ``x2``, are set by the radii;
    they are negative for a contracting cone, which the formulation handles
    without special-casing (the apex simply lies downstream).

    The exponentials are referenced to ``x1`` rather than to the apex. This is
    a change of basis that cancels in ``V(x1) V(x2)^-1``, but it keeps the
    exponent bounded by ``|k| L`` instead of ``|k| x``, which is what stops a
    gently tapered segment -- whose apex may be kilometres away -- from
    overflowing.

    A segment whose radius barely changes is delegated to
    :func:`cylinder_matrix`, both for speed and because the spherical-wave
    form becomes ill-conditioned as the apex recedes to infinity.
    """
    freqs = np.asarray(freqs, dtype=float)

    if length <= 0.0:
        raise ValueError("segment length must be positive")
    if radius_in <= 0.0 or radius_out <= 0.0:
        raise ValueError("segment radii must be positive")

    mean_radius = 0.5 * (radius_in + radius_out)
    if abs(radius_out - radius_in) / mean_radius < _CONE_FLATNESS_TOL:
        return cylinder_matrix(freqs, mean_radius, length, air)

    # Signed distances from the apex to each end.
    x1 = radius_in * length / (radius_out - radius_in)
    x2 = x1 + length

    # Losses are set by the local radius; use the segment mean.
    k = wavenumber(freqs, mean_radius, air)
    omega = 2.0 * np.pi * freqs

    # S(x) = Omega x^2, evaluated at the input end.
    solid_angle = np.pi * radius_in**2 / x1**2
    gain = solid_angle / (air.density * omega)

    def basis(x: float, xi: float) -> np.ndarray:
        """Basis matrix V(x) with exponentials referenced to x1 (xi = x - x1)."""
        fwd = np.exp(-1j * k * xi)
        bwd = np.exp(1j * k * xi)
        out = np.empty(freqs.shape + (2, 2), dtype=complex)
        out[..., 0, 0] = fwd / x
        out[..., 0, 1] = bwd / x
        out[..., 1, 0] = gain * (k * x - 1j) * fwd
        out[..., 1, 1] = -gain * (k * x + 1j) * bwd
        return out

    return basis(x1, 0.0) @ invert(basis(x2, length))


def invert(matrix: np.ndarray) -> np.ndarray:
    """Invert a stack of 2x2 matrices by the closed-form adjugate.

    ``np.linalg.inv`` dispatches to LAPACK per matrix, which for a stack of
    tens of thousands of 2x2s costs roughly seven times what the two-line
    analytic formula does. Since a bore is a few hundred segments evaluated
    over a dense frequency grid, that difference dominates the whole model.
    """
    a = matrix[..., 0, 0]
    b = matrix[..., 0, 1]
    c = matrix[..., 1, 0]
    d = matrix[..., 1, 1]
    det = a * d - b * c

    out = np.empty_like(matrix)
    out[..., 0, 0] = d / det
    out[..., 0, 1] = -b / det
    out[..., 1, 0] = -c / det
    out[..., 1, 1] = a / det
    return out


def chain(matrices: list[np.ndarray]) -> np.ndarray:
    """Multiply a list of segment matrices, input end first."""
    if not matrices:
        raise ValueError("need at least one segment")
    total = matrices[0]
    for matrix in matrices[1:]:
        total = total @ matrix
    return total


def input_impedance(matrix: np.ndarray, load: np.ndarray) -> np.ndarray:
    """Input impedance of a bore described by ``matrix`` and terminated in ``load``.

    Pass ``load = 0`` for an ideal pressure-release open end (useful for
    testing against textbook results), or the output of
    :func:`radiation_impedance` for a realistic one.
    """
    a = matrix[..., 0, 0]
    b = matrix[..., 0, 1]
    c = matrix[..., 1, 0]
    d = matrix[..., 1, 1]
    return (a * load + b) / (c * load + d)
