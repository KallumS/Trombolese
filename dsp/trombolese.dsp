declare name "Trombolese";
declare description "Trombone-oboe hybrid with a continuously morphable bore";
declare version "0.1.0";

// ===========================================================================
// STATUS: PARTLY VERIFIED.
//
// A port of the Python reference in src/trombolese/, which remains the
// authority. What has been checked, with faust 2.70.3:
//
//   * every definition type-checks
//   * `valve` reproduces reed.py's _Valve to nine significant figures,
//     compared sample by sample against the Python
//
// What has not:
//
//   * the ladder assembly, which is not written yet (see below)
//   * `flow`, `junction`, `termination` -- these compile, but compiling is
//     not evidence. `valve` compiled too, and was wrong.
//
// That last point is worth the space. The first version of `valve` compiled
// cleanly, ran, and produced plausible output that was 2.3% off after ten
// milliseconds and completely wrong at the attack, because Faust initialises
// feedback state to zero: a state holding the valve's *opening* starts the
// instrument with the valve shut instead of at rest. Carrying the deviation
// from rest instead fixes it, since that really is zero at rest. Nothing but
// a numerical comparison against the reference would have caught it.
//
// So verify in this order, and verify numerically:
//   1. valve alone, constant dp            -> DONE, matches to 9 figures
//   2. flow alone, against test_reed.py's closed-form check
//   3. ladder alone, impulse in            -> peaks match stage 1
//   4. all together                        -> matches scripts/render_audio.py
// ===========================================================================

import("stdfaust.lib");

// ---------------------------------------------------------------------------
// The one architectural problem this port has to solve
// ---------------------------------------------------------------------------
// Faust fixes structure at compile time, but the Trombolese changes length
// while it plays: the slide moves, and under pitch compensation the morph
// itself stretches the bore. The Python reference handles this by re-sectioning
// and resampling the delay lines (Waveguide.adopt); Faust cannot resize `par`.
//
// The port therefore fixes the ladder at NSECTIONS -- enough for the longest
// configuration -- and takes up the difference with one interpolated delay in
// the reflection path, which is also where the radiation end correction and
// the wall-drag dispersion already live. The morph then only ever changes
// coefficients, never structure, which is what Faust wants anyway.
//
// NSECTIONS is a compile-time constant. 640 covers the pitch-compensated bore
// out to alpha = 1 at 48 kHz; recompute it if the geometry or rate changes.

NSECTIONS = 640;
MAXTRIM   = 512;   // samples of length trim the termination delay can absorb

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
// alpha and beta have no precedent on any instrument, so they get no familiar
// mapping; see the README's control-surface section for the intended MPE
// assignment.

pressure   = hslider("[0] breath [unit:Pa]", 0, 0, 9000, 1) : si.smoo;
embouchure = hslider("[1] embouchure [unit:Hz]", 176, 30, 400, 0.1) : si.smoo;
slide      = hslider("[2] slide [unit:m]", 0, 0, 0.6, 0.001) : si.smoo;
alpha      = hslider("[3] bore morph [style:knob]", 0, 0, 1, 0.001) : si.smoo;
beta       = hslider("[4] reed morph [style:knob]", 0, 0, 1, 0.001) : si.smoo;

// ---------------------------------------------------------------------------
// The excitation
// ---------------------------------------------------------------------------
// A single-degree-of-freedom valve. `striking` is the whole morph: +1 blows
// open (brass lips), -1 blows closed (double reed). See reed.py for why the
// embouchure control must stay live at both ends.

striking  = 1.0 - 2.0 * beta;
valveFreq = embouchure * (1.0 + beta * 2.5);      // ratio 1 -> 3.5
valveQ    = 15.0 + beta * (12.0 - 15.0);
valveW    = 0.012 + beta * (0.010 - 0.012);
restOpen  = 3.0e-4;
closingP  = 12000.0 + beta * (7000.0 - 12000.0);

rho = 1.2041;                                      // air density at 20 C
omega = 2.0 * ma.PI * valveFreq;
invMass = (omega * omega * restOpen) / closingP;

// Flow through the opening against the bore's wave impedance. dp and U define
// each other, so the pair is solved exactly as a quadratic rather than
// iterated -- an explicit one-sample-delayed guess goes unstable when blown
// hard. Mirrors Reed.step.
flow(incident, zc, opening) = ma.signum(incident) * root
with {
    area = valveW * max(opening, 0.0);
    k    = 2.0 * area * area / rho;
    disc = (k * zc) ^ 2 + 4.0 * k * abs(incident);
    root = 0.5 * (0.0 - k * zc + sqrt(max(disc, 0.0)));
};

// Valve displacement: y'' + (w/Q) y' + w^2 (y - y0) = sigma dp / mu, by
// explicit Euler, hard-limited at shut and at three times the rest opening.
// The `~` supplies the unit delay that puts the previous sample on the
// right-hand side, matching reed.py's _Valve.advance.
//
// The state is the valve's DEVIATION from its rest opening, not the opening
// itself, because Faust initialises feedback to zero: a state holding the
// opening would start the instrument with the valve shut rather than at rest.
// Verified against the Python to nine significant figures.
valve(dp) = ((equation ~ si.bus(2)) : !, _) : +(restOpen)
with {
    dt = 1.0 / ma.SR;
    equation(v, d) = vNew, dNew
    with {
        accel = striking * dp * invMass - (omega / valveQ) * v - omega * omega * d;
        vNew = v + accel * dt;
        dNew = max(0.0 - restOpen, min(2.0 * restOpen, d + vNew * dt));
    };
};

// ---------------------------------------------------------------------------
// One ladder section
// ---------------------------------------------------------------------------
// Kelly-Lochbaum junction in its one-multiply form, with the per-section wall
// loss and the one-pole that carries the wall drag's phase delay. `r` comes
// from the areas either side: r = (S1 - S2) / (S1 + S2).
//
// junction(r) : (pForward, pBackward) -> (pForward', pBackward')

junction(r, fwd, bwd) = fwd + d, bwd + d with { d = r * (fwd - bwd); };

// Per-section loss and dispersion, as computed by section_bore().
lossy(g, x)      = x * g;
dispersive(p, x) = x : si.smooth(p);

// ---------------------------------------------------------------------------
// Termination
// ---------------------------------------------------------------------------
// Radiation from the bell mouth: a one-pole fitted to |(Zr - Zc)/(Zr + Zc)|,
// preceded by a delay carrying three things at once --
//   * the radiation end correction, 2 * 0.6133 * a (omitted, everything is
//     sharp by tens of cents),
//   * the wall-drag dispersion not already distributed along the ladder,
//   * the length trim standing in for Faust's fixed NSECTIONS.
// Coefficients come from _fit_mouth_reflection(); recompute if the bell moves.

MOUTHGAIN = 0.9477;
MOUTHPOLE = 0.9200;

endCorrection = 18.5;                              // samples, 2 * 0.6133 * a
lengthTrim    = slide * ma.SR / 343.4 * 2.0;       // out and back

termination = de.fdelay(MAXTRIM, endCorrection + lengthTrim)
            : si.smooth(MOUTHPOLE)
            : *(0.0 - MOUTHGAIN);

// ---------------------------------------------------------------------------
// Assembly -- THE PART STILL TO WRITE
// ---------------------------------------------------------------------------
// The remaining work is routing NSECTIONS junctions into a single feedback
// loop. In Faust that is a `~` around a `par`/`seq` composition with the two
// travelling waves interleaved, and the coefficient vectors read from
// rdtable(s) indexed by section, re-filled at control rate from alpha.
//
// Two things to get right, both of which the Python reference already pins
// down and both of which are easy to get subtly wrong here:
//   * the loop's total delay must come to 2 * NSECTIONS samples, not
//     2 * NSECTIONS - 1; test_uniform_ladder_matches_the_textbook is the
//     check, and it is worth porting that test before this code.
//   * the dispersion one-poles need separate state per direction of travel.
//
// Until that exists this file is a set of verified-by-eye primitives, not an
// instrument. Do not ship it as one.

process = _;
