# Session log — design and build of the Trombolese

A full record of the session that took this project from an empty repository to
a playable instrument. `CLAUDE.md` is the distillation, kept short so it gets
read; this is the archive, kept complete so nothing has to be re-measured.

Everything below was measured in this repository unless marked otherwise.
Where a number appears, it came out of a run, not an estimate.

**Contents**

1. [Format decision](#1-format-decision)
2. [Stage 1 — bore acoustics](#2-stage-1--bore-acoustics)
3. [Pitch neutrality](#3-pitch-neutrality)
4. [Stage 2 — the waveguide](#4-stage-2--the-waveguide)
5. [Stage 2 — the excitation](#5-stage-2--the-excitation)
6. [Conical-end register control](#6-conical-end-register-control)
7. [The reed morph compensation](#7-the-reed-morph-compensation)
8. [Faust toolchain](#8-faust-toolchain)
9. [Dead ends and disproved hypotheses](#9-dead-ends-and-disproved-hypotheses)
10. [Final parameter set](#10-final-parameter-set)
11. [Commit history](#11-commit-history)

---

## 1. Format decision

The opening question was what format the instrument should take: plugin, Reaper
script, or orchestral library.

**Recommendation: a plugin, prototyped in Faust, shipped as CLAP.** Reasoning
as given at the time:

- **Sample library — eliminated on principle.** A sampled library is static
  snapshots crossfaded together; physical modelling's value is continuous
  state. The things that make a Trombolese interesting — a slide portamento
  while the reed crescendos, multiphonics from an unstable reed — are exactly
  what sampling cannot hold. And you cannot record source material for an
  instrument that does not exist.
- **Reaper script — wrong layer.** ReaScript is a control/automation layer and
  cannot run per-sample DSP. JSFX *can* (per-sample EEL2, JIT-compiled) and is
  an excellent scratchpad, but it is Reaper-only, has no real GUI, no SIMD, and
  no distributable installer.
- **Plugin — correct.** Faust for the DSP core, because `physmodels.lib` ships
  waveguide and reed primitives and one source compiles to JSFX, CLAP, VST3 and
  WebAssembly. CLAP first for its per-note modulation.

The user chose this stack and dropped VST3/AU until CLAP is ready.

The instrument's design space was identified at the same time: a trombone is a
**lip reed driving a cylindrical bore**, an oboe a **double reed driving a
conical bore**, and those two bore geometries produce fundamentally different
partial series. Making the bore a continuous control became the project's
defining idea.

---

## 2. Stage 1 — bore acoustics

An offline NumPy model: the bore as a chain of cylindrical and conical frusta,
each carrying a 2×2 transfer matrix in pressure and volume flow, with
viscothermal losses in a complex wavenumber and a radiation impedance at the
mouth. Peaks of `|Z_in|` are the playing regimes, because both a lip reed and a
double reed are pressure-controlled valves behaving close to a flow source.

### Validation against closed-form results

| check | result |
|---|---|
| near-cylindrical cone vs true cylinder | max relative error 5.1e-11 |
| `det(M)` for cone (reciprocity) | deviation 4.8e-15 |
| `det(M)` for cylinder | deviation 1.1e-15 |
| cylinder, ideal open end | 83.7, 253.8, 424.4, 595.3, 766.2 Hz → ratios 1, 3.03, 5.07, 7.11, 9.15 |
| cone, ideal open end | 168.1, 336.4, 504.7, 673.1, 841.6 Hz → ratios 1, 2.001, 3.002, 4.004, 5.006 |

The cylinder's ~2.4% flattening against the lossless textbook value
(`c/4L = 85.8` Hz vs 83.7 measured) is the boundary-layer phase-velocity
correction, confirmed analytically: α/k₀ = 2.48% at that frequency. This
number mattered enormously later.

### Design choice: the conical matrix

Rather than transcribing a closed-form conical transfer matrix from memory, the
two independent spherical-wave solutions were collected into a basis matrix
`V(x)` and the segment matrix taken as `V(x₁)V(x₂)⁻¹`. This is harder to get
wrong, handles contracting cones with no special case, and is checked against
the cylindrical limit in the tests.

Exponentials are referenced to the segment's own start rather than to the apex.
This is a change of basis that cancels in the product, but it bounds the
exponent by `|k|L` instead of `|k|x`, which is what stops a gently tapered
segment — apex kilometres away — from overflowing.

### The morph parameterisation

Blending a constant radius profile with a linear one gives another linear
profile, so the morphed bore is **exactly one conical frustum at any alpha**.
The morph simply walks the throat-end radius inward. No discretisation.

### Two geometry findings

**(a) A long cylindrical section at the throat destroys conical behaviour.**

The first version kept a trombone's 0.55 m cylindrical slide ahead of the
taper. Result — upper resonances ignored the morph almost entirely:

| config | α=0 | α=0.5 | α=1 |
|---|---|---|---|
| slide+body, ideal open end | 38.5, 117.2, 196.4, 275.6 | 46.5, 116.9, 195.9, 276.0 | 62.9, 116.4, 195.5, 278.0 |

Only the fundamental moved. A truncated cone behaves like a complete one only
if whatever replaces its missing apex has roughly that apex's volume; 0.55 m of
narrow tube has several times too much (6.9 ml against 2.7 ml) and acts as a
transmission line, not a compliance. The slide became part of the taper.

**(b) A cone cannot reach the bell at a trombone's bore radius.**

Single frustum over 2.15 m, ideal open end:

| throat | bell entry | truncation | ratios |
|---|---|---|---|
| 6.95 mm | 6.95 mm | — | 1, 3.05, 5.11, 7.17 |
| 2.00 mm | 6.95 mm | 0.288 | 1, 2.21, 3.51, 4.86 |
| 2.00 mm | 10.0 mm | 0.200 | 1, 2.11, 3.29, 4.50 |
| 2.00 mm | 15.0 mm | 0.133 | 1, 2.05, 3.14, 4.26 |
| 2.00 mm | 22.0 mm | 0.091 | 1, 2.02, 3.07, 4.13 |

So the conical limit must open to ~15 mm — nearer a euphonium than a trombone.
The bell *mouth* was held fixed at 108 mm so the radiating aperture does not
confound the sweep; only the bell's entry follows the bore.

### Result

| alpha | throat | truncation | f₁ | f₂/f₁ | ratios |
|---|---|---|---|---|---|
| 0.00 | 6.95 mm | — | 35.9 | 3.044 | 1, 3.04, 5.08, 7.05 |
| 0.25 | 5.71 mm | 0.637 | 43.0 | 2.588 | 1, 2.59, 4.24, 5.82 |
| 0.50 | 4.47 mm | 0.408 | 50.2 | 2.290 | 1, 2.29, 3.66, 4.94 |
| 0.75 | 3.24 mm | 0.249 | 57.4 | 2.108 | 1, 2.11, 3.26, 4.30 |
| 1.00 | 2.00 mm | 0.133 | 63.9 | 2.023 | 1, 2.02, 3.03, 3.87 |

Smooth, monotonic, no discontinuities.

### Performance work

| change | effect |
|---|---|
| analytic 2×2 inverse instead of `np.linalg.inv` | 7× faster on that op |
| bell slices 160 → 48 | converged to <0.1 cent; 3.3× fewer segments |
| frequency grid 80k → 20k | resonances agree to **0.002 cents** (parabolic interpolation) |

Single response: 8.43 s → 1.63 s. A 41-point sweep: ~345 s → 4.87 s.

The grid-density result is worth keeping: peak positions are refined by
parabolic interpolation, so a 10,000-point grid matches an 80,000-point one to
0.002 cents. Do not pay for dense frequency grids.

---

## 3. Pitch neutrality

Uncompensated, the morph lifts the pitch by most of an octave, because a cone
sounds `c/2L` where a cylinder of the same length sounds `c/4L`.

`compensate_pitch` bisects for the bore length that puts a chosen resonance on
a target at each morph position. Holding **f₁** gave 0.00 cents across the whole
sweep, at the cost of the instrument growing ×1.814 (2.80 → 4.55 m overall).

**Compensation cannot disturb harmonicity**, and provably so: the cone's
truncation ratio works out to be simply `r_in / r_out`, with the length
cancelling entirely. Verified numerically (0.1333 both ways). Lengthening moves
every resonance together.

### The correction: only one regime can be held

The first rendered glissando *jumped between partials*. The framing was wrong,
structurally: the morph exists to change the **ratios** between regimes, so
holding all of them still is a contradiction. Pin f₁ and regime 2 still slides
from 3.04×f₁ to 2.05×f₁ — over an octave.

`compensate_pitch(regime=N)` therefore takes the regime actually played.
Holding the third:

| alpha | bore length | f₁ | f₂ | **f₃** |
|---|---|---|---|---|
| 0.00 | 2.150 m | 35.85 | 109.14 | **182.07** |
| 0.25 | 2.155 m | 42.95 | 111.17 | **182.07** |
| 0.50 | 2.173 m | 49.73 | 113.94 | **182.07** |
| 0.75 | 2.218 m | 55.67 | 117.46 | **182.07** |
| 1.00 | 2.312 m | 59.64 | 120.89 | **182.07** |

7.5% extra length instead of 81%. Rendered, the bore glissando then held
194–198 Hz across the entire morph (~35 cents, much of it FFT bin resolution)
while the spectrum transformed.

---

## 4. Stage 2 — the waveguide

A Kelly–Lochbaum scattering ladder: the bore sliced into sections one sample of
propagation long (7.15 mm at 48 kHz), pressure waves travelling both ways,
scattering wherever cross-section changes. Junction in one-multiply form:

```
r = (S1 - S2) / (S1 + S2)
d = r * (p+ - p-)
p+ out = p+ + d ;  p- out = p- + d
```

Chosen over the usual single-delay-loop-plus-bell-filter because that hides the
bore shape inside a filter fit, which is the one thing that has to vary here.

### The 25–60 cent sharpness, and how it was found

Initial agreement with stage 1:

| alpha | deviation (cents) |
|---|---|
| 0.0 | +60.4, +33.5, +25.2, +22.6, +39.9 |
| 0.5 | +45.0, +35.2, +30.1, +29.9, +41.9 |
| 1.0 | +24.1, +25.4, +25.6, +35.3, +38.3 |

Three hypotheses were tested and **all three were innocent**:

| hypothesis | test | verdict |
|---|---|---|
| spatial discretisation | sample rate 48k → 384k | error *grew* slightly (61 → 65 cents). Not it. |
| the mouthpiece's tiny features | remove mouthpiece | 66 cents, unchanged. Not it. |
| the bell's stepped approximation | remove bell | 68 cents, unchanged. Not it. |

Meanwhile a bare 400-section uniform ladder matched `(2n-1)c/4L` to within
0.58–1.15 cents, so the core machinery was correct.

**It was the boundary-layer phase-velocity correction.** Stage 1's complex
wavenumber has `Re(k) = ω/c + α`; a plain ladder propagates at exactly `c`.
Predicted error at f₁: **65.6 cents**. Measured: **68.3**.

### Distributed, not lumped

Adding the correction to the termination did nothing for the low regimes:

| dispersion reference | resonances at α=0 |
|---|---|
| 60 Hz (delay 9.98 samples) | 37.14, 111.22, 184.42, 241.40 |
| 250 Hz (delay 4.89) | 37.14, 111.23, 184.56, 251.27 |
| 1000 Hz (delay 2.44) | 37.14, 111.24, 184.61, 253.32 |

The first three resonances are **completely insensitive** to the termination
delay. The reason is physical and is the bell cutoff: at low frequency the
flare reflects the wave long before it reaches the mouth, so a delay there is
never traversed.

Moved into every section as a one-pole carrying that section's share of the
excess delay, f₁ went from **+61 to −3 cents**.

The radiation **end correction** (2 × 0.6133 × 108 mm ≈ 18.5 samples round
trip) matters for the same reason and is in the termination, where the high
regimes do reach.

### Final agreement

| alpha | f₂/f₁ stage 1 | f₂/f₁ waveguide | f₃/f₁ stage 1 | f₃/f₁ waveguide |
|---|---|---|---|---|
| 0.00 | 3.044 | 2.995 | 5.078 | 4.972 |
| 0.25 | 2.599 | 2.568 | 4.270 | 4.200 |
| 0.50 | 2.310 | 2.295 | 3.716 | 3.678 |
| 0.75 | 2.132 | 2.127 | 3.339 | 3.326 |
| 1.00 | 2.045 | 2.044 | 3.120 | 3.119 |

Absolute pitch remained up to ~30 cents off, because a one-pole's group delay
is flat where the true excess delay falls as 1/√f. `tune_to_waveguide`
re-solves the compensation against the ladder's own resonances — the waveguide
*is* the instrument, so the instrument is what gets tuned. Result: within 2.5
cents across the morph.

### Morphing mid-note

The ladder grows from 391 to 636 sections between the morph's limits, so
`Waveguide.adopt` resamples the delay lines onto the new section count.
Sweeping a control changes the count a section at a time, and stretching the
contents by one section is gentle. This is what lets the bore morph happen
while a note sustains.

---

## 5. Stage 2 — the excitation

Brass lips are **outward-striking** (pressure blows them open); a double reed
is **inward-striking** (pressure blows it shut). Nearly everything separating
the two families follows from that sign.

Flow and pressure define each other through the bore impedance, so the pair is
solved **exactly as a quadratic** rather than iterated — this is what keeps it
stable when blown hard.

### Two findings that made it playable

**Q below ~7 does not oscillate.** The first version (Q=3) produced only a
ring-down from the attack. Envelope at Q=3 vs Q=10, same conditions:

| Q | opening mean/std (µm) | output std |
|---|---|---|
| 3 | 598.8 / 0.6 | 3.5e-3 (dead) |
| 10 | 562.4 / 218.7 | 2.55 (oscillating) |

**Closing pressure must stay well above blowing pressure.** Otherwise the valve
slams into its end stops every cycle and becomes a relaxation oscillator the
bore alone controls. Embouchure tracking, lip → sounding:

| p_close | Q | lip 36, 72, 108, 182, 253 → |
|---|---|---|
| 8000 | 15 | 40, 70, **240**, **395**, 265 (erratic) |
| 15000 | 15 | 40, 70, 115, 190, 260 (tracks) |

With p_close = 12000–15000 and Q = 15 the embouchure selects regimes correctly,
landing on the nearest regime above the lip — correct outward-striking
behaviour. Silence between partials is also correct.

### Behaviour achieved

| property | evidence |
|---|---|
| self-sustains | envelope flat over 0.8 s |
| louder with breath | rms 0.03 → 4.7 over 2000–8000 Pa |
| **brighter** with breath | spectral centroid 116 → 392 Hz |
| overblows under pressure | jumps to 244 Hz at 6000 Pa |
| lips through partials | monotone across embouchure |
| silent between partials | lip=145 Hz produces nothing |
| cracks up a partial | slide run out under held embouchure: 187.5 → 181.6 → 175.8 → **210.9** |

That last one was initially a test failure. It is correct trombone behaviour;
the test was over-reaching and became two tests.

---

## 6. Conical-end register control

Reported problem: at α=1 the embouchure stopped selecting regimes cleanly.

### It was not the cone

| compensation | length at α=1 | mean regime spacing | sounded vs embouchure (90–250 Hz) |
|---|---|---|---|
| regime 1 | 4.55 m | 35.6 Hz | 557, 275, 311, 521, 521 — **erratic** |
| regime 3 | 2.96 m | 54.6 Hz | 223, 305, 375, 516, 580 — **monotone** |

Holding the pedal note builds a 4.5 m instrument whose regimes sit at two
thirds the spacing, leaving the excitation too little to discriminate. The fix
was a tuning decision made two sections earlier; the original diagnosis mistook
a consequence for a cause.

The problem case was specifically α=1 **with β=1**. Brass lips at α=1 were
already monotone.

### The register vent

Built anyway, because it is a real mechanism the instrument lacked. A side hole
shunts the bore through the inertance of the air in it, `Z = jωρt/S`, whose
impedance *rises* with frequency — low regimes see near a short circuit, high
ones see something they can ignore.

Acoustic model, vent open vs shut:

| alpha | shut | open |
|---|---|---|
| 0.00 | 35.8, 109.1, 182.1, 252.8 | 168.8, 187.9, 261.8, 356.2 |
| 0.50 | 50.1, 114.8, 183.7, 248.1 | 173.3, 184.8, 367.5, 443.0 |
| 1.00 | 63.6, 128.2, 193.2, 246.9 | 185.1, 195.2, 387.8, 422.1 |

In the waveguide the same shunt produces a **first-order high-pass** at the
junction, corner `ω_c = Zc / 2L_h`, proportional to open area. Measured floor
raised 1.30–1.74× in the sounding instrument.

### The vent must travel

A register hole sits at a pressure node of the regime it preserves, and here
the bore changes shape while playing, so the node moves:

| alpha | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| position (fraction of bore) | 0.16 | 0.18 | 0.20 | 0.22 | 0.26 |
| detune of target regime | 55 c | 17 c | 10 c | 16 c | 18 c |

### Two scoring mistakes

1. **The search reported infinite promotion at a useless position.** A vent
   that destroys the target regime as well leaves nothing underneath to compare
   against, so the score was `kept − (−∞)`. Pitch is a **constraint**, not a
   quantity to trade; and "nothing left below" needs a finite sentinel.
2. **A regime has several nodes**, all of which promote it, so the search
   jumped between them from one morph position to the next — schedule
   `[0.16, 0.58, 0.18, …]`. Taking the earliest node that does essentially as
   well made it monotone: `[0.16, 0.18, 0.18, 0.18, 0.20, 0.20, 0.22, 0.24, 0.26]`.

### Verdict

The vent works but is **not** the answer to the problem it was built for.
Opening it while lipping fights the embouchure, because it pins one regime
while the player wants others. It wants a deliberate place in a fingering
scheme, as an octave key, not an automatic one.

---

## 7. The reed morph compensation

`beta` dragged the pitch by 833 cents across its range.

### The dead zone

Running one coupling coefficient from +1 to −1 puts a **zero at the midpoint**,
where the valve stops responding to pressure and the embouchure loses all
authority. Measured: 324 cents off target at β=0.5, unreachable by any
embouchure, while every other position came within ~50 cents.

Replaced by a **crossfade between two valves**, one blowing open and one
blowing closed, each keeping full coupling, their openings blended into one
effective aperture. Uncompensated drift then:

| beta | 0.00 | 0.12 | 0.25 | 0.38 | 0.50 | 0.62 | 0.75 | 1.00 |
|---|---|---|---|---|---|---|---|---|
| cents | 0 | 0 | −52 | −52 | −52 | +1723 | +1723 | +1723 |

The whole lower half became nearly pitch-stable unaided.

### Two causes, two corrections

- **Which regime speaks** — set by the embouchure, corrected by scaling it.
- **Where inside that regime the note settles** — cannot be corrected by
  embouchure at all. An outward-striking valve sounds a resonance sharp, an
  inward-striking one flat: **217 cents apart on the same regime**. Only length
  moves it, so the second part is a slide trim, and it goes negative. `slide`
  became a signed trim for this.

### It is a surface, not a curve

A correction calibrated at α=0.5 and applied elsewhere missed by over an
octave (1214 cents at α=0, 1127 at α=1). The two morphs are not separable.

### Interpolation rule

The embouchure entry was first looked up by **nearest neighbour, never
interpolated** — it picks a regime, and a regime is discrete. A table whose
every grid point was within 26 cents produced 1200-cent errors *between* them
until the lookup stopped blending. The slide trim *is* interpolated, since
pitch goes smoothly with length once the regime is fixed.

That rule was later refined to **interpolate within a plateau, snap across a
cliff** — see the next section. Snapping everywhere turned out to be too blunt:
most of the table is genuinely continuous, and stepping through it is audible
under a sweep.

### Result, before the off-grid gap was closed

| condition | drift |
|---|---|
| uncompensated | 833 cents |
| on the calibration grid | 26 c worst, 12 c median, 0 of 24 over 50 |
| between grid points | 15 c median, **5 of 19 jump an octave** |

Audio at this point: untuned rises 835 cents and stays; tuned holds 198–200 Hz
with one brief mid-sweep excursion.

This is the state the next section starts from.

### Closing the off-grid gap

The correction above held its own grid to 26 cents and missed by an octave at a
quarter of the points between grid nodes. Four attempts from the table's side
failed; one change away from the table fixed most of it.

**Diagnosis.** The multiplier table was not a smooth surface but two plateaus
with a cliff between them, the cliff a factor of ~1.94 — an octave in valve
frequency. Rows for α ≤ 0.5 were *identical*; only the cliff's position moved
with α. Meanwhile the bore's impedance peak heights reorder across the morph
(at α=0 the lowest regime is strongest at 153 dB descending; at α=1 regime 3 is
strongest at 164.7 dB), so which regime wins the loop-gain competition changes.

**Root cause.** The valve's resonance sweeps by `reed_frequency_ratio` as β runs
0→1, walking it straight across regime boundaries. Worst jump between adjacent
β steps, fixed embouchure:

Worst jump between adjacent β steps, **over the cylindrical half of the morph
(α ≤ 0.5)**, which is where the ratios separate cleanly:

| ratio | 1.0 | 1.25 | 1.5 | 2.0 |
|---|---|---|---|---|
| worst adjacent jump | silent dead zone | **55 c** | 471 c | 969 c |

Ratio 1.0 opens a silent dead zone through the middle of the morph — both
valves at the same frequency with opposite striking, so the aperture stops
modulating. 1.25 is the optimum.

**The α ≤ 0.5 qualification is load-bearing.** A later sweep of 1.10–1.25
over α ∈ {0, 0.375, 0.75, 1.0} found *every* ratio jumping at high α: worst
adjacent jump 1141 c at α=0.75 for ratio 1.10, 1114 c for 1.25, and at α=1.0
ratio 1.25 gives 866 c where 2.0 gives only 333 c. So the ratio does not order
the ratios consistently everywhere — it buys smoothness across most of the
plane, and the α ≥ 0.75 corner is hard whatever the valve does. That corner is
exactly what survives in the final result below.

(One caveat on that sweep: the 1226 c figures it reported at α=0.375 came from
the *first* β sample reading 393 Hz, an attack artefact on a fresh voice rather
than a jump within the morph. The α=0.75 and α=1.0 figures are real.)

**The lookup was generalised** from "always snap" to **interpolate within a
plateau, snap across a cliff**, deciding from the four bracketing entries
(`PLATEAU_TOLERANCE = 1.12`; a plateau varies a few percent, the smallest
regime step on this instrument is ~30%).

**Four things that failed**, each measured:

| attempt | result |
|---|---|
| Denser grid, 9×9 vs 5×9 | worse off-grid (6/18 vs 5/19) |
| Higher valve Q as a regime-locking term (15 → 25 → 40) | worse; Q=40 introduced a jump at α=0 that was not there |
| Prefer the middle of a run of acceptable candidates | 3 off-grid failures instead of 1 |
| Re-solve only failing entries with a long held note | repaired one entry to a value its neighbours did not share; the new cliff broke two nearby points |
| Longer probes, 0.8 s vs 0.3 s | *identical* results under the smooth table, 2.6× the cost (they had been actively worse under the old one) |

The last two failed the same way: **the table's smoothness is worth more than
any individual entry's accuracy.**

**Result**, 51 points across the (α, β) plane, fresh voice and 1 s settle:

| | before | after |
|---|---|---|
| off-grid worst | 1215 c | **100 c** |
| off-grid over 50 c | 5/19 | **1/19** |
| on-grid over 50 c | 0/24 | 1/23 (720 c at α=0.75, β=0.75) |
| deep-interior points | not measured | 1/9, worst 58 c |
| **overall median** | 15 c | **11 c**, 3/51 over 50 |

Audio: the tuned reed sweep now holds 44 cents with no jumps, against a 695-cent
mid-sweep excursion before; the untuned sweep fell from 835 cents with an octave
jump to 161 cents of smooth drift. The bore glissando is unaffected at 35 cents.

**One test failure was informative rather than cosmetic.** The coarse test
fixture kept missing its bar, and the cause was candidate *resolution*, not the
method: at 9 candidates the scan steps by ×1.34 between embouchures and walks
past the right one. Measured on the same grid — 10/15 within 120 cents at 9
candidates, 12/15 at 15, **15/15 at 19**. Coarsening the grid is cheap;
coarsening the candidates is not.

---

## 8. Faust toolchain

**Faust is in Ubuntu's universe repository and installs cleanly.** An earlier
attempt in this session failed only because `apt-get update` had not run — the
error reads "unable to locate package", which looks exactly like the package
not existing. That mistake cost a whole round of writing the port blind.

`.claude/hooks/session-start.sh` now installs the Python stack and Faust, and
sets `PYTHONPATH`; `.claude/settings.json` registers it. It exits immediately
when not in a remote container. Verified: `FAUST Version 2.70.3`.

### The valve bug

With a compiler available, the Faust `valve` was checked against the Python
sample by sample. It **compiled, ran, and was wrong**:

| sample | Faust (first version) | Python |
|---|---|---|
| 1 | 1.857678e-07 | 3.000265e-04 |
| 2 | 5.569195e-07 | 3.000796e-04 |

**Faust initialises feedback state to zero**, so a state holding the valve's
*opening* starts the instrument with the valve shut rather than at rest — 2.3%
off after 10 ms, completely wrong at the attack. Carrying the deviation from
rest fixes it, since that really is zero at rest.

After the fix, agreement to nine significant figures:

| sample | Faust | Python |
|---|---|---|
| 1 | 3.000265382e-04 | 3.000265383e-04 |
| 6 | 3.005548947e-04 | 3.005548963e-04 |

Also ruled out along the way: float vs double precision (no change), feedback
output ordering (swapping gave −0.267, wildly wrong), and a one-sample offset
(checked 478–482 steps, none matched).

**Compiling is not evidence.** The status block in `dsp/trombolese.dsp` now
separates what has been checked numerically from what has only been compiled.

---

## 9. Dead ends and disproved hypotheses

Kept because each cost time and each looks plausible enough to be re-tried.

| hypothesis | how tested | outcome |
|---|---|---|
| A denser reed-compensation grid fixes the octave jumps | 9×9 vs 5×9 | **Disproved.** Worse off-grid. |
| Higher valve Q locks the reed to the intended regime | Q 15/25/40 | **Disproved.** Worse; Q=40 added a jump at α=0. |
| Preferring the middle of a candidate run is more robust | rebuilt the table | **Disproved.** 3 off-grid failures instead of 1. |
| Re-solving failing entries with a long held note repairs them | added a repair pass | **Disproved.** Created a cliff that broke two good neighbours. |
| Waveguide sharpness is spatial discretisation | fs 48k → 384k | **Disproved.** Error did not shrink. |
| …is the mouthpiece's undersampled features | removed mouthpiece | **Disproved.** Unchanged. |
| …is the bell's stepped approximation | removed bell | **Disproved.** Unchanged. |
| Dispersion can be lumped at the termination | swept reference 60–1000 Hz | **Disproved.** Low regimes insensitive — the bell reflects them first. |
| A per-section one-pole can match the √f law | analysed group delay | **Disproved.** A one-pole's delay is flat across 30–900 Hz. Exact at one frequency only. |
| Longer calibration probes are more accurate | 0.8 s vs 0.3 s, twice | **Disproved twice.** Verified distinctly *worse* under the cliff-ridden table, then *identically* under the smooth one, for 2.6× the cost. Whatever remains is not a settling problem. |
| The register vent fixes conical-end register control | measured with and without | **Disproved.** Vent works, but the cause was the compensation target. Vent makes lipping worse. |
| Voice state leaks across `reset()` | fresh vs reused voice at a failing point | **Disproved.** Both 199.0 Hz. The apparent contradiction was between two different calibration tables. |
| A lower `reed_frequency_ratio` reduces drift enough on its own | ratios 1.5–3.5, before the crossfade and slide trim existed | **Superseded, and it was the answer.** Judged "partly" at the time because after compensation 1.5 and 3.5 both sat at 217 c worst, so ratio 2.0 was kept. Re-measured later across 1.0–2.0 it proved to be the whole fix: the ratio decides whether the valve crosses regime boundaries at all. See §7. Default is now 1.25. |
| Higher blowing-pressure or aperture tuning fixes the conical end | parameter sweeps | **Not the lever.** The conical end's problem was the compensation target (§6); the reed morph's was the frequency ratio (§7). Neither was an amplitude parameter. |

### Process mistakes worth not repeating

- **`pkill -f explore_morph.py` matched its own shell** and killed the session's
  bash (exit 144).
- **Piping a long script to `head`** sent SIGPIPE part-way through, leaving a
  half-written figure set that looked current; a stale figure was then read and
  briefly believed.
- **Reusing one `Voice` across a measurement sweep** produced 1200-cent
  artifacts that were not real. Use a fresh voice and ~1 s settle per point.
- **Asserting more than the method delivers.** Two tests initially claimed every
  point would land; both had to be restated to what was actually measured.

---

## 10. Final parameter set

### Geometry (`Trombolese`)

| parameter | value | why |
|---|---|---|
| `bore_length` | 2.15 m | before compensation |
| `cyl_throat_radius` / `cyl_bell_entry_radius` | 6.95 mm | tenor trombone bore |
| `cone_throat_radius` | 2.00 mm | oboe-ish reed end |
| `cone_bell_entry_radius` | 15.0 mm | forced by truncation ratio (§2b) |
| `bell_length` / `bell_radius` / `bell_flare` | 0.60 m / 108 mm / 3.5 | tenor trombone bell |
| `bell_segments` | 48 | converged to <0.1 cent |

### Excitation (`ReedParameters`)

| parameter | value | why |
|---|---|---|
| `lip_q` / `reed_q` | 15 / 12 | below ~7 it does not oscillate |
| `lip_closing_pressure` | 12000 Pa | must exceed blowing pressure |
| `reed_closing_pressure` | 7000 Pa | as tuned |
| `reed_frequency_ratio` | **1.25** | decides whether the morph crosses regime boundaries; see §7 |
| `lip_width` / `reed_width` | 12 / 10 mm | |
| rest openings | 0.30 mm both | |

### Waveguide

| parameter | value |
|---|---|
| section length | `c / fs` (7.15 mm at 48 kHz) |
| `dispersion_reference_hz` | 150 Hz (balances residual across the register) |
| `loss_reference_hz` | 250 Hz |
| mouth reflection fit | gain 0.9477, pole 0.9200 |
| end correction | 2 × 0.6133 × 108 mm ≈ 18.5 samples |

### Reed compensation lookup

| parameter | value | why |
|---|---|---|
| `PLATEAU_TOLERANCE` | 1.12 | separates "same regime" from "different regime" in the four bracketing entries. A plateau varies a few percent; the smallest regime step on this instrument is ~30% |
| `n_candidates` | 19 | candidate *resolution* is not the place to economise: 10/15 grid points land at 9 candidates, 12/15 at 15, 15/15 at 19 |
| `seconds` (probe) | 0.3 | 0.8 s measured identically for 2.6× the cost |

### Playing defaults

- `pitch_neutral(..., regime=3)` then `tune_to_waveguide(..., regime=3)`
- blowing pressure ~4200 Pa
- held regime ≈ 181–182 Hz

---

## 11. Commit history

| commit | what |
|---|---|
| `d67d40a` | Stage 1: offline acoustic model of the bore |
| `e380693` | Pitch neutrality |
| `98f8058` | Stage 2: the sounding instrument |
| `e211b53` | Register vent, and what the conical end's problem really was |
| `8ef0fea` | Faust valve verified against the Python reference |
| `3a1ccc8` | Reed morph tuned out |
| `7e54432` | CLAUDE.md |
| `aa7dc81` | `docs/SESSION-LOG.md`, this file |
| `763f23b` | Closed the reed compensation's off-grid gap |

All on `claude/trombolese-format-qak2pz`. 129 tests at time of writing.

---

## What remains

1. Close the **last** of the reed compensation's misses — about one point in
   twenty, almost all at `alpha >= 0.75, beta >= 0.75`, worst case 720 cents at
   one on-grid point. Most of the gap is closed (§7); what is left is where the
   valve genuinely prefers a neighbouring regime rather than being
   mis-tabulated, so it needs pitch tracking inside the loop. Four
   table-shaped attempts are recorded in §9 as already failed.
2. Replace the constant-delay dispersion with a fitted fractional-order filter
   (closes the remaining ~30 cents).
3. Per-section one-pole losses for frequency-dependent damping.
4. Write the Faust ladder assembly. Its one real problem: Faust fixes structure
   at compile time, but this instrument changes length while it plays. The
   intended fix is a fixed `NSECTIONS` with an interpolated delay absorbing the
   difference.
5. Voice allocation, MPE, then the CLAP wrapper.
6. Nonlinear (brassy) propagation — not optional for forte trombone timbre.
7. Give the register vent a role in a fingering scheme.
8. Notation for a bore glissando.
