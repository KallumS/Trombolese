# CLAUDE.md

Working notes for the Trombolese: a physical-modelling wind instrument that
crosses a trombone with an oboe and morphs its bore between them while playing.

The README is the narrative. This file is the operational guidance and, mostly,
the **traps** — the things that cost real time to find, that look fine until
they are measured, and that are easy to break back.

---

## The one rule

**Stage 1 is the authority for stage 2.** The transfer-matrix model
(`acoustics.py`, `bore.py`, `analysis.py`) decides what the instrument *is*;
the waveguide (`waveguide.py`, `reed.py`, `synth.py`) has to reproduce it and
is checked against it. When the two disagree, assume the waveguide is wrong
until proven otherwise — that has been true every time so far.

Corollary: **never "fix" a disagreement by loosening a stage-1 assertion.**
Three of them exist precisely to catch regressions that sounded plausible.

---

## Layout

```
src/trombolese/
    constants.py   air properties vs temperature
    acoustics.py   complex wavenumber, radiation, transfer matrices, toneholes   \
    bore.py        geometry, morph, compensation, input impedance                 > stage 1
    analysis.py    resonances, harmonic fit, pitch compensation, vent placement  /
    plots.py       study figures, light and dark
    waveguide.py   Kelly-Lochbaum scattering ladder                              \
    reed.py        two-valve morphable excitation                                 > stage 2
    synth.py       playable voice, controls, reed compensation                   /
dsp/trombolese.dsp Faust port — partly verified, see its status block
scripts/           explore_morph.py (study), render_audio.py (audio)
tests/             123 tests
```

---

## Commands and how long they take

```bash
pytest                                  # ~5 min — plan for it
pytest tests/test_acoustics.py -q       # ~20 s, the fast feedback loop
python scripts/explore_morph.py         # ~30 s, table + 10 figures
python scripts/render_audio.py          # ~1 min
python scripts/render_audio.py --tune-reed   # ~8 min (reed calibration)
faust -double dsp/trombolese.dsp -o /dev/null   # type-check only
```

Expensive calibrations, for planning:

| operation | cost |
|---|---|
| `compensate_pitch(n_alpha=21)` | ~15 s |
| `tune_to_waveguide` | ~25 s |
| `fit_register_vent(n_alpha=9)` | ~2 min |
| `compensate_reed_morph(5 alpha x 9 beta)` | ~6 min |

Run anything over a couple of minutes with `run_in_background` and wait on it
with an `until` loop, not a foreground sleep.

---

## Physics worth not re-deriving

- **Cylinder**: odd harmonics (1, 3, 5, 7), overblows at the twelfth.
  **Cone**: complete series (1, 2, 3, 4), overblows at the octave. The morph
  travels between these, 3.04 → 2.05.
- **Truncation ratio = `r_throat / r_bell_entry`.** The length cancels exactly.
  So how complete a cone is depends only on how much it opens, never on how
  long it is — which is what makes pitch compensation safe, since lengthening
  cannot rearrange the modes. Below ~0.2 for a usable harmonic series.
- **A cone cannot reach the bell at a trombone's bore radius.** 2 → 6.95 mm is
  29% truncated and gives 1, 2.21, 3.51. The conical limit has to open to
  ~15 mm, so it is nearer a euphonium than a trombone.
- **Radiation end correction is 0.6133·a** — 66 mm on this bell, 2.4% of the
  instrument. Leaving it out makes everything sharp by tens of cents.
- **Boundary-layer loss also slows the wave**: `Re(k) = ω/c + α`, so stage 1's
  resonances sit ~2.4% flat of the lossless textbook values. This is not a bug
  and the tests encode it.

---

## Traps that have already bitten

### Stage 1

**Only one regime can be held by pitch compensation.** The morph exists to
change the *ratios* between regimes, so holding all of them is a contradiction.
`compensate_pitch(regime=N)` — and it must be the regime actually played.
Compensating the pedal (`regime=1`) builds a 4.55 m instrument whose regimes
sit 35.6 Hz apart instead of 54.6, and the excitation can no longer
discriminate between them. **This, not the cone, was the "conical end is
unruly" problem.** Default to `regime=3` for anything that will be played.

**A long cylindrical section at the throat destroys conical behaviour.** An
early version kept a 0.55 m slide ahead of the taper; it has several times the
volume of the cone's missing apex, and the upper resonances then ignore the
morph entirely — only the fundamental moves. The slide is part of the taper.

### Stage 2 — the waveguide

**Dispersion must be distributed, never lumped.** The ladder propagates at
exactly `c` and so plays ~65 cents sharp at f₁. Folding the correction into the
termination does *nothing* for the low regimes, because at low frequency the
bell reflects the wave long before it reaches the mouth, so a delay there is
never traversed. Per-section one-poles took f₁ from +61 to −3 cents.

**Before blaming the ladder, rule these out** — all three were tested and
innocent: spatial discretisation (the error does not shrink from 48 kHz to
384 kHz), the mouthpiece (removing it changes nothing), the bell's stepped
approximation (removing it changes nothing).

The bare ladder is accurate to ~1 cent (`test_uniform_ladder_matches_the_textbook`).
If that test fails, the loop delay is wrong — it must come to `2 * N` samples.

### Stage 2 — the reed

- **Below about Q = 7 the valve does not oscillate at all**, it only rings down
  from the attack. The first version was silent-but-plausible for this reason.
- **Closing pressure must stay well above blowing pressure.** Otherwise the
  valve slams into its end stops every cycle and becomes a relaxation
  oscillator the bore alone controls — the embouchure stops selecting regimes
  and the instrument is unplayable. This was lip=108 Hz sounding 255 Hz instead
  of 115 Hz.
- **The morph is a crossfade between two valves, not one coupling coefficient
  swept through zero.** A single coefficient running +1 → −1 has a dead zone at
  the midpoint where the valve stops responding to pressure and the player
  loses the instrument entirely. Do not "simplify" this back.
- **The embouchure must stay live at both ends.** Pinning the double reed to an
  absolute frequency leaves the player with no register control whatsoever.
  Its frequency is a *multiple* of the embouchure (`reed_frequency_ratio`).

### Faust

- **`apt-get update` before `apt-get install faust`.** Faust is in universe and
  installs fine; without the update it fails with a bare "unable to locate
  package", which reads exactly like the package not existing. This cost a
  whole round of writing the port blind. `.claude/hooks/session-start.sh`
  handles it now.
- **Faust initialises feedback state to zero.** A state holding the valve's
  *opening* starts the instrument with the valve shut rather than at rest.
  Carry the deviation from rest instead. The wrong version compiled, ran, and
  was 2.3% off after 10 ms and completely wrong at the attack.
- **Compiling is not evidence.** Check numerically against the Python, sample
  by sample. The valve compiled too, and was wrong.

---

## Measurement discipline

Most wrong conclusions in this project came from measurement, not modelling.

- **Use a fresh `Voice` per measured point, and let it settle ~1 s.** Reusing
  one voice across a sweep produced 1200-cent artifacts that were not real.
- **Notes can change regime after the attack.** Some sound one regime for a
  third of a second, then jump to the octave above and stay. A 0.3 s probe
  measures a transient — *but* see the counter-measurement below.
- **Peak frequencies are refined by parabolic interpolation**, so grid density
  barely matters: 10k points agree with 80k to 0.002 cents. Do not pay for a
  dense frequency grid; do pay for a long enough time window.
- **Never pipe a long-running script to `head`.** SIGPIPE kills it part-way and
  you get a half-written set of outputs that look current. Redirect to a file.
- **`pkill -f <pattern>` will match its own shell** if the pattern appears in
  the command line. It killed the session's own bash once.

### Two measurements that contradict the obvious reasoning

Recorded because the reasoning is more persuasive than the data, and the data
won:

1. **Longer calibration probes made the reed compensation worse.** Probes were
   lengthened 0.3 s → 0.8 s precisely because of the settling problem above,
   and the resulting table verified distinctly worse — grid points fell silent
   or landed an octave out. No mechanism is known. The default stayed at 0.3 s.
2. **A denser calibration grid made it worse too.** 9×9 against 5×9, verified
   worse off-grid. The residual failures are not a resolution problem.

If you revisit either, re-measure before re-reasoning.

---

## Interpolation rule

Anything that **selects a regime is discrete and must not be interpolated**.
`ReedCompensation.multiplier_at` uses nearest-neighbour for exactly this
reason: blending two grid points that chose different regimes lands on a third
and misses by an octave. A table whose every grid point was within 26 cents
produced 1200-cent errors between them until the lookup stopped blending.

Anything that **tunes within a regime is continuous and should be
interpolated** — `slide_at`, the vent schedule, the pitch compensation curve.

Related: when searching for something with multiple valid answers (vent nodes,
embouchure multipliers), **tie-break toward continuity with the neighbours you
have already solved**, or the resulting curve lurches and is useless as a
control even though every individual point is optimal.

---

## Search and scoring

When writing a search over a design space here, the pattern that works is:
constrain what must not move, *then* maximise what should. Two failures from
getting that backwards:

- The vent search reported **infinite** promotion at a useless position,
  because a vent that destroys the target regime leaves nothing underneath to
  compare against. Pitch is a **constraint**, not a quantity to trade away.
- Scores that can be infinite will win. Give "nothing left below" a finite
  sentinel.

---

## Conventions

- Physics modules carry their approximations in the module docstring, with the
  validity bound stated. Keep doing this — several of the traps above are
  documented there rather than in comments.
- Comments explain *why*, especially where a constraint looks arbitrary. Most
  of the odd-looking ones encode a measurement.
- Tests are named for the behaviour, and the ones guarding a past bug say so.
- Figures: `plots.py` uses validated ordinal ramps (5 steps, both modes) —
  alpha is a continuous magnitude, so it gets a sequential ramp, never
  categorical colours. In matplotlib, `set_title(loc="left")` uses a *different*
  Text object than `ax.title`, so the theme colour must be passed explicitly or
  dark mode gets a black title.
- Commit messages: what changed, what was learned, what was measured.

---

## Known limitations

- **Reed compensation jumps an octave at ~a quarter of off-grid points.** Needs
  pitch tracking in the loop or a regime-locking term in the excitation — not a
  bigger table (measured).
- **The register vent works but has no role yet.** It does what a register key
  does, but the problem it was built for had another cause, and opening it
  while lipping fights the embouchure. It wants a deliberate place in a
  fingering scheme.
- **Waveguide absolute pitch is ~30 cents off stage 1** because a one-pole's
  group delay is flat where the true excess delay falls as 1/√f. Closing it
  means a fitted fractional-order filter. `tune_to_waveguide` papers over it
  for the played regime.
- **Loss is frequency-independent per section** (scalar at a reference
  frequency), so the model is slightly too bright low and too damped high.
- **No nonlinear (brassy) propagation.** Not optional if it is ever to sound
  like a trombone at forte.
- **The Faust ladder assembly is not written.** Its one real problem is stated
  in the file: Faust fixes structure at compile time, but this instrument
  changes length while it plays.
