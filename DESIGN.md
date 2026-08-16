# Design

The visual and interaction record for OpenOptiSim's interface. It exists because
[`docs/ui-mockup.html`](docs/ui-mockup.html) is a *build*, and a build without a
written rationale is a set of numbers nobody can argue with later. Everything
here was decided against the audience and use scene in [PRODUCT.md](PRODUCT.md);
where a decision was made by measurement rather than judgement, the measurement
is given.

---

## 1. What the surface is

An **Operate** surface: a schematic editor, used to complete a task rather than
to be admired or read through. Scanability, consistency and the real usage scene
outrank expression. Brand lives in precise details, not in gestures.

The engineer is *iterating* — change a parameter, run, look at the constellation,
change it again — dozens of times an hour. Three consequences follow, and they
decide most of what is below:

1. **Density beats generosity.** Screen space spent on decoration is space not
   spent on the schematic or the plot.
2. **Recognition beats novelty.** The layout is the one this audience already
   knows from Blender, VS Code and EDA tools. Nothing is moved to be different.
3. **Nothing on screen may be invented.** Every number, every parameter, every
   port comes from a real engine run. See §7.

---

## 2. Two grounds

The interface ships **paper** and **graphite**, and defaults to paper.

A schematic is a document before it is a screen. Its plots leave the tool for
reports, theses and papers, where a screenshot off a black canvas is the wrong
artefact. Graphite is one click away for the night session beside a terminal.

Neither is a skin over the other:

| | paper | graphite |
| :--- | :--- | :--- |
| ground | `#e9edf3` bed under white panels | `#0b0e13` under `#11151c` panels |
| a block | white card on a grey bed | raised panel on a dark one |
| hover | adds **ink** (darker) | adds **light** |
| plot bed | `#ffffff` — a plot is on paper | `#080b10` |

The port hues are the identity and survive both, but their **values are
re-derived per ground rather than inverted** — on paper the spectrum darkens to
hold contrast against a light bed.

**Structure.** The bare `:root` block carries the complete paper palette. The
graphite tokens appear twice: once under
`@media (prefers-color-scheme: dark)` guarded as `:root:not([data-theme="light"])`,
and once under `:root[data-theme="dark"]` so an explicit choice wins in both
directions. No colour is ever defined *only* inside a media or `[data-theme]`
block — that is the classic unreadable-page bug, and it is checked rather than
assumed (§8).

---

## 3. The spectral port identity

**Every port colour is a wavelength, not a preference.** A wire's colour tells
you what travels down it.

| port type | paper | graphite |
| :--- | :--- | :--- |
| optical (C-band) | `#0a8279` | `#22d3c5` |
| electrical | `#a35f00` | `#f0a030` |
| binary | `#5a6474` | `#7c86a0` |
| symbol | `#6539cc` | `#9b7cf6` |
| metric | `#bc2668` | `#e85d9b` |

This is the one place the interface is allowed to be memorable, and it earns its
place by being *information*: a typed-port system that refuses invalid wiring at
edit time is meaningless if the types are invisible.

The shown project is chosen so that **all five types appear on the canvas at
once**. That is why it is a coherent link rather than an on-off-keyed one: only
a coherent chain carries binary into symbols, symbols into two electrical
drives, an optical field, two photocurrents back, and symbols out again.

Accent is `--optical`, because the tool is an optical simulator. Semantic colour
(`--good` / `--warn` / `--bad`) is kept separate from it and does not count as
the accent.

---

## 4. The ink ladder

Four layers, spaced so hierarchy comes from the **distance between them** rather
than from pushing the weakest below readability.

| token | paper | vs white | graphite | role |
| :--- | :--- | ---: | :--- | :--- |
| `--ink` | `#0f141c` | 18.5 | `#e6ecf5` | values, headings |
| `--ink-2` | `#333d4b` | 11.0 | `#a8b4c6` | labels, body |
| `--ink-3` | `#4c5666` | 7.4 | `#8e9bb0` | section titles, axes |
| `--ink-4` | `#5f6a79` | 5.5 | `#7a869b` | units, ranges, disabled |

The bottom rung is set by the **tinted grounds it actually lands on** — a unit
inside an input well, a disabled toolbar label — not by white, where it would
look too pale. Against the darkest surface it still clears 4.5:1.

---

## 5. Layout

Four regions, top to bottom:

```
menu bar            34px   File / Edit / Simulate / View / Help, ground switch, engine state
action toolbar      40px   Run, Stop, Sweep · select/pan/align · zoom · run context
body                1fr    palette 216px | node canvas 1fr | inspector 260px
results dock        292px  tabs + metric strip, plot + side readouts
status bar          24px   block and link counts, selection, sample count, precision
```

**Run sits leftmost in the toolbar**, where the eye lands first, because it is
the action taken dozens of times an hour.

**The dock is 292px** because the constellation's plot is square and therefore
sized by the dock's *height*, not its width. A shorter dock wastes the width it
has.

Below 940px the palette and inspector collapse and a note says so. This is a
desktop tool; the real build would dock them as overlays rather than dropping
them, and the note says that too rather than pretending the narrow layout is the
intended one.

---

## 6. Plots

Two reduced result types, and they share a visual language because they mean the
same thing: **distance from the page is how many landed there**.

- **Eye diagram** — density ramp bed → cyan → far stop.
- **Constellation** — the same ramp; axes through the origin, because a
  constellation is read relative to zero and not to the corner of a box; ideal
  points marked with **crosses rather than filled dots**, so a cluster's own
  centre stays visible underneath and a bias offset reads instead of being
  covered by the marker meant to locate it.
- **Sensitivity** — one curve per format, each labelled where it crosses the FEC
  threshold rather than in a legend box. The threshold is 1e-3, not Q = 6:
  nobody operates an error-free channel any more, they operate one a
  soft-decision code can close.

The ramp's mid stop is placed by **luminance**, at roughly the same fraction of
each ground's span, which is what keeps sparse outliers subordinate to the dense
regions on both.

A canvas has no cascade, so the plots read the same tokens as everything else at
draw time and are redrawn when the ground changes. Nothing below the token block
names a colour.

---

## 7. Real data, always

The palette is generated from `manifests()` — the same call the real GUI will
make. The inspector shows true parameters, units, ranges and docstrings. The
constellation, eye and sweep are a real run, exported by
[`examples/export_ui_data.py`](examples/export_ui_data.py).

This is not a purity exercise. Building the mockup against real data has twice
found engine defects that the test suite missed:

- `eye_histogram` accepted any `time_bins` and rendered a 32-sample trace across
  96 columns as **vertical banding**. Time resolution is now capped at one
  column per sample.
- Placing an analyser after the differential decoder reported **EVM of exactly
  zero** however bad the link was, because that block emits decisions. The
  mockup now carries two analysers — soft measurement before the decoder, error
  count after — which is also how a bench does it.

---

## 8. What is verified, and how

Not by looking at downscaled screenshots. Each publish is checked by
measurement, in **both grounds**:

- **Contrast** — every text node's computed colour against its resolved
  background; ~110 elements per dock tab. Target 4.5:1 (3:1 for large text).
  Current status: **zero failures**, both grounds, all four tabs.
- **Token completeness** — every `var(--…)` the stylesheet references must
  resolve from bare `:root` alone. Current: 34/34.
- **Theme-block drift** — the two graphite blocks are compared key by key.
- **Canvas repaint** — plot beds are sampled after switching, in both
  directions.
- **Geometry** — no SVG element may stray outside the viewBox; no horizontal
  page overflow at 1440px.

Two findings worth keeping:

- A contrast audit run while the browser pane was not compositing reported a
  clean result that **meant nothing** — the page had laid out at its narrow
  breakpoint, where the palette and inspector are `display: none` and were never
  checked. Verify the viewport before trusting the audit.
- The design detector once ran degraded and returned `[]`. An empty result from
  a degraded tool is an undercount, not a pass.

---

## 9. Decisions worth not re-litigating

- **Paper is the default**, and it is stamped before first paint so a dark host
  never flashes through. The stylesheet still answers `prefers-color-scheme` on
  its own for the no-script case.
- **A backward wire is the schematic wrapping**, not a mistake. It needs a
  *tighter* bezier control offset than a forward one; scaling the offset with
  the span throws the curve outside the canvas on exactly the wire that already
  travels furthest.
- **Node labels truncate with the full name in a tooltip.** A fixed-width box
  that lets long names spill across neighbours is worse than one that clips.
- **Hover on paper adds ink, not light.** Lightening a hover on a white ground
  moves it towards invisible.

---

## 10. Not done

- The mockup shows a single-carrier link. The dual-polarization link is in
  [`examples/dualpol_link.py`](examples/dualpol_link.py) and is not on the
  canvas: at ~20 blocks the node text stops being readable at this canvas size,
  and a schematic nobody can read is not a better demonstration.
- No motion beyond the run pulse and the control transitions.
- No empty, loading or error states — there is no session server yet to produce
  them.
