# OpenOptiSim — product context

## What it is

An open-source simulator for optical communication links and photonic systems. Engineers build a
link as a block diagram — laser, modulator, fiber, amplifier, detector — run it, and read the
result as an eye diagram, a BER number, an OSNR figure.

The engine exists and is validated: 295 tests, every physics block checked against a closed-form
result in CI. What does not exist yet is the interface.

## Who uses it

**Primary: the practising link engineer.** Designs or verifies an optical link. Comes from
OptiSystem or VPIphotonics and expects their muscle memory to transfer. Cares about being right,
and about being able to prove it.

**Secondary: the graduate researcher.** Needs a specific effect modelled correctly and needs to
extend the tool with their own component. Will read the source. Cannot afford a commercial licence
— which is the reason this project exists.

**Tertiary: the student.** Learning what dispersion does to a pulse. Needs the tool to be
legible before it is powerful.

## The scene

A desk, two monitors, a long working session. The engineer is *iterating*: change a parameter,
run, look at the eye, change it again. Dozens of times an hour. Sometimes at night in a dim room,
just as often in a lit office. Frequently alongside a datasheet PDF and a terminal.

That scene decides several things. The interface ships **two grounds and defaults to the light
one** — a schematic is a document before it is a screen, and the plots leave here for reports,
theses and papers, where a screenshot off a black canvas is the wrong artefact. Graphite is one
click away for the night session beside a terminal. Neither is a skin over the other: the port
hues are the identity and survive both, but their values are re-derived per ground rather than
inverted, and the eye diagram's density ramp is rebuilt so that in both cases *distance from the
page* means count.

The interface is also **dense** — screen space spent on decoration is screen space not spent
on the schematic or the plot. And the run/inspect cycle must feel **instant**, because it happens
constantly and any latency compounds.

## What the GUI must do

1. Build and edit a link as a node graph, with invalid connections refused at edit time.
2. Edit component parameters with their real units visible.
3. Run, and show progress on a run that takes seconds to minutes.
4. Display results: eye diagram, spectrum, BER curves, sweep results.
5. Open and save `.oosim` projects.
6. Sweep a parameter and plot the curve.

## Constraints that are already decided

* **The GUI is a client of the public Python API.** If a feature is not reachable from Python, it
  does not exist. The GUI gets no privileged access.
* **The component palette is generated** from `oosim.manifests()`, never hand-written, so it
  cannot drift from what the engine actually offers.
* **Result data is reduced in the engine**, never in the browser. An eye diagram arrives as a
  binned histogram of fixed size; the browser never receives a raw sample buffer.
* **Port types are enforced**: optical, electrical, binary, symbol, metric. The editor must make
  a type mismatch visibly impossible rather than merely erroring on run.
* Web-first, React Flow for the canvas. Optional desktop wrapper is packaging, not architecture.

## Brand commitments

**Spectral identity** (chosen by the project owner). Accent colours derive from real optical
wavelengths rather than being picked for taste: C-band cyan as the primary, and the port types
keyed to distinct points so a wire's colour tells an engineer what travels down it. The tool is
about light; its palette should be too.

## What would make a polished result feel wrong

* Consumer-app airiness. Generous whitespace, large type, and gentle animation read as *slow* to
  someone doing this fifty times an hour.
* Hiding density behind progressive disclosure. This audience wants the numbers on screen.
* A novel arrangement for the schematic editor. Familiarity here is a feature; invention belongs
  in the physics, not in where the palette lives.
* Any claim the engine cannot back. Every number shown must come from a real run.

## Success

An engineer who has used OptiSystem sits down, finds everything where they expect it, builds a
link in two minutes, and gets a number they trust.
