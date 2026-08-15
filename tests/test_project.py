"""Validation of the .oosim project file format and the component registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oosim import (
    Graph,
    ProjectError,
    SimulationContext,
    UnknownComponentError,
    load,
    manifests,
    registered_names,
    save,
)
from oosim.components import (
    BERAnalyzer,
    Combiner,
    CWLaser,
    ElectricalFilter,
    Fiber,
    MachZehnderModulator,
    NRZDriver,
    PINPhotodiode,
    PowerMeter,
    PRBSGenerator,
)
from oosim.project import SCHEMA_VERSION, graph_to_dict, ui_from_dict
from oosim.registry import lookup


def _link_graph() -> Graph:
    """A full OOK link, so the round trip covers every kind of node and edge."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=512, seed=7)
    g = Graph(ctx)
    prbs = g.add(PRBSGenerator(order=15.0, label="prbs"))
    driver = g.add(NRZDriver(v_low=4.0, v_high=0.0, label="driver"))
    laser = g.add(CWLaser(power=-18.0, wavelength=1550.0, label="laser"))
    mzm = g.add(MachZehnderModulator(v_pi=4.0, extinction_ratio=30.0, label="mzm"))
    fiber = g.add(Fiber(length=40.0, attenuation=0.2, dispersion=17.0, label="fiber"))
    pin = g.add(PINPhotodiode(responsivity=0.8, shot_noise=True, label="pin"))
    lpf = g.add(ElectricalFilter(bandwidth=7.0, label="lpf"))
    analyzer = g.add(BERAnalyzer(label="ber"))

    g.chain(prbs, driver)
    g.connect(laser, mzm["optical_in"])
    g.connect(driver, mzm["electrical_in"])
    g.chain(mzm, fiber, pin, lpf)
    g.connect(lpf, analyzer["in"])
    g.connect(prbs["out"], analyzer["reference"])
    return g


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_built_in_components_are_registered_on_import() -> None:
    names = registered_names()
    assert "CWLaser" in names
    assert "MachZehnderModulator" in names
    assert lookup("Fiber") is Fiber


def test_unknown_component_names_the_problem_and_the_fix() -> None:
    with pytest.raises(UnknownComponentError, match="import that package"):
        lookup("SomeoneElsesLaser")


def test_a_project_file_cannot_name_an_arbitrary_import_path() -> None:
    """Resolving a name by importing a dotted path out of a file would make
    opening someone else's project equivalent to running their code. Only
    already-registered names resolve.
    """
    with pytest.raises(UnknownComponentError):
        lookup("os.system")
    with pytest.raises(UnknownComponentError):
        lookup("oosim.components.sources.CWLaser")


def test_manifests_cover_every_registered_component() -> None:
    """The GUI palette is generated from the classes, so it cannot drift."""
    generated = manifests()
    assert set(generated) == set(registered_names())
    assert generated["CWLaser"]["ports"] == {"inputs": {}, "outputs": {"out": "optical"}}


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------


def test_saving_and_loading_reproduces_the_results_exactly(tmp_path: Path) -> None:
    """The property that makes a project file worth having: a saved experiment
    re-run later must give the same answer, bit for bit."""
    original = _link_graph()
    before = original.run()[next(c for c in original.components if isinstance(c, BERAnalyzer))]

    reloaded = load(save(original, tmp_path / "link.oosim"))
    analyzer = next(c for c in reloaded.components if isinstance(c, BERAnalyzer))
    after = reloaded.run()[analyzer]

    assert after.q_factor == before.q_factor
    assert after.errors == before.errors
    assert after.threshold == before.threshold


def test_round_trip_preserves_structure(tmp_path: Path) -> None:
    original = _link_graph()
    reloaded = load(save(original, tmp_path / "link.oosim"))

    assert [c.label for c in reloaded.components] == [c.label for c in original.components]
    assert reloaded.edges.keys() == original.edges.keys()
    assert reloaded.ctx == original.ctx


def test_structural_config_round_trips(tmp_path: Path) -> None:
    """A combiner's port count is not a parameter — it changes the shape of the
    component — so it is stored separately and must survive the trip."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=4, sequence_length=16)
    g = Graph(ctx)
    for i in range(3):
        g.add(CWLaser(wavelength=1550.0 + i, label=f"ch{i}"))
    mux = g.add(Combiner(3, label="mux"))
    meter = g.add(PowerMeter(label="meter"))
    for i in range(3):
        g.connect(g.components[i], mux[f"in{i}"])
    g.connect(mux, meter)

    reloaded = load(save(g, tmp_path / "wdm.oosim"))
    restored = next(c for c in reloaded.components if isinstance(c, Combiner))

    assert restored.num_inputs == 3
    assert set(restored.inputs) == {"in0", "in1", "in2"}
    assert len(reloaded.run()[reloaded.components[-1]].bands) == 3


def test_the_file_is_readable_json(tmp_path: Path) -> None:
    path = save(_link_graph(), tmp_path / "link.oosim")
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["schema_version"] == SCHEMA_VERSION
    assert "oosim_version" in data
    assert data["context"]["bit_rate"] == 10e9
    assert {node["id"] for node in data["nodes"]} == {
        "prbs",
        "driver",
        "laser",
        "mzm",
        "fiber",
        "pin",
        "lpf",
        "ber",
    }
    assert {"from": ["mzm", "out"], "to": ["fiber", "in"]} in data["edges"]


def test_only_explicitly_set_parameters_are_stored() -> None:
    """A file records the choices its author made, not the defaults they
    accepted. That keeps diffs about the physics."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=4, sequence_length=16)
    g = Graph(ctx)
    laser = g.add(CWLaser(power=3.0, label="laser"))  # wavelength/linewidth left at default
    meter = g.add(PowerMeter(label="meter"))
    g.chain(laser, meter)

    node = next(n for n in graph_to_dict(g)["nodes"] if n["id"] == "laser")
    assert node["params"] == {"power": 3.0}


def test_ui_data_is_separate_from_the_physics(tmp_path: Path) -> None:
    """Node positions must not appear anywhere a physics diff would show them,
    and a project without them must still run."""
    g = _link_graph()
    positions = {"laser": {"x": 100.0, "y": 200.0}, "fiber": {"x": 300.0, "y": 200.0}}
    path = save(g, tmp_path / "link.oosim", ui=positions)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert ui_from_dict(data) == positions
    for node in data["nodes"]:
        assert "x" not in node.get("params", {})

    # Strip the UI section entirely: the project still loads and still runs.
    for node in data["nodes"]:
        node.pop("ui", None)
    stripped = tmp_path / "headless.oosim"
    stripped.write_text(json.dumps(data), encoding="utf-8")

    reloaded = load(stripped)
    analyzer = next(c for c in reloaded.components if isinstance(c, BERAnalyzer))
    assert reloaded.run()[analyzer].q_factor > 0


# --------------------------------------------------------------------------
# Rejecting broken files, by name
# --------------------------------------------------------------------------


def test_a_file_without_a_schema_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.oosim"
    path.write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")
    with pytest.raises(ProjectError, match="no schema_version"):
        load(path)


def test_a_future_schema_version_is_rejected(tmp_path: Path) -> None:
    """Silently reading a newer file would be worse than refusing: the fields it
    means differently would be interpreted with today's meanings."""
    path = tmp_path / "future.oosim"
    path.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION + 1, "context": {}, "nodes": []}),
        encoding="utf-8",
    )
    with pytest.raises(ProjectError, match="schema version"):
        load(path)


def test_malformed_json_is_reported_as_such(tmp_path: Path) -> None:
    path = tmp_path / "broken.oosim"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ProjectError, match="not valid JSON"):
        load(path)


def test_an_unknown_component_type_is_reported_by_name(tmp_path: Path) -> None:
    path = tmp_path / "plugin.oosim"
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "context": {
                    "bit_rate": 10e9,
                    "samples_per_symbol": 4,
                    "sequence_length": 16,
                    "seed": 0,
                    "precision": "single",
                },
                "nodes": [{"id": "x", "type": "QuantumFluxLaser", "params": {}}],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(UnknownComponentError, match="QuantumFluxLaser"):
        load(path)


def test_an_edge_to_a_missing_node_is_rejected(tmp_path: Path) -> None:
    data = graph_to_dict(_link_graph())
    data["edges"].append({"from": ["ghost", "out"], "to": ["fiber", "in"]})
    path = tmp_path / "dangling.oosim"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ProjectError, match="unknown node 'ghost'"):
        load(path)


def test_an_out_of_range_parameter_is_caught_on_load(tmp_path: Path) -> None:
    """Validation is not skipped just because the values came from a file."""
    data = graph_to_dict(_link_graph())
    next(n for n in data["nodes"] if n["id"] == "laser")["params"]["wavelength"] = 50.0
    path = tmp_path / "invalid.oosim"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ProjectError, match="below the minimum"):
        load(path)


def test_port_type_checking_still_applies_on_load(tmp_path: Path) -> None:
    data = graph_to_dict(_link_graph())
    data["edges"] = [{"from": ["prbs", "out"], "to": ["fiber", "in"]}]
    path = tmp_path / "mistyped.oosim"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ProjectError, match="port types differ"):
        load(path)
