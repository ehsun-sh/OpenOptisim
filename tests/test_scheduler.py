"""Tests for graph validation, scheduling, and result retention."""

from __future__ import annotations

import pytest

from oosim import CycleError, Graph, GraphError, PortType, SimulationContext
from oosim.components import Attenuator, CWLaser, Fiber, PowerMeter


@pytest.fixture
def ctx() -> SimulationContext:
    return SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=16)


# --------------------------------------------------------------------------
# Validation — every one of these must fail before any component runs
# --------------------------------------------------------------------------


def test_unconnected_input_is_rejected(ctx: SimulationContext) -> None:
    g = Graph(ctx)
    g.add(CWLaser())
    fiber = g.add(Fiber())
    g.add(PowerMeter())
    with pytest.raises(GraphError, match="is not connected"):
        g.run()
    assert fiber in g.components


def test_port_type_mismatch_is_rejected_at_connect_time(ctx: SimulationContext) -> None:
    """A metric output cannot drive an optical input. Catching this at edit time
    is the whole point of typing ports."""
    g = Graph(ctx)
    meter = g.add(PowerMeter())
    fiber = g.add(Fiber())
    with pytest.raises(GraphError, match="port types differ"):
        g.connect(meter["out"], fiber["in"])


def test_an_input_takes_exactly_one_connection(ctx: SimulationContext) -> None:
    g = Graph(ctx)
    a = g.add(CWLaser(label="a"))
    b = g.add(CWLaser(label="b"))
    fiber = g.add(Fiber())
    g.connect(a, fiber)
    with pytest.raises(GraphError, match="already driven by"):
        g.connect(b, fiber)


def test_connecting_a_component_outside_the_graph_is_rejected(ctx: SimulationContext) -> None:
    g = Graph(ctx)
    laser = g.add(CWLaser())
    orphan = Fiber()
    with pytest.raises(GraphError, match="was not added to the graph"):
        g.connect(laser, orphan)


def test_duplicate_labels_are_rejected(ctx: SimulationContext) -> None:
    g = Graph(ctx)
    g.add(CWLaser(label="tx"))
    with pytest.raises(GraphError, match="duplicate component label"):
        g.add(CWLaser(label="tx"))


def test_unknown_port_name_names_the_available_ports(ctx: SimulationContext) -> None:
    fiber = Fiber()
    with pytest.raises(KeyError, match="has no port"):
        fiber["optical_in"]


def test_unknown_parameter_is_rejected_at_construction() -> None:
    with pytest.raises(TypeError, match="has no parameter"):
        CWLaser(powr=10.0)


def test_parameter_range_is_enforced() -> None:
    with pytest.raises(ValueError, match="below the minimum"):
        CWLaser(wavelength=100.0)
    with pytest.raises(ValueError, match="above the maximum"):
        CWLaser(wavelength=9000.0)


# --------------------------------------------------------------------------
# Scheduling
# --------------------------------------------------------------------------


def test_feedback_loop_is_reported_as_a_cycle(ctx: SimulationContext) -> None:
    """Loops are legitimate physics but need an explicit iteration count, so an
    implicit one is an error rather than something the scheduler guesses at."""
    g = Graph(ctx)
    a = g.add(Attenuator(label="a"))
    b = g.add(Attenuator(label="b"))
    g.connect(a, b)
    g.connect(b, a)
    with pytest.raises(CycleError, match="feedback loop"):
        g.run()


def test_components_run_in_dependency_order_not_insertion_order(
    ctx: SimulationContext,
) -> None:
    """Adding components out of order must not change the result."""
    g = Graph(ctx)
    meter = g.add(PowerMeter())
    fiber = g.add(Fiber(length=50.0, attenuation=0.2))
    laser = g.add(CWLaser(power=0.0))
    g.connect(laser, fiber)
    g.connect(fiber, meter)

    assert g.run()[meter].power_dbm == pytest.approx(-10.0, abs=1e-4)


def test_diamond_topology_feeds_both_branches(ctx: SimulationContext) -> None:
    """One output feeding two consumers must reach both of them intact."""
    g = Graph(ctx)
    laser = g.add(CWLaser(power=0.0))
    left = g.add(Attenuator(attenuation=3.0, label="left"))
    right = g.add(Attenuator(attenuation=6.0, label="right"))
    m_left = g.add(PowerMeter(label="m_left"))
    m_right = g.add(PowerMeter(label="m_right"))
    g.connect(laser, left)
    g.connect(laser, right)
    g.connect(left, m_left)
    g.connect(right, m_right)

    results = g.run()
    assert results[m_left].power_dbm == pytest.approx(-3.0, abs=1e-4)
    assert results[m_right].power_dbm == pytest.approx(-6.0, abs=1e-4)


# --------------------------------------------------------------------------
# Result retention
# --------------------------------------------------------------------------


def test_intermediate_signals_are_released_once_consumed(ctx: SimulationContext) -> None:
    """Peak memory should be the graph cut width, not the whole graph.

    The fiber output has been consumed by the meter by the time the run ends, so
    it is dropped; asking for it says so and explains how to keep it.
    """
    g = Graph(ctx)
    laser = g.add(CWLaser())
    fiber = g.add(Fiber())
    meter = g.add(PowerMeter())
    g.chain(laser, fiber, meter)

    results = g.run()
    assert meter in results
    with pytest.raises(KeyError, match="released once consumed"):
        results.port(fiber, "out")


def test_keep_retains_named_intermediates(ctx: SimulationContext) -> None:
    g = Graph(ctx)
    laser = g.add(CWLaser())
    fiber = g.add(Fiber())
    meter = g.add(PowerMeter())
    g.chain(laser, fiber, meter)

    results = g.run(keep=[fiber])
    assert results.port(fiber, "out").num_bands == 1


# --------------------------------------------------------------------------
# Component introspection — the GUI manifest is generated, never hand-written
# --------------------------------------------------------------------------


def test_manifest_is_generated_from_the_class() -> None:
    manifest = CWLaser.manifest()

    assert manifest["name"] == "CW Laser"
    assert manifest["category"] == "Optical Sources"
    assert manifest["ports"]["outputs"] == {"out": "optical"}
    assert manifest["ports"]["inputs"] == {}

    wavelength = manifest["parameters"]["wavelength"]
    assert wavelength == {
        "type": "float",
        "default": 1550.0,
        "unit": "nm",
        "min": 1200.0,
        "max": 1700.0,
        "doc": "Vacuum wavelength",
    }


def test_si_conversion_uses_the_declared_unit() -> None:
    laser = CWLaser(wavelength=1550.0, power=10.0, linewidth=100.0)
    assert laser.si("wavelength") == pytest.approx(1550e-9)
    assert laser.si("power") == pytest.approx(1e-2)
    assert laser.si("linewidth") == pytest.approx(1e5)
    value, unit = laser.display("wavelength")
    assert value == pytest.approx(1550.0)
    assert unit == "nm"


def test_automatic_labels_depend_only_on_the_graph(ctx: SimulationContext) -> None:
    """Two identically-built graphs must label their components identically.

    A component's label seeds its random stream, so if labels came from a
    process-global counter the same script would produce different noise on every
    run and nothing stochastic could be regression-tested. Building the same
    graph twice in one process is exactly the case that would break.
    """
    labels = []
    for _ in range(2):
        g = Graph(ctx)
        g.add(CWLaser())
        g.add(Fiber())
        g.add(CWLaser())
        labels.append([c.label for c in g.components])

    assert labels[0] == labels[1] == ["CWLaser1", "Fiber1", "CWLaser2"]


def test_identical_graphs_produce_identical_noise(ctx: SimulationContext) -> None:
    """The end-to-end consequence of the label rule above."""
    import numpy as np

    def build_and_run() -> np.ndarray:
        g = Graph(ctx)
        laser = g.add(CWLaser(power=0.0, linewidth=1000.0))
        meter = g.add(PowerMeter())
        g.chain(laser, meter)
        return np.asarray(g.run(keep=[laser]).port(laser, "out").bands[0].Ex)

    np.testing.assert_array_equal(build_and_run(), build_and_run())


def test_an_explicit_label_survives_being_added(ctx: SimulationContext) -> None:
    g = Graph(ctx)
    laser = g.add(CWLaser(label="tx_laser"))
    assert laser.label == "tx_laser"


def test_ports_are_per_instance_so_port_count_can_be_configured(
    ctx: SimulationContext,
) -> None:
    from oosim.components import Combiner

    two = Combiner(2)
    four = Combiner(4)
    assert set(two.inputs) == {"in0", "in1"}
    assert set(four.inputs) == {"in0", "in1", "in2", "in3"}
    assert two.inputs["in0"] is PortType.OPTICAL
