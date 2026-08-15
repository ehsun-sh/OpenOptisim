"""The ``.oosim`` project file: a graph, saved.

The format is versioned JSON, chosen so a project is readable, diffable, and
usable without this library — a schematic that can only be opened by the program
that wrote it is not a durable record of an experiment.

Three rules shape it:

* **Semantic data and UI data are separate.** Node positions live under ``ui``,
  so a diff shows what changed about the *physics* without a hundred lines of
  moved boxes on top of it.
* **Only explicitly-set parameters are stored.** A file records the choices its
  author made, not the defaults they accepted, which keeps files small and
  diffs legible. The trade-off is real and worth stating: if a model's default
  changes in a later release, a file that never overrode it will simulate
  slightly differently. ``oosim_version`` is recorded so that is diagnosable.
* **Names are looked up, never imported.** See :mod:`oosim.registry`.

A project file is always runnable headless: nothing here requires a GUI, and
the ``ui`` section is optional.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .component import Component
from .context import SimulationContext
from .graph import Graph, GraphError
from .registry import lookup

#: Bumped whenever the on-disk structure changes incompatibly.
SCHEMA_VERSION = 1

_CONTEXT_FIELDS = ("bit_rate", "samples_per_symbol", "sequence_length", "seed", "precision")


class ProjectError(Exception):
    """A project file could not be read."""


def graph_to_dict(graph: Graph, *, ui: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    """Serialise a graph to a JSON-compatible dictionary."""
    from . import __version__

    positions = ui or {}
    nodes = []
    for component in graph.components:
        node: dict[str, Any] = {
            "id": component.label,
            "type": component.type_name(),
            "params": dict(component._values),
        }
        config = component.structural_config()
        if config:
            node["config"] = config
        if component.label in positions:
            node["ui"] = positions[component.label]
        nodes.append(node)

    edges = [
        {"from": [source.component.label, source.name], "to": [dst_label, dst_port]}
        for (dst_label, dst_port), source in sorted(graph.edges.items())
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "oosim_version": __version__,
        "context": {field: getattr(graph.ctx, field) for field in _CONTEXT_FIELDS},
        "nodes": nodes,
        "edges": edges,
    }


def graph_from_dict(data: dict[str, Any]) -> Graph:
    """Rebuild a graph from a dictionary produced by :func:`graph_to_dict`."""
    version = data.get("schema_version")
    if version is None:
        raise ProjectError("not an oosim project: no schema_version")
    if version != SCHEMA_VERSION:
        raise ProjectError(
            f"project uses schema version {version}, this build reads version {SCHEMA_VERSION}"
        )

    try:
        context = SimulationContext(**data["context"])
    except KeyError as exc:
        raise ProjectError(f"project is missing {exc.args[0]!r}") from None
    except TypeError as exc:
        raise ProjectError(f"invalid context: {exc}") from None

    graph = Graph(context)
    for node in data.get("nodes", []):
        for field in ("id", "type"):
            if field not in node:
                raise ProjectError(f"node is missing {field!r}: {node}")
        component_class = lookup(node["type"])
        config = node.get("config", {})
        params = node.get("params", {})
        try:
            component: Component = component_class(label=node["id"], **config, **params)
        except (TypeError, ValueError) as exc:
            raise ProjectError(f"cannot build node {node['id']!r}: {exc}") from None
        graph.add(component)

    by_label = {c.label: c for c in graph.components}
    for edge in data.get("edges", []):
        try:
            (src_label, src_port), (dst_label, dst_port) = edge["from"], edge["to"]
        except (KeyError, ValueError) as exc:
            raise ProjectError(f"malformed edge {edge}: {exc}") from None
        for label in (src_label, dst_label):
            if label not in by_label:
                raise ProjectError(f"edge refers to unknown node {label!r}")
        try:
            graph.connect(by_label[src_label][src_port], by_label[dst_label][dst_port])
        except (GraphError, KeyError) as exc:
            raise ProjectError(f"cannot connect {edge}: {exc}") from None

    return graph


def ui_from_dict(data: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Node positions from a project dictionary, if it carries any."""
    return {node["id"]: node["ui"] for node in data.get("nodes", []) if "ui" in node}


def save(graph: Graph, path: str | Path, *, ui: dict[str, dict[str, float]] | None = None) -> Path:
    """Write ``graph`` to ``path`` as a ``.oosim`` file."""
    destination = Path(path)
    destination.write_text(
        json.dumps(graph_to_dict(graph, ui=ui), indent=2) + "\n", encoding="utf-8"
    )
    return destination


def load(path: str | Path) -> Graph:
    """Read a ``.oosim`` file back into a runnable graph."""
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProjectError(f"{source} is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ProjectError(f"{source} does not contain a project object")
    return graph_from_dict(data)
