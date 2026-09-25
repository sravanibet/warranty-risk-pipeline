"""
Renders the station topology of a plant config as a directed-graph diagram,
so the Setup phase can show the user what the agent understood before they
confirm it.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "outputs" / "diagrams"


def render_topology(config: dict, out_path: Path = None) -> Path:
    stations = config.get("stations", [])

    graph = nx.DiGraph()
    for station in stations:
        uid = station.get("station_uid")
        if not uid:
            continue
        graph.add_node(uid, label=f"{uid}\n{station.get('station_name', '')}")
        for target in station.get("connected_to", []) or []:
            graph.add_edge(uid, target)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No stations to display", ha="center", va="center")
        ax.axis("off")
    else:
        try:
            pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
        except Exception:
            pos = _layered_layout(graph)

        labels = {n: graph.nodes[n].get("label", n) for n in graph.nodes}
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=2600,
                                node_color="#AED6F1", edgecolors="#2E86C1")
        nx.draw_networkx_edges(graph, pos, ax=ax, arrows=True, arrowsize=18,
                                edge_color="#5D6D7E", connectionstyle="arc3,rad=0.05")
        nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=7)
        ax.set_title(config.get("plant_name", "Plant Topology"), fontsize=12, fontweight="bold")
        ax.axis("off")

    fig.tight_layout()

    out_path = out_path or (_OUTPUT_DIR / "agent_station_topology.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _layered_layout(graph: "nx.DiGraph") -> dict:
    """Fallback layout (no graphviz binary required) — layers nodes by BFS depth."""
    roots = [n for n in graph.nodes if graph.in_degree(n) == 0] or list(graph.nodes)[:1]
    depth = {}
    frontier = [(r, 0) for r in roots]
    seen = set()
    while frontier:
        node, d = frontier.pop(0)
        if node in seen:
            continue
        seen.add(node)
        depth[node] = d
        for succ in graph.successors(node):
            frontier.append((succ, d + 1))
    for n in graph.nodes:
        depth.setdefault(n, 0)

    layers = {}
    for n, d in depth.items():
        layers.setdefault(d, []).append(n)

    pos = {}
    for d, nodes in layers.items():
        for i, n in enumerate(nodes):
            pos[n] = (d * 2.5, -i * 1.5)
    return pos
