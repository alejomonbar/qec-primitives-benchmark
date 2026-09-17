"""Device maps, circuit partitions and depth sweeps (house style of the MCM repo)."""

from __future__ import annotations

import networkx as nx
import numpy as np

from .layout import grid_layout

ROLE_COLOURS = ["#3b6fd4", "#2f8f4e", "#8a4bc9", "#b8560f"]


def plot_device_map(G, values, pos=None, ax=None, cmap="viridis", vmin=0.0, vmax=1.0, label="",
                    node_size=420, font_size=8, missing="#d9d9d9", title=None):
    """Chip with every qubit coloured by a scalar; qubits without a value stay grey."""
    import matplotlib.pyplot as plt

    pos = grid_layout(G) if pos is None else pos
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    colours, norm = plt.get_cmap(cmap), plt.Normalize(vmin=vmin, vmax=vmax)
    fill = [colours(norm(values[q])) if q in values else missing for q in G.nodes]
    nx.draw_networkx_edges(G, pos, ax=ax, width=4, alpha=0.25, edge_color="#4d4d4d")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=fill, node_size=node_size, edgecolors="black",
                           linewidths=1.0)
    for q, c in zip(G.nodes, fill):
        r, g, b = plt.matplotlib.colors.to_rgb(c)
        ink = "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"
        nx.draw_networkx_labels(G, pos, ax=ax, labels={q: q}, font_size=font_size, font_color=ink)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    sm = plt.cm.ScalarMappable(cmap=colours, norm=norm)
    sm.set_array([])
    ax.figure.colorbar(sm, ax=ax, fraction=0.035, pad=0.02).set_label(label)
    return ax


def plot_coverage(G, instances, pos=None, ax=None, cost=None, cmap="viridis_r", node_size=420,
                  font_size=8, title=None, label="predicted error per layer"):
    """What the experiment touches, and how it is wired.

    Pale grey qubits are untouched, hollow ones only carry data, filled ones are benchmarked as
    ancillas, and the thick edges are the couplers the gadgets actually run on.  Roles are
    categorical, so they get a legend rather than a colour ramp.

    ``cost`` (``{instance: value}``, e.g. the calibration error budget per layer) is the one
    continuous quantity worth shading here: with it the ancillas are coloured by the value of
    their own instance and a colourbar is drawn, so the map shows *where* the benchmark is
    expected to struggle before it is run.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pos = grid_layout(G) if pos is None else pos
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    ancillas = {a for inst in instances for a in inst.ancillas}
    data_only = {q for inst in instances for q in inst.data_qubits} - ancillas
    unused = set(G.nodes) - ancillas - data_only
    couplers = sorted({c for inst in instances for c in inst.couplers})

    nx.draw_networkx_edges(G, pos, ax=ax, width=1.2, alpha=0.25, edge_color="#b0b0b0")
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=couplers, width=3.5, edge_color="#4d4d4d",
                           alpha=0.8)
    if unused:
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=sorted(unused), node_size=node_size * 0.5,
                               node_color="#ececec", edgecolors="#cfcfcf", linewidths=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=sorted(data_only), node_size=node_size * 0.8,
                           node_color="white", edgecolors="#4d4d4d", linewidths=1.6)

    value = {}
    for inst in instances:                       # the best (lowest) budget wins a shared ancilla
        for a in inst.ancillas:
            if cost is not None and inst in cost:
                value[a] = min(value.get(a, float("inf")), cost[inst])
    if value:
        vals = [value.get(a, max(value.values())) for a in sorted(ancillas)]
        # Error rates span decades: a chip with a few dead couplers puts ~1 next to a median of
        # ~0.02, and on a linear ramp every healthy ancilla collapses into one colour.  A log
        # scale keeps the good qubits distinguishable while still showing the broken ones apart.
        span = max(vals) / max(min(vals), 1e-12)
        norm = (plt.matplotlib.colors.LogNorm(vmin=max(min(vals), 1e-12), vmax=max(vals))
                if span > 20 else plt.Normalize(vmin=min(vals), vmax=max(vals)))
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=sorted(ancillas), node_color=vals,
                                       cmap=plt.get_cmap(cmap), node_size=node_size,
                                       edgecolors="black", linewidths=1.0)
        nodes.set_norm(norm)
        bar = ax.figure.colorbar(nodes, ax=ax, fraction=0.035, pad=0.02)
        bar.set_label(label + (" (log scale)" if span > 20 else ""))
    else:
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=sorted(ancillas), node_size=node_size,
                               node_color="#3b6fd4", edgecolors="black", linewidths=1.0)
    if font_size:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size,
                                labels={q: q for q in sorted(ancillas | data_only)})
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=10, markeredgecolor="black",
               markerfacecolor="#3b6fd4" if not value else "#777777",
               label=f"benchmarked as ancilla ({len(ancillas)})"),
        Line2D([], [], marker="o", linestyle="", markersize=9, markeredgecolor="#4d4d4d",
               markerfacecolor="white", label=f"carries data only ({len(data_only)})"),
        Line2D([], [], marker="o", linestyle="", markersize=7, markeredgecolor="#cfcfcf",
               markerfacecolor="#ececec", label=f"not used ({len(unused)})"),
        Line2D([], [], color="#4d4d4d", lw=3.5, label=f"couplers in use ({len(couplers)})"),
    ]
    # with a colourbar the right margin is taken, so the legend goes under the chip instead
    ax.legend(handles=handles, frameon=False, fontsize=max(8, font_size),
              **({"loc": "upper center", "bbox_to_anchor": (0.5, -0.01), "ncol": 2} if value else
                 {"loc": "upper left", "bbox_to_anchor": (1.02, 1.0)}))
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax


def plot_groups(G, groups, pos=None, ax=None, node_size=420, font_size=8, title=None,
                highlight=None):
    """Chip coloured by a categorical partition (IQM's feed-forward groups), with a legend.

    Group identity is a label, not a magnitude, so it gets distinct hues and a legend instead of
    a colour ramp.  ``highlight`` (instances) draws the couplers that will actually be used.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pos = grid_layout(G) if pos is None else pos
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    of_qubit = {q: g for g, qubits in groups.items() for q in qubits}
    colour = {g: ROLE_COLOURS[i % len(ROLE_COLOURS)] for i, g in enumerate(sorted(groups))}

    nx.draw_networkx_edges(G, pos, ax=ax, width=1.2, alpha=0.25, edge_color="#b0b0b0")
    if highlight:
        nx.draw_networkx_edges(G, pos, ax=ax, width=3.5, alpha=0.8, edge_color="#4d4d4d",
                               edgelist=sorted({c for inst in highlight for c in inst.couplers}))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, edgecolors="black", linewidths=1.0,
                           node_color=[colour.get(of_qubit.get(q), "#ececec") for q in G.nodes])
    if font_size:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size, font_color="white")
    handles = [Line2D([], [], marker="o", linestyle="", markersize=10, markeredgecolor="black",
                      markerfacecolor=c, label=f"group {g} ({len(groups[g])} qubits)")
               for g, c in colour.items()]
    if highlight:
        handles.append(Line2D([], [], color="#4d4d4d", lw=3.5, label="couplers in use"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
              fontsize=max(8, font_size))
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax


def plot_partition(G, batches, pos=None, ncols=2, node_size=120, figsize=None, title=None):
    """One chip per circuit: the instances that run together (ancilla filled, data hollow)."""
    import matplotlib.pyplot as plt

    pos = grid_layout(G) if pos is None else pos
    nrows = -(-len(batches) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (5 * ncols, 4.6 * nrows), squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        ax.set_aspect("equal")
        ax.axis("off")
        if k >= len(batches):
            continue
        colour = ROLE_COLOURS[k % len(ROLE_COLOURS)]
        batch = batches[k]
        nx.draw_networkx_edges(G, pos, ax=ax, width=1.0, alpha=0.3, edge_color="#b0b0b0")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size * 0.5, linewidths=0, node_color="#e0e0e0")
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[c for i in batch for c in i.couplers], width=3,
                               edge_color=colour)
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[a for i in batch for a in i.ancillas],
                               node_size=node_size, node_color=colour, edgecolors="black", linewidths=0.6)
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[d for i in batch for d in i.data_qubits],
                               node_size=node_size * 0.8, node_color="white", edgecolors=colour,
                               linewidths=1.6)
        ax.set_title(f"circuit {k + 1}: {len(batch)} instances, "
                     f"{sum(len(i.qubits) for i in batch)} qubits", fontsize=11)
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig, axes


def plot_depth_spread(results, value="r", ax=None, color="#2f6f9f", label="instances", reference=True):
    """Faint line per instance, median and IQR band, plus the noiseless and random references."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    depths = sorted({p for by in results.values() for p in by})
    for by in results.values():
        ps = sorted(by)
        ax.plot(ps, [by[p][value] for p in ps], color=color, alpha=0.1, lw=1)
    stats = [np.array([by[p][value] for by in results.values() if p in by]) for p in depths]
    ax.fill_between(depths, [np.quantile(s, 0.25) for s in stats], [np.quantile(s, 0.75) for s in stats],
                    color=color, alpha=0.25, lw=0)
    ax.plot(depths, [np.median(s) for s in stats], "o-", color=color, mec="black", ms=8,
            label=f"median of {len(results)} {label}")
    if reference and value == "r":
        first = next(iter(results.values()))
        ax.plot(depths, [np.median([by[p]["r_ideal"] for by in results.values() if p in by]) for p in depths],
                "--", color="black", lw=1.5, label="noiseless")
        rr = first[min(first)]
        ax.axhspan(rr["r_rand"] - 3 * rr["r_rand_std"], rr["r_rand"] + 3 * rr["r_rand_std"], color="gray",
                   alpha=0.3, lw=0, label="random 3σ")
    elif reference and value == "r_ovl":
        ax.axhline(0, color="gray", ls="--")
    ax.set_xlabel("LR-QAOA depth $p$")
    ax.set_ylabel({"r": "$r$", "r_ovl": r"$r_{\rm ovl}$"}.get(value, value))
    ax.set_xticks(depths)
    ax.legend(frameon=False)
    return ax
