"""
network.py

Publication-ready directed reaction pathway graph visualizer (F tensor).
Represents species nodes sized by concentration and directed edges sized and colored by allocated sub-mechanism fluxes.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.transforms import offset_copy
import networkx as nx

def plot_reaction_network(
    F: np.ndarray,
    tracked_species: List[str],
    sub_mechs: List[str],
    palette: Dict[str, str],
    time: Optional[np.ndarray] = None,
    df: Optional[pd.DataFrame] = None,
    t_span: Optional[Tuple[float, float]] = None,
    t_idx: Optional[int] = None,
    target_species: str = "NO",
    threshold: float = 0.05,
    pos: Optional[Dict[str, Tuple[float, float]]] = None,
    title: Optional[str] = "Reaction Pathway Network",
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Tuple[float, float]]]:
    """
    Plots a directed element flux graph where nodes are species (area ~ conc)
    and edges are allocated elemental mass fluxes (color ~ sub-mechanism, width ~ magnitude).
    """
    nS, _, nM, nL = F.shape
    
    # 1. Aggregate F tensor over requested slice/span
    if t_idx is not None:
        F_agg = F[:, :, :, t_idx]
        agg_desc = f"at point index {t_idx}"
        df_slice = df.iloc[[t_idx]] if df is not None else None
    elif t_span is not None and time is not None:
        mask = (time >= t_span[0]) & (time <= t_span[1])
        t_sub = time[mask]
        F_slice = F[:, :, :, mask]
        if len(t_sub) > 1:
            F_agg = np.trapz(F_slice, x=t_sub, axis=3)
        else:
            F_agg = np.zeros((nS, nS, nM))
        agg_desc = f"integrated over {t_span[0]:.2e} to {t_span[1]:.2e} s"
        df_slice = df.iloc[mask] if df is not None else None
    elif time is not None:
        F_agg = np.trapz(F, x=time, axis=3)
        agg_desc = "integrated over full trajectory"
        df_slice = df
    else:
        F_agg = np.sum(F, axis=3)
        agg_desc = "summed over all states"
        df_slice = df

    total_flux_ij = np.sum(F_agg, axis=2)
    max_flux = np.max(total_flux_ij)
    if max_flux <= 1e-30 or np.isnan(max_flux):
        raise ValueError(f"No active elemental fluxes found above zero ({agg_desc}).")

    min_flux_thresh = max_flux * threshold
    G = nx.MultiDiGraph()
    edge_max_width = 14.0

    # 2. Populate graph edges
    for i in range(nS):
        for j in range(nS):
            if i == j:
                continue
            if total_flux_ij[i, j] < min_flux_thresh:
                continue
            for k, mech in enumerate(sub_mechs):
                flux = F_agg[i, j, k]
                if flux >= min_flux_thresh:
                    G.add_node(tracked_species[j])
                    G.add_node(tracked_species[i])
                    w = max(0.8, (flux / max_flux) * edge_max_width)
                    c = palette.get(mech, "#333333")
                    G.add_edge(tracked_species[j], tracked_species[i], weight=flux, mech=mech, color=c, width=w)

    if len(G.nodes) == 0:
        raise ValueError(f"No pathways found above threshold {threshold*100:.1f}%.")

    # 3. Node sizes based on log10 concentration
    concs = {}
    if df_slice is not None:
        for node in G.nodes():
            for col_cand in [f"X_{node}", f"Y_{node}", node]:
                if col_cand in df_slice.columns:
                    concs[node] = float(df_slice[col_cand].mean())
                    break
            if node not in concs:
                concs[node] = 1e-12

    non_zero = [c for c in concs.values() if c > 1e-12]
    min_c = min(non_zero) if non_zero else 1e-12
    max_c = max(concs.values()) if concs else 1.0
    min_log = np.log10(min_c)
    max_log = np.log10(max_c)

    nodes_list = list(G.nodes())
    size_dict = {}
    node_sizes = []
    for node in nodes_list:
        c = concs.get(node, min_c)
        if c <= min_c:
            s = 1000.0
        else:
            frac = (np.log10(c) - min_log) / (max_log - min_log) if max_log > min_log else 1.0
            s = 1000.0 + 8000.0 * frac
        size_dict[node] = s
        node_sizes.append(s)

    # 4. Hierarchical layer layout
    if pos is None:
        layers = {}
        for node in nodes_list:
            if "N2" in G.nodes():
                try:
                    layers[node] = nx.shortest_path_length(G, "N2", node)
                except nx.NetworkXNoPath:
                    layers[node] = 1
            else:
                layers[node] = 1

        if target_species in G.nodes():
            max_l = max(layers.values()) if layers else 0
            layers[target_species] = max_l + 1

        layer_nodes = {}
        for node, lay in layers.items():
            layer_nodes.setdefault(lay, []).append(node)

        pos = {}
        max_y = max(layer_nodes.keys()) if layer_nodes else 1
        for lay, n_list in layer_nodes.items():
            n_list = sorted(n_list)
            y = 1.0 - 2.0 * (lay / max_y) if max_y > 0 else 0.0
            if len(n_list) == 1:
                xs = [0.0]
            else:
                xs = np.linspace(-0.8, 0.8, len(n_list))
            for idx, n in enumerate(n_list):
                pos[n] = (xs[idx], y)

    # 5. Draw graph
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 10))
    else:
        fig = ax.figure

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color="#FAFAFA", edgecolors="#333333", linewidths=2.0)

    # Node labels
    for node in nodes_list:
        is_target = (node == target_species)
        wt = "bold" if is_target else "normal"
        r_pts = np.sqrt(size_dict[node] / np.pi)
        f_size = max(7, int(min(14, r_pts * 0.7)))
        nx.draw_networkx_labels(G, pos, labels={node: node}, ax=ax, font_size=f_size, font_weight=wt, font_family="sans-serif")

    # Directed edges
    for u, v in set(G.edges()):
        edges_uv = G.get_edge_data(u, v)
        n_edges = len(edges_uv)
        has_rev = G.has_edge(v, u)

        if n_edges == 1:
            rads = [0.15 if has_rev else 0.0]
        else:
            rad_max = 0.15 * (n_edges - 1)
            rads = np.linspace(-rad_max, rad_max, n_edges)

        r_u_pts = np.sqrt(size_dict[u] / np.pi)
        r_v_pts = np.sqrt(size_dict[v] / np.pi)

        for idx, (ek, edata) in enumerate(edges_uv.items()):
            rad = rads[idx]
            cur_w = edata["width"]
            cur_arr = max(16, int(cur_w * 2.5))
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                connectionstyle=f"arc3,rad={rad}",
                edge_color=edata["color"],
                width=cur_w,
                arrowsize=cur_arr,
                arrowstyle="->",
                arrows=True,
                ax=ax,
                node_size=0,
                min_source_margin=r_u_pts + 2,
                min_target_margin=r_v_pts + 2
            )

    # Legends
    active_mechs = set(data["mech"] for _, _, data in G.edges(data=True))
    legend_handles = [
        mlines.Line2D([], [], color=palette.get(m, "black"), linewidth=6, label=m)
        for m in sub_mechs if m in active_mechs
    ]
    if legend_handles:
        leg1 = ax.legend(handles=legend_handles, title="Sub-Mechanisms", loc="upper right", frameon=True)
        ax.add_artist(leg1)

    ax.set_title(f"{title}\n({agg_desc})", fontsize=14, pad=15)
    ax.axis("off")
    fig.tight_layout()
    return fig, ax, pos
