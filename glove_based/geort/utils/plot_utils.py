#!/usr/bin/env python3
# Copyright (c) 2025 Wonik Robotics
#
# This software is licensed under the MIT License.
# See the LICENSE file in the project root for full license text.

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from statistics import mean, stdev
import numpy as np
from pathlib import Path


# Helper utilities for loss analysis


def _flatten_gradients(grads):
    """Flatten a list of gradient tensors into a single 1-D vector.
    grads may contain None entries (allow_unused=True in autograd.grad).
    Returns a 1-D tensor (may be empty).
    """
    vecs = []
    for g in grads:
        if g is None:
            continue
        vecs.append(g.detach().view(-1))
    if len(vecs) == 0:
        return torch.tensor([], device=grads[0].device if grads else 'cpu')
    return torch.cat(vecs)

def _compute_grad_norm_and_vec(loss_term, params, retain_graph=True):
    """Compute autograd gradients of loss_term w.r.t params and return (norm, vector).
    This uses torch.autograd.grad so it won't accumulate into .grad.
    """
    grads = torch.autograd.grad(loss_term, params, retain_graph=retain_graph, allow_unused=True)
    vec = _flatten_gradients(grads)
    norm = float(vec.norm().item()) if vec.numel() > 0 else 0.0
    return norm, vec

def draw_chamfer_loss(inp_list, tgt_list, dmat_list, nn_idx_list, fig_finger_name, human_name="unknown", robot_name="unknown", RIGHT=True, scale=1.0):
    """Render per-finger + combined chamfer visualization HTML.

    Works for any number of fingers `n` (was previously hardcoded to 4):
      - small subplots are laid out in 2 rows x ceil(n/2) cols on the left
      - the right-most column holds a "All fingers" plot spanning both rows
    `scale` is recorded in the title but not used to transform coordinates
    (caller already applies scale to the human points before computing dmat/nn).
    """
    import math
    assert len(inp_list) == len(tgt_list) == len(dmat_list) == len(nn_idx_list)
    n = len(inp_list)

    if n == 0:
        print("[warning] draw_chamfer_loss: empty input lists, nothing to plot.")
        return

    # Layout: 2 rows x (small_cols + 1) cols. Last col spans both rows for the big plot.
    small_cols = max(2, math.ceil(n / 2))
    total_cols = small_cols + 1

    specs = [
        [{"type": "scene"} for _ in range(small_cols)] + [{"type": "scene", "rowspan": 2}],
        [{"type": "scene"} for _ in range(small_cols)] + [None],
    ]

    col_titles = fig_finger_name  # config json order, length == n
    # subplot_titles must have length rows*cols (=2*total_cols).
    # Order: row1 left-to-right, then row2 left-to-right.
    subplot_titles = []
    # Row 1
    for i in range(small_cols):
        subplot_titles.append(col_titles[i] if i < n else "")
    subplot_titles.append("All fingers")
    # Row 2
    for i in range(small_cols):
        idx = small_cols + i
        subplot_titles.append(col_titles[idx] if idx < n else "")
    subplot_titles.append(None)  # covered by rowspan

    # widths: small cols share the left half, big plot gets the right half
    small_width = 0.5 / small_cols
    column_widths = [small_width] * small_cols + [0.5]

    figp = make_subplots(
        rows=2, cols=total_cols,
        specs=specs,
        subplot_titles=subplot_titles,
        column_widths=column_widths,
        row_heights=[0.5, 0.5],
        horizontal_spacing=0.04, vertical_spacing=0.06,
    )

    # mapping: small subplot index -> (row, col).
    # Fill row 1 left-to-right, then row 2 left-to-right.
    small_positions = []
    for i in range(small_cols):
        small_positions.append((1, i + 1))
    for i in range(small_cols):
        small_positions.append((2, i + 1))

    # cycle colors for arbitrary finger count
    finger_colors_pool = ['blue', 'green', 'magenta', 'orange', 'red', 'cyan', 'purple', 'olive']
    human_colors_pool  = ['darkblue', 'darkgreen', 'darkmagenta', 'darkorange', 'darkred', 'darkcyan', 'indigo', 'darkolivegreen']
    finger_colors = [finger_colors_pool[i % len(finger_colors_pool)] for i in range(n)]
    human_data_colors = [human_colors_pool[i % len(human_colors_pool)] for i in range(n)]

    # add each finger's small subplot and also collect traces for the big plot
    for idx in range(n):
        inp0 = inp_list[idx][0].numpy()
        tgt0 = tgt_list[idx][0].numpy()
        # dmat and nn_idx are unused now (heatmap removed), but keep variables to avoid break
        _ = dmat_list[idx]
        nn_idx = nn_idx_list[idx]

        # small subplot coordinates
        r, c = small_positions[idx]

        # input scatter (small)
        inp_scatter = go.Scatter3d(x=inp0[:, 0], y=inp0[:, 1], z=inp0[:, 2],
                                mode='markers',
                                marker=dict(size=3, opacity=0.6, color=human_data_colors[idx]),
                                name=f'Input ({col_titles[idx]})',
                                showlegend=False)  # legend only in combined plot
        figp.add_trace(inp_scatter, row=r, col=c)

        # target scatter (small)
        tgt_scatter = go.Scatter3d(x=tgt0[:, 0], y=tgt0[:, 1], z=tgt0[:, 2],
                                mode='markers',
                                marker=dict(size=2, opacity=0.4, color=finger_colors[idx], symbol='diamond'),
                                name=f'Target ({col_titles[idx]})',
                                showlegend=False)
        figp.add_trace(tgt_scatter, row=r, col=c)

        # connections (small)
        xs, ys, zs = [], [], []
        for j, p in enumerate(inp0):
            q = tgt0[nn_idx[j]]
            xs.extend([p[0], q[0], None])
            ys.extend([p[1], q[1], None])
            zs.extend([p[2], q[2], None])
        conn_trace = go.Scatter3d(x=xs, y=ys, z=zs, mode='lines',
                                line=dict(color='gray', width=1),
                                opacity=0.9,
                                name=f'Conn ({col_titles[idx]})',
                                showlegend=False)
        figp.add_trace(conn_trace, row=r, col=c)

        # add corresponding traces to the big combined plot (right column)
        # Input points (per-finger)
        big_inp = go.Scatter3d(x=inp0[:, 0], y=inp0[:, 1], z=inp0[:, 2],
                            mode='markers',
                            marker=dict(size=3, opacity=0.6, color=human_data_colors[idx]),
                            name=f'Input {col_titles[idx]}',
                            showlegend=True)
        figp.add_trace(big_inp, row=1, col=total_cols)

        # Target points (per-finger)
        big_tgt = go.Scatter3d(x=tgt0[:, 0], y=tgt0[:, 1], z=tgt0[:, 2],
                            mode='markers',
                            marker=dict(size=2, opacity=0.4, color=finger_colors[idx], symbol='diamond'),
                            name=f'Target {col_titles[idx]}',
                            showlegend=True)
        figp.add_trace(big_tgt, row=1, col=total_cols)

        # Optional: connections in big plot (light, optional)
        big_xs, big_ys, big_zs = [], [], []
        for j, p in enumerate(inp0):
            q = tgt0[nn_idx[j]]
            big_xs.extend([p[0], q[0], None])
            big_ys.extend([p[1], q[1], None])
            big_zs.extend([p[2], q[2], None])
        big_conn = go.Scatter3d(x=big_xs, y=big_ys, z=big_zs, mode='lines',
                                line=dict(color='gray', width=2),
                                opacity=0.9,
                                name=f'Conn {col_titles[idx]}',
                                showlegend=True)
        figp.add_trace(big_conn, row=1, col=total_cols)

    # set axis titles for all scene subplots (iterate over scene keys)
    for key in list(figp.layout):
        if key.startswith('scene'):
            try:
                figp.layout[key].update(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
            except Exception:
                # fallback safe update
                figp.layout[key].xaxis.title.text = 'X'

    # layout tuning: bigger size, legend on top, margins
    title_str = f"Chamfer Distance — left: per-finger | right: all fingers (scale={scale})"
    figp.update_layout(
        title=title_str,
        height=1000,
        width=1600,
        autosize=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0.5, xanchor='center'),
        margin=dict(t=120, r=40, l=20)
    )

    # Import get_data_root to save in correct location
    from geort.utils.path import get_data_root

    # Format: chamfer_{human_name}_{robot_name}.html
    filename = f"chamfer_{human_name}_{robot_name}.html"
    out_path = Path(get_data_root()) / filename

    figp.write_html(str(out_path), include_plotlyjs='cdn')

    print(f"[debug] Plotly debug HTML saved to: {out_path.as_posix()}")
