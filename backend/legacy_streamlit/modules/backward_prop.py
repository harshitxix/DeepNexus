import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ══════════════════════════════════════════════════════════════════════════════
# UI BREAKPOINT CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DIAGRAM_MAX_NODES = 6
DIAGRAM_MAX_LAYERS = 6
MANUAL_MAX_NEURONS = 6
MANUAL_MAX_INPUTS = 6

# ══════════════════════════════════════════════════════════════════════════════
# ACTIVATION FUNCTIONS + DERIVATIVES
# ══════════════════════════════════════════════════════════════════════════════

ACTIVATIONS = {
    "Sigmoid": {
        "fn": lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500))),
        "deriv": lambda a: a * (1 - a),
        "formula": "sigmoid(z) = 1/(1+e^-z)",
        "deriv_formula": "sigmoid'(a) = a*(1-a)",
    },
    "ReLU": {
        "fn": lambda z: np.maximum(0, z),
        "deriv": lambda a: (a > 0).astype(float),
        "formula": "ReLU(z) = max(0, z)",
        "deriv_formula": "ReLU'(a) = 1 if a>0 else 0",
    },
    "Tanh": {
        "fn": lambda z: np.tanh(z),
        "deriv": lambda a: 1 - a ** 2,
        "formula": "tanh(z) = (e^z-e^-z)/(e^z+e^-z)",
        "deriv_formula": "tanh'(a) = 1 - a^2",
    },
    "Linear": {
        "fn": lambda z: z,
        "deriv": lambda a: np.ones_like(a),
        "formula": "f(z) = z",
        "deriv_formula": "f'(z) = 1",
    },
}


def apply_activation(z, name):
    return ACTIVATIONS[name]["fn"](z)


def activation_deriv(a, name):
    return ACTIVATIONS[name]["deriv"](a)


# ══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS + DERIVATIVES
# ══════════════════════════════════════════════════════════════════════════════

LOSSES = {
    "MSE": {
        "fn": lambda y_pred, y_true: 0.5 * np.mean((y_pred - y_true) ** 2),
        "deriv": lambda y_pred, y_true: y_pred - y_true,
        "formula": "L = 0.5 * (y_pred - y_true)^2",
    },
    "Binary Cross-Entropy": {
        "fn": lambda y_pred, y_true: -np.mean(
            y_true * np.log(np.clip(y_pred, 1e-9, 1)) +
            (1 - y_true) * np.log(np.clip(1 - y_pred, 1e-9, 1))
        ),
        "deriv": lambda y_pred, y_true: (
            -(y_true / np.clip(y_pred, 1e-9, 1)) +
            (1 - y_true) / np.clip(1 - y_pred, 1e-9, 1)
        ),
        "formula": "L = -(y*log(y_pred) + (1-y)*log(1-y_pred))",
    },
    "MAE": {
        "fn": lambda y_pred, y_true: np.mean(np.abs(y_pred - y_true)),
        "deriv": lambda y_pred, y_true: np.sign(y_pred - y_true),
        "formula": "L = |y_pred - y_true|",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# FORWARD PASS
# ══════════════════════════════════════════════════════════════════════════════

def forward_pass(X, weights, hidden_acts, output_act):
    """Forward pass through network."""
    layer_Z, layer_A, A_prev = [], [X], X
    for l_idx, (W, b) in enumerate(weights):
        Z = W @ A_prev + b
        act = output_act if l_idx == len(weights) - 1 else hidden_acts[l_idx]
        A = apply_activation(Z, act)
        layer_Z.append(Z)
        layer_A.append(A)
        A_prev = A
    return layer_Z, layer_A


# ══════════════════════════════════════════════════════════════════════════════
# BACKWARD PASS
# ══════════════════════════════════════════════════════════════════════════════

def backward_pass(weights, layer_A, layer_Z, y_true, hidden_acts, output_act, loss_fn):
    """Full backpropagation through all layers."""
    n_layers = len(weights)
    grads = [None] * n_layers

    y_pred = layer_A[-1]
    dL_dA_out = LOSSES[loss_fn]["deriv"](y_pred, np.array([[y_true]]))
    dA_dZ_out = activation_deriv(layer_A[-1], output_act)
    dL_dZ_out = dL_dA_out * dA_dZ_out
    dL_dW_out = dL_dZ_out @ layer_A[-2].T
    dL_db_out = dL_dZ_out.copy()
    dL_dA_prev = weights[-1][0].T @ dL_dZ_out

    grads[-1] = {
        "dL_dA": dL_dA_out,
        "dA_dZ": dA_dZ_out,
        "dL_dZ": dL_dZ_out,
        "dL_dW": dL_dW_out,
        "dL_db": dL_db_out,
        "dL_dA_prev": dL_dA_prev,
    }

    dL_dA_curr = dL_dA_prev
    for l_idx in range(n_layers - 2, -1, -1):
        act = hidden_acts[l_idx]
        dA_dZ = activation_deriv(layer_A[l_idx + 1], act)
        dL_dZ = dL_dA_curr * dA_dZ
        dL_dW = dL_dZ @ layer_A[l_idx].T
        dL_db = dL_dZ.copy()
        dL_dA_prev = weights[l_idx][0].T @ dL_dZ

        grads[l_idx] = {
            "dL_dA": dL_dA_curr,
            "dA_dZ": dA_dZ,
            "dL_dZ": dL_dZ,
            "dL_dW": dL_dW,
            "dL_db": dL_db,
            "dL_dA_prev": dL_dA_prev,
        }
        dL_dA_curr = dL_dA_prev

    return grads, dL_dA_out


# ══════════════════════════════════════════════════════════════════════════════
# GRADIENT FLOW DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════

def draw_gradient_flow(layer_sizes, layer_labels, grads, layer_A):
    """Visualize gradient magnitude flowing backwards through the network."""
    n_layers = len(layer_sizes)
    max_nodes = max(layer_sizes)
    COLLAPSE = DIAGRAM_MAX_NODES

    node_size = max(16, min(34, int(150 / max(max_nodes, 1))))
    font_size = max(7, min(10, int(node_size * 0.27)))
    y_margin = 0.06
    x_margin = 0.07

    height = max(300, max_nodes * (node_size + 12) + 100)
    width = max(420, n_layers * 155)

    layer_xs = (
        np.linspace(x_margin, 1 - x_margin, n_layers).tolist()
        if n_layers > 1 else [0.5]
    )

    grad_magnitudes = []
    grad_magnitudes.append(None)
    for l_idx in range(len(grads)):
        if grads[l_idx] is not None:
            grad_magnitudes.append(np.abs(grads[l_idx]["dL_dZ"]).flatten())
        else:
            grad_magnitudes.append(None)

    node_positions = []
    for l_idx, (lx, n_nodes) in enumerate(zip(layer_xs, layer_sizes)):
        positions = []
        if n_nodes <= COLLAPSE:
            ys = np.linspace(1 - y_margin, y_margin, n_nodes) if n_nodes > 1 else [0.5]
            for n_idx, y in enumerate(ys):
                positions.append((lx, float(y), n_idx, False))
        else:
            show_top = 3
            ys = np.linspace(1 - y_margin, y_margin, show_top + 2)
            for n_idx in range(show_top):
                positions.append((lx, float(ys[n_idx]), n_idx, False))
            positions.append((lx, float(ys[show_top]), -1, True))
            positions.append((lx, float(ys[show_top+1]), n_nodes-1, False))
        node_positions.append(positions)

    palette_hidden = [
        ("#7C3AED", "#5227FF"),
        ("#9CA3AF", "#6B7280"),
        ("#D1D5DB", "#9CA3AF"),
        ("#E5E7EB", "#D1D5DB"),
    ]

    fig = go.Figure()

    for l_idx in range(n_layers - 1):
        for (x0, y0, _, ell_s) in node_positions[l_idx]:
            for (x1, y1, _, ell_d) in node_positions[l_idx + 1]:
                if ell_s or ell_d:
                    continue
                fig.add_shape(
                    type="line",
                    x0=x0, y0=y0, x1=x1, y1=y1,
                    xref="paper", yref="paper",
                    line=dict(color="#D1D5DB", width=0.7),
                    layer="below",
                )

    for l_idx in range(n_layers - 1, 0, -1):
        x_src = layer_xs[l_idx]
        x_dst = layer_xs[l_idx - 1]
        y_mid = 0.5
        if l_idx <= len(grads) and grads[l_idx - 1] is not None:
            mag = float(np.mean(np.abs(grads[l_idx - 1]["dL_dA_prev"])))
            arrow_color = "#EF4444" if mag > 0.1 else "#F97316" if mag > 0.01 else "#FCD34D"
            fig.add_annotation(
                x=x_dst + (x_src - x_dst) * 0.35,
                y=y_mid + 0.06 * (l_idx % 2),
                ax=60,
                ay=0,
                xref="paper", yref="paper",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=2,
                arrowcolor=arrow_color,
                text=f"d={mag:.3f}",
                font=dict(size=8, color=arrow_color),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor=arrow_color,
                borderwidth=1,
            )

    for l_idx, positions in enumerate(node_positions):
        for (nx, ny, n_idx, is_ellipsis) in positions:
            if is_ellipsis:
                fig.add_trace(go.Scatter(
                    x=[nx], y=[ny], mode="text",
                    text=["..."], textfont=dict(size=14, color="#D1D5DB"),
                    showlegend=False, hoverinfo="skip",
                ))
                continue

            if l_idx == 0:
                fill, border = "#5227FF", "#3D1A99"
                label = f"x{n_idx+1}"
                val_str = ""
            elif l_idx == n_layers - 1:
                fill, border = "#9CA3AF", "#6B7280"
                label = "y"
                mag = float(np.abs(grads[-1]["dL_dZ"].flatten()[0])) if grads[-1] else 0
                val_str = f"\ndZ={mag:.3f}"
            else:
                base_fill, border = palette_hidden[(l_idx - 1) % len(palette_hidden)]
                label = f"h{n_idx+1}"
                g_arr = grad_magnitudes[l_idx]
                if g_arr is not None and n_idx < len(g_arr):
                    mag = g_arr[n_idx]
                    val_str = f"\ndZ={mag:.3f}"
                    fill = base_fill
                else:
                    fill = base_fill
                    val_str = ""

            display = f"{label}{val_str}"
            fig.add_trace(go.Scatter(
                x=[nx], y=[ny],
                mode="markers+text",
                marker=dict(
                    size=node_size,
                    color=fill,
                    line=dict(color=border, width=1.5),
                ),
                text=[display],
                textposition="middle center",
                textfont=dict(size=font_size, color="white", family="monospace"),
                showlegend=False,
                hoverinfo="skip",
            ))

    for lx, lbl in zip(layer_xs, layer_labels):
        fig.add_annotation(
            x=lx, y=-0.06,
            text=f"<b>{lbl}</b>",
            showarrow=False,
            font=dict(size=10, color="#0F172A"),
            xanchor="center",
            xref="paper", yref="paper",
        )

    fig.add_annotation(
        x=0.99, y=0.99,
        text="gradient flow ->",
        showarrow=False,
        font=dict(size=9, color="#DC2626"),
        xanchor="right",
        xref="paper", yref="paper",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#DC2626",
        borderwidth=1,
    )

    fig.update_layout(
        height=height,
        width=width,
        margin=dict(l=20, r=20, t=30, b=55),
        xaxis=dict(visible=False, range=[-0.02, 1.02]),
        yaxis=dict(visible=False, range=[-0.1, 1.12]),
        plot_bgcolor="#F9FAFB",
        paper_bgcolor="#F9FAFB",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# GRADIENT MAGNITUDE BAR CHART
# ══════════════════════════════════════════════════════════════════════════════

def plot_gradient_bars(grads, all_labels):
    """Bar chart of mean |dL/dW| per layer."""
    labels, magnitudes = [], []
    for l_idx, g in enumerate(grads):
        if g is not None:
            lbl = all_labels[l_idx + 1]
            mag = float(np.mean(np.abs(g["dL_dW"])))
            labels.append(lbl)
            magnitudes.append(mag)

    colors = ["#EF4444" if m > 1.0 else "#F59E0B" if m > 0.1 else "#3B82F6"
              for m in magnitudes]

    fig = go.Figure(go.Bar(
        x=labels, y=magnitudes,
        marker=dict(color=colors, line=dict(color="#1E293B", width=1)),
        text=[f"{m:.4f}" for m in magnitudes],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="Mean |dL/dW| per Layer", font=dict(size=14, color="#111827")),
        xaxis=dict(title="Layer", gridcolor="#E5E7EB", color="#374151"),
        yaxis=dict(title="Gradient Magnitude", gridcolor="#E5E7EB", color="#374151"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#F9FAFB",
        font=dict(color="#374151"),
        margin=dict(t=50, b=40, l=50, r=20),
        showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# LOG RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def render_log(placeholder, lines):
    combined = "".join(lines)
    placeholder.markdown(
        f"""<div style="background-color:#0e1117;color:#00ff88;
            font-family:'Courier New',monospace;font-size:12.5px;
            padding:12px 16px;border-radius:8px;height:300px;
            overflow-y:auto;white-space:pre;border:1px solid #2a2a2a;
        ">{combined}</div>""",
        unsafe_allow_html=True,
    )


def count_parameters(layer_sizes):
    total = 0
    for i in range(len(layer_sizes) - 1):
        total += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
    return int(total)


def plot_architecture_map(all_labels, all_sizes):
    fig = go.Figure()

    layer_x = np.linspace(0.07, 0.93, len(all_sizes))
    layer_positions = []

    node_colors = []
    for li in range(len(all_sizes)):
        if li == 0:
            node_colors.append("#2563eb")
        elif li == len(all_sizes) - 1:
            node_colors.append("#0f766e")
        else:
            node_colors.append("#7c3aed")

    for li, n_nodes in enumerate(all_sizes):
        if n_nodes == 1:
            ys = np.array([0.5])
        else:
            ys = np.linspace(0.16, 0.84, n_nodes)
        layer_positions.append([(layer_x[li], float(y)) for y in ys])

    for li in range(len(all_sizes) - 1):
        left_nodes = layer_positions[li]
        right_nodes = layer_positions[li + 1]
        for x0, y0 in left_nodes:
            for x1, y1 in right_nodes:
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0 + (0.08 if y1 >= y0 else -0.08)
                fig.add_trace(go.Scatter(
                    x=[x0, cx, x1],
                    y=[y0, cy, y1],
                    mode="lines",
                    line=dict(color="rgba(100,116,139,0.28)", width=1.2),
                    hoverinfo="skip",
                    showlegend=False,
                ))

    for li, nodes in enumerate(layer_positions):
        xs = [p[0] for p in nodes]
        ys = [p[1] for p in nodes]
        labels = [f"{all_labels[li][0]}{ni+1}" for ni in range(len(nodes))]

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            marker=dict(
                size=38,
                color=node_colors[li],
                line=dict(color="rgba(255,255,255,0.85)", width=1.4),
            ),
            text=labels,
            textposition="middle center",
            textfont=dict(color="white", family="Courier New", size=11),
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

        fig.add_annotation(
            x=xs[0],
            y=0.05,
            text=f"<b>{all_labels[li]}</b>",
            showarrow=False,
            font=dict(size=12, color="#1f2937"),
        )

    fig.update_layout(
        title="Architecture Map",
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        height=430,
        margin=dict(l=15, r=15, t=48, b=18),
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#f8fafc",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "bp_log": [],
        "bp_computed": False,
        "bp_grads": None,
        "bp_layer_Z": None,
        "bp_layer_A": None,
        "bp_weights_old": None,
        "bp_weights_new": None,
        "bp_loss": None,
        "bp_n_inputs": 2,
        "bp_result_n_inputs": 2,
        "bp_hidden_sizes": [2],
        "bp_input_vals": None,
        "bp_y_true": 1.0,
        "bp_all_labels": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_state():
    for k in [k for k in st.session_state if k.startswith("bp_")]:
        del st.session_state[k]
    _init_state()


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def _weight_key(n_inputs, hidden_sizes):
    return "bp_w_" + str(n_inputs) + "_" + "_".join(str(h) for h in hidden_sizes)


def _make_weights(n_inputs, hidden_sizes):
    weights, in_sz = [], n_inputs
    for h in hidden_sizes:
        weights.append((
            np.random.uniform(-1, 1, (h, in_sz)),
            np.random.uniform(-1, 1, (h, 1)),
        ))
        in_sz = h
    weights.append((
        np.random.uniform(-1, 1, (1, in_sz)),
        np.array([[np.random.uniform(-1, 1)]]),
    ))
    return weights


def _get_weights(n_inputs, hidden_sizes):
    key = _weight_key(n_inputs, hidden_sizes)
    if key not in st.session_state:
        st.session_state[key] = _make_weights(n_inputs, hidden_sizes)
    return st.session_state[key]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE - HIGHLY CONDENSED
# ══════════════════════════════════════════════════════════════════════════════

def backward_propagation_page():
    st.title("Backward Propagation")
    st.caption(
        "How networks compute gradients and update weights using chain rule. "
        "dL/dW = dL/dZ * A_prev.T | dL/dZ = dL/dA * activation'(Z) | W_new = W_old - lr * dL/dW"
    )

    _init_state()

    st.divider()
    st.subheader("Network Architecture")

    c1, c2 = st.columns(2)
    with c1:
        n_inputs = st.slider("Input features", 1, 20, 2, key="bp_n_inputs")
    with c2:
        n_hidden_layers = st.slider("Hidden layers", 1, 5, 1, key="bp_n_hidden")

    st.caption("Neurons per hidden layer:")
    hidden_sizes = []
    ncols = st.columns(min(n_hidden_layers, 5))
    for l in range(n_hidden_layers):
        n = ncols[l % 5].slider(f"Layer {l+1}", 1, 20, 2, key=f"bp_hl_{l}")
        hidden_sizes.append(n)

    all_sizes = [n_inputs] + hidden_sizes + [1]
    all_labels = ["Input"] + [f"Hidden {i+1}" for i in range(n_hidden_layers)] + ["Output"]
    total_params = count_parameters(all_sizes)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Input", n_inputs)
    s2.metric("Hidden Layers", n_hidden_layers)
    s3.metric("Output", 1)
    s4.metric("Parameters", total_params)

    arch_str = f"Input({n_inputs}) -> " + " -> ".join([f"H{i+1}({h})" for i, h in enumerate(hidden_sizes)]) + " -> Output(1)"
    st.code(arch_str, language="text")
    st.plotly_chart(
        plot_architecture_map(all_labels, all_sizes),
        use_container_width=True,
        key="bp_arch_map",
    )

    st.divider()
    st.subheader("Input and Target")

    in_cols = st.columns(min(n_inputs + 1, 5))
    X_vals = []
    for i in range(n_inputs):
        val = in_cols[i % 4].number_input(
            f"x{i+1}", value=round(0.3 + i * 0.15, 2),
            step=0.1, format="%.2f", key=f"bp_x{i}"
        )
        X_vals.append(val)

    y_true = in_cols[min(n_inputs, 4)].number_input(
        "Target y", value=1.0, step=0.1, format="%.2f", key="bp_ytrue"
    )
    X = np.array(X_vals).reshape(-1, 1)

    st.divider()
    st.subheader("Hyperparameters and Activations")

    h1, h2, h3 = st.columns(3)
    with h1:
        learning_rate = st.number_input(
            "Learning Rate", value=0.1, min_value=0.0001, max_value=1.0,
            step=0.01, format="%.4f", key="bp_lr"
        )
    with h2:
        loss_fn = st.selectbox("Loss Function", list(LOSSES.keys()), key="bp_loss_fn")
    with h3:
        hidden_act = st.selectbox("Hidden activation", list(ACTIVATIONS.keys()), index=0, key="bp_h_act")
        output_act = st.selectbox("Output activation", list(ACTIVATIONS.keys()), index=3, key="bp_o_act")

    hidden_acts = [hidden_act] * n_hidden_layers

    st.divider()
    st.subheader("Initial Weights")

    can_manual = (n_inputs <= MANUAL_MAX_INPUTS and all(h <= MANUAL_MAX_NEURONS for h in hidden_sizes))

    if not can_manual:
        st.warning("Manual entry disabled. Using random weights.")
        mode = "Random"
    else:
        mode = st.radio("Mode", ["Random", "Manual"], horizontal=True, key="bp_weight_mode")

    weights = _get_weights(n_inputs, hidden_sizes)

    if mode == "Random":
        if st.button("Randomize", key="bp_rand"):
            st.session_state[_weight_key(n_inputs, hidden_sizes)] = _make_weights(n_inputs, hidden_sizes)
            st.rerun()

    st.divider()

    btn_col, reset_col = st.columns([4, 1])
    with reset_col:
        if st.button("Reset", use_container_width=True, key="bp_reset"):
            _reset_state()
            st.rerun()

    log_exp = st.expander("Computation Log", expanded=False)
    with log_exp:
        log_ph = st.empty()

    if st.session_state.bp_log:
        render_log(log_ph, st.session_state.bp_log)

    with btn_col:
        run_clicked = st.button("Run Backward Propagation", type="primary", use_container_width=True, key="bp_run")

    if run_clicked:
        log_lines = []

        def log(line=""):
            log_lines.append(line + "\n")
            render_log(log_ph, log_lines)

        log("FORWARD PASS")
        log("=" * 65)

        layer_Z, layer_A = forward_pass(X, weights, hidden_acts, output_act)

        y_pred = layer_A[-1][0][0]
        loss = LOSSES[loss_fn]["fn"](layer_A[-1], np.array([[y_true]]))

        log(f"Output: {y_pred:.6f}   Target: {y_true:.4f}   Loss: {loss:.6f}")
        log()
        log("BACKWARD PASS")
        log("=" * 65)

        grads, _ = backward_pass(weights, layer_A, layer_Z, y_true, hidden_acts, output_act, loss_fn)

        log("Weight updates (W_new = W_old - lr * dL/dW):")
        weights_new = []
        for l_idx, ((W, b), g) in enumerate(zip(weights, grads)):
            W_new = W - learning_rate * g["dL_dW"]
            b_new = b - learning_rate * g["dL_db"]
            weights_new.append((W_new, b_new))
            lbl = "Output" if l_idx == len(weights) - 1 else f"Hidden {l_idx+1}"
            mean_delta = float(np.mean(np.abs(W_new - W)))
            log(f"{lbl}: mean delta W = {mean_delta:.6f}")

        log()
        log("=" * 65)
        log("Done.")

        st.session_state.bp_log = log_lines
        st.session_state.bp_computed = True
        st.session_state.bp_grads = grads
        st.session_state.bp_layer_Z = layer_Z
        st.session_state.bp_layer_A = layer_A
        st.session_state.bp_weights_old = weights
        st.session_state.bp_weights_new = weights_new
        st.session_state.bp_loss = float(loss)
        st.session_state.bp_result_n_inputs = n_inputs
        st.session_state.bp_hidden_sizes = hidden_sizes
        st.session_state.bp_input_vals = X_vals
        st.session_state.bp_y_true = y_true
        st.session_state.bp_all_labels = all_labels

    if not st.session_state.bp_computed:
        return

    st.divider()
    st.subheader("Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Loss", f"{st.session_state.bp_loss:.6f}")
    m2.metric("y_pred", f"{st.session_state.bp_layer_A[-1][0][0]:.4f}")
    m3.metric("y_true", f"{st.session_state.bp_y_true:.4f}")
    m4.metric("Error", f"{st.session_state.bp_layer_A[-1][0][0] - st.session_state.bp_y_true:+.4f}")

    tab_flow, tab_grads, tab_weights = st.tabs([
        "Gradient Flow", "Gradient Magnitudes", "Weight Updates"
    ])

    with tab_flow:
        s_all_sizes = [st.session_state.bp_result_n_inputs] + st.session_state.bp_hidden_sizes + [1]
        if len(s_all_sizes) <= DIAGRAM_MAX_LAYERS:
            fig_flow = draw_gradient_flow(s_all_sizes, st.session_state.bp_all_labels, st.session_state.bp_grads, st.session_state.bp_layer_A)
            st.plotly_chart(fig_flow, use_container_width=True, key="bp_flow")

    with tab_grads:
        fig_bars = plot_gradient_bars(st.session_state.bp_grads, st.session_state.bp_all_labels)
        st.plotly_chart(fig_bars, use_container_width=True, key="bp_bars")

    with tab_weights:
        for l_idx, ((W_old, b_old), (W_new, b_new)) in enumerate(zip(st.session_state.bp_weights_old, st.session_state.bp_weights_new)):
            is_out = l_idx == len(st.session_state.bp_weights_old) - 1
            lbl = "Output Layer" if is_out else f"Hidden Layer {l_idx+1}"

            with st.expander(lbl, expanded=(l_idx == 0)):
                col_b, col_a, col_d = st.columns(3)
                col_b.caption("Before")
                col_b.dataframe(pd.DataFrame(W_old).round(4), use_container_width=True)
                col_a.caption("After")
                col_a.dataframe(pd.DataFrame(W_new).round(4), use_container_width=True)
