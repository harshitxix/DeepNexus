import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ══════════════════════════════════════════════════════════════════════════════
# ACTIVATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

ACTIVATIONS = {
    "Sigmoid": {
        "fn": lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500))),
        "formula": "sigmoid(z) = 1/(1+e^-z)",
    },
    "ReLU": {
        "fn": lambda z: np.maximum(0, z),
        "formula": "ReLU(z) = max(0, z)",
    },
    "Tanh": {
        "fn": lambda z: np.tanh(z),
        "formula": "tanh(z) = (e^z-e^-z)/(e^z+e^-z)",
    },
    "Linear": {
        "fn": lambda z: z,
        "formula": "f(z) = z",
    },
}


def apply_activation(z, name):
    return ACTIVATIONS[name]["fn"](z)


# ══════════════════════════════════════════════════════════════════════════════
# UI CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

MANUAL_MAX_NEURONS = 6
MANUAL_MAX_INPUTS = 6

# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def _weight_key(n_inputs, hidden_sizes):
    return "fp_w_" + str(n_inputs) + "_" + "_".join(str(h) for h in hidden_sizes)


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
# FORWARD PASS
# ══════════════════════════════════════════════════════════════════════════════

def forward_pass(X, weights, hidden_acts, output_act):
    """
    Forward pass through network.
    Returns layer_Z, layer_A where layer_A[0] = X.
    """
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
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "fp_computed": False,
        "fp_layer_Z": None,
        "fp_layer_A": None,
        "fp_n_inputs": 2,
        "fp_hidden_sizes": [2],
        "fp_input_vals": None,
        "fp_all_labels": None,
        "fp_hidden_acts": None,
        "fp_output_act": None,
        "fp_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_state():
    for k in [k for k in st.session_state if k.startswith("fp_")]:
        del st.session_state[k]
    _init_state()


# ══════════════════════════════════════════════════════════════════════════════
# LOG RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def render_log(placeholder, lines):
    combined = "".join(lines)
    placeholder.markdown(
        f"""<div style="background-color:#F1F5F9;color:#0F172A;
            font-family:'Courier New',monospace;font-size:12.5px;
            padding:12px 16px;border-radius:8px;height:400px;
            overflow-y:auto;white-space:pre;border:2px solid #5227FF;
        ">{combined}</div>""",
        unsafe_allow_html=True,
    )


def count_parameters(layer_sizes):
    total = 0
    for i in range(len(layer_sizes) - 1):
        total += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
    return int(total)


def plot_signal_map(all_labels, all_sizes, layer_A=None):
    """Draw a neural signal-map diagram with curved links and node values."""
    fig = go.Figure()

    layer_x = np.linspace(0.07, 0.93, len(all_sizes))
    layer_positions = []

    node_colors = []
    for li, n_nodes in enumerate(all_sizes):
        if li == 0:
            node_colors.append("#5227FF")
        elif li == len(all_sizes) - 1:
            node_colors.append("#9CA3AF")
        else:
            node_colors.append("#7C3AED")

    for li, n_nodes in enumerate(all_sizes):
        if n_nodes == 1:
            ys = np.array([0.5])
        else:
            ys = np.linspace(0.16, 0.84, n_nodes)
        layer_positions.append([(layer_x[li], float(y)) for y in ys])

    # Links between adjacent layers
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

    # Nodes with labels
    for li, nodes in enumerate(layer_positions):
        xs = [p[0] for p in nodes]
        ys = [p[1] for p in nodes]
        labels = []
        for ni in range(len(nodes)):
            if layer_A is not None:
                value = float(layer_A[li][ni, 0])
                labels.append(f"{all_labels[li][0]}{ni+1}<br>{value:.3f}")
            else:
                labels.append(f"{all_labels[li][0]}{ni+1}")

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
            font=dict(size=12, color="#0F172A"),
        )

    fig.update_layout(
        title=dict(text="Signal Map", font=dict(size=16, color="#0F172A")),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        height=430,
        margin=dict(l=15, r=15, t=48, b=18),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

def forward_propagation_page():
    st.title("Forward Propagation")
    st.caption(
        "Trace how input signals flow through each layer of the network. "
        "Compute z = W*A_prev + b, then apply activation: a = activation(z)"
    )
    
    _init_state()
    
    # ══════════════════════════════════════════════════════════════════════════
    # ARCHITECTURE
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Network Architecture")
    
    c1, c2 = st.columns(2)
    with c1:
        n_inputs = st.slider("Input features", 1, 20, 2, key="fp_n_inputs_slider")
    with c2:
        n_hidden_layers = st.slider("Hidden layers", 1, 5, 1, key="fp_n_hidden_slider")
    
    st.caption("Neurons per hidden layer:")
    hidden_sizes = []
    ncols = st.columns(min(n_hidden_layers, 5))
    for l in range(n_hidden_layers):
        n = ncols[l % 5].slider(f"Layer {l+1}", 1, 20, 2, key=f"fp_hl_{l}")
        hidden_sizes.append(n)
    
    all_sizes = [n_inputs] + hidden_sizes + [1]
    all_labels = ["Input"] + [f"Hidden {i+1}" for i in range(n_hidden_layers)] + ["Output"]

    total_params = count_parameters(all_sizes)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Input", n_inputs)
    m2.metric("Hidden Layers", n_hidden_layers)
    m3.metric("Output", 1)
    m4.metric("Parameters", total_params)

    arch_str = f"Input({n_inputs}) -> " + " -> ".join([f"H{i+1}({h})" for i, h in enumerate(hidden_sizes)]) + " -> Output(1)"
    st.code(arch_str, language="text")
    st.plotly_chart(
        plot_signal_map(all_labels, all_sizes),
        use_container_width=True,
        key="fp_arch_signal_map",
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # INPUT VALUES
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Input Values")
    
    in_cols = st.columns(min(n_inputs, 5))
    X_vals = []
    for i in range(n_inputs):
        val = in_cols[i % 4].number_input(
            f"x{i+1}", value=round(0.3 + i * 0.15, 2),
            step=0.1, format="%.2f", key=f"fp_x{i}"
        )
        X_vals.append(val)
    
    X = np.array(X_vals).reshape(-1, 1)
    
    # ══════════════════════════════════════════════════════════════════════════
    # ACTIVATION FUNCTIONS
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Activation Functions")
    
    same_act = st.checkbox("Same activation for all hidden layers", value=True, key="fp_same_act")
    hidden_acts = []
    
    if same_act:
        a1, a2 = st.columns(2)
        with a1:
            act = st.selectbox("Hidden layers", list(ACTIVATIONS.keys()), index=0, key="fp_hidden_act")
        hidden_acts = [act] * n_hidden_layers
    else:
        act_cols = st.columns(min(n_hidden_layers, 5))
        for l in range(n_hidden_layers):
            a = act_cols[l % 5].selectbox(
                f"Layer {l+1}", list(ACTIVATIONS.keys()), index=0,
                key=f"fp_act_{l}"
            )
            hidden_acts.append(a)
    
    o1, o2 = st.columns(2)
    with o1:
        output_act = st.selectbox(
            "Output layer", list(ACTIVATIONS.keys()), index=3,
            key="fp_output_act_select"
        )
    
    # ══════════════════════════════════════════════════════════════════════════
    # WEIGHTS
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Initial Weights")
    
    can_manual = (
        n_inputs <= MANUAL_MAX_INPUTS and
        all(h <= MANUAL_MAX_NEURONS for h in hidden_sizes)
    )
    
    if not can_manual:
        st.warning(
            f"Manual entry disabled. Using random weights."
        )
        mode = "Random"
    else:
        mode = st.radio("Mode", ["Random", "Manual"], horizontal=True, key="fp_weight_mode")
    
    weights = _get_weights(n_inputs, hidden_sizes)
    
    if mode == "Random":
        if st.button("Randomize Weights", key="fp_randomize"):
            st.session_state[_weight_key(n_inputs, hidden_sizes)] = _make_weights(
                n_inputs, hidden_sizes
            )
            st.rerun()
        
        weights = _get_weights(n_inputs, hidden_sizes)
        
        with st.expander("Current Weights (read-only)", expanded=False):
            for l_idx, (W, b) in enumerate(weights):
                is_out = l_idx == len(weights) - 1
                lbl = "Output" if is_out else f"Hidden {l_idx+1}"
                st.caption(f"**{lbl}** W{W.shape} b{b.shape}")
                df = pd.DataFrame(
                    W,
                    columns=[f"in{i+1}" for i in range(W.shape[1])],
                    index=[f"n{j+1}" for j in range(W.shape[0])]
                )
                df["bias"] = b.flatten()
                st.dataframe(df.round(4), use_container_width=True)
    
    else:  # Manual
        weights_manual = []
        in_sz = n_inputs
        for l_idx, h_sz in enumerate(hidden_sizes):
            W = np.zeros((h_sz, in_sz))
            b = np.zeros((h_sz, 1))
            with st.expander(f"Hidden Layer {l_idx+1} W({h_sz}x{in_sz})", expanded=True):
                for j in range(h_sz):
                    row = st.columns(in_sz + 1)
                    for i in range(in_sz):
                        W[j, i] = row[i].number_input(
                            f"W[{j+1},{i+1}]",
                            value=0.5 if i == j else 0.0,
                            min_value=-1.0, max_value=1.0,
                            step=0.1, format="%.3f",
                            key=f"fp_mw_{l_idx}_{j}_{i}"
                        )
                    b[j, 0] = row[in_sz].number_input(
                        f"b[{j+1}]", value=0.0,
                        min_value=-1.0, max_value=1.0,
                        step=0.1, format="%.3f",
                        key=f"fp_mb_{l_idx}_{j}"
                    )
            weights_manual.append((W, b))
            in_sz = h_sz
        
        W_o = np.zeros((1, in_sz))
        b_o = np.zeros((1, 1))
        with st.expander(f"Output Layer W(1x{in_sz})", expanded=True):
            out_row = st.columns(in_sz + 1)
            for j in range(in_sz):
                W_o[0, j] = out_row[j].number_input(
                    f"W_o[{j+1}]", value=1.0,
                    min_value=-1.0, max_value=1.0,
                    step=0.1, format="%.3f",
                    key=f"fp_mwo_{j}"
                )
            b_o[0, 0] = out_row[in_sz].number_input(
                "b_o", value=0.0,
                min_value=-1.0, max_value=1.0,
                step=0.1, format="%.3f", key="fp_mbo"
            )
        weights_manual.append((W_o, b_o))
        weights = weights_manual
    
    # ══════════════════════════════════════════════════════════════════════════
    # RUN FORWARD PASS
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    
    btn_col, reset_col = st.columns([4, 1])
    with reset_col:
        if st.button("Reset", use_container_width=True, key="fp_reset"):
            _reset_state()
            st.rerun()
    
    log_exp = st.expander("Computation Log", expanded=False)
    with log_exp:
        log_ph = st.empty()
    
    if st.session_state.fp_log:
        render_log(log_ph, st.session_state.fp_log)
    
    with btn_col:
        run_clicked = st.button(
            "Run Forward Propagation", type="primary", use_container_width=True,
            key="fp_run_button"
        )
    
    if run_clicked:
        log_lines = []
        
        def log(line=""):
            log_lines.append(line + "\n")
            render_log(log_ph, log_lines)
        
        log("FORWARD PROPAGATION")
        log("=" * 70)
        log("INPUTS")
        for i, v in enumerate(X_vals):
            log(f"   x{i+1} = {v:.4f}")
        log()
        
        layer_Z, layer_A = forward_pass(X, weights, hidden_acts, output_act)
        
        for l_idx, (W, b) in enumerate(weights):
            is_out = l_idx == len(weights) - 1
            act_name = output_act if is_out else hidden_acts[l_idx]
            lbl = "OUTPUT" if is_out else f"HIDDEN {l_idx+1}"
            Z = layer_Z[l_idx]
            A = layer_A[l_idx + 1]
            A_prev = layer_A[l_idx]
            
            log(f"{lbl} LAYER [activation: {act_name}]")
            if W.shape[0] <= 8 and W.shape[1] <= 8:
                for j in range(W.shape[0]):
                    terms = " + ".join([
                        f"({W[j,i]:.3f}*{A_prev[i,0]:.3f})"
                        for i in range(W.shape[1])
                    ])
                    log(f"   z{j+1}: {terms} + {b[j,0]:.3f}")
                    log(f"        = {Z[j,0]:.4f}  ->  a{j+1} = {A[j,0]:.4f}")
            else:
                log(f"   Z (first 5): {np.round(Z.flatten()[:5], 4).tolist()}")
                log(f"   A (first 5): {np.round(A.flatten()[:5], 4).tolist()}")
            log()
        
        st.session_state.fp_log = log_lines
        st.session_state.fp_computed = True
        st.session_state.fp_layer_Z = layer_Z
        st.session_state.fp_layer_A = layer_A
        st.session_state.fp_n_inputs = n_inputs
        st.session_state.fp_hidden_sizes = hidden_sizes
        st.session_state.fp_input_vals = X_vals
        st.session_state.fp_all_labels = all_labels
        st.session_state.fp_hidden_acts = hidden_acts
        st.session_state.fp_output_act = output_act
    
    # ══════════════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    if not st.session_state.fp_computed:
        return
    
    layer_A = st.session_state.fp_layer_A
    layer_Z = st.session_state.fp_layer_Z
    s_all_labels = st.session_state.fp_all_labels
    s_all_sizes = [st.session_state.fp_n_inputs] + st.session_state.fp_hidden_sizes + [1]
    
    st.divider()
    st.subheader("Results")
    
    y_pred = float(layer_A[-1][0][0])
    m1, m2, m3 = st.columns(3)
    m1.metric("Output Value", f"{y_pred:.6f}")
    m2.metric("Layers", len(s_all_labels))
    total_params = count_parameters(s_all_sizes)
    m3.metric("Total Parameters", int(total_params))

    st.subheader("Output Signal Map")
    st.plotly_chart(
        plot_signal_map(s_all_labels, s_all_sizes, layer_A=layer_A),
        use_container_width=True,
        key="fp_result_signal_map",
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # LAYER-BY-LAYER BREAKDOWN
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Layer-by-Layer Breakdown")
    
    for l_idx in range(len(layer_Z)):
        is_out = l_idx == len(layer_Z) - 1
        lbl = "Output Layer" if is_out else f"Hidden Layer {l_idx+1}"
        
        with st.expander(lbl, expanded=(l_idx == 0)):
            Z = layer_Z[l_idx]
            A = layer_A[l_idx + 1]
            A_prev = layer_A[l_idx]
            
            col_a, col_z, col_act = st.columns(3)
            
            with col_a:
                st.caption("Input A (from previous layer):")
                a_df = pd.DataFrame(
                    A_prev.flatten(),
                    columns=["Value"]
                )
                a_df.index = [f"a{i+1}" for i in range(len(a_df))]
                st.dataframe(a_df.round(4), use_container_width=True)
            
            with col_z:
                st.caption("Pre-activation Z:")
                z_df = pd.DataFrame(
                    Z.flatten(),
                    columns=["Value"]
                )
                z_df.index = [f"z{i+1}" for i in range(len(z_df))]
                st.dataframe(z_df.round(4), use_container_width=True)
            
            with col_act:
                st.caption(f"Post-activation A ({st.session_state.fp_output_act if is_out else st.session_state.fp_hidden_acts[l_idx]}):")
                a_post_df = pd.DataFrame(
                    A.flatten(),
                    columns=["Value"]
                )
                a_post_df.index = [f"a{i+1}" for i in range(len(a_post_df))]
                st.dataframe(a_post_df.round(4), use_container_width=True)
