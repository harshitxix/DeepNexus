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
        "deriv": lambda a: a * (1 - a),
    },
    "ReLU": {
        "fn": lambda z: np.maximum(0, z),
        "deriv": lambda a: (a > 0).astype(float),
    },
    "Tanh": {
        "fn": lambda z: np.tanh(z),
        "deriv": lambda a: 1 - a ** 2,
    },
    "Linear": {
        "fn": lambda z: z,
        "deriv": lambda a: np.ones_like(a),
    },
}


def apply_act(z, name):
    return ACTIVATIONS[name]["fn"](z)


def act_deriv(a, name):
    return ACTIVATIONS[name]["deriv"](a)


# ══════════════════════════════════════════════════════════════════════════════
# SOFTMAX
# ══════════════════════════════════════════════════════════════════════════════

def softmax(z):
    z_s = z - np.max(z, axis=0, keepdims=True)
    e = np.exp(z_s)
    return e / np.sum(e, axis=0, keepdims=True)


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def standardize(df_num):
    means = df_num.mean()
    stds = df_num.std().replace(0, 1)
    return (df_num - means) / stds, means, stds


def preprocess(df, num_cols, cat_cols, means=None, stds=None, dummy_cols=None):
    if num_cols:
        num_df = df[num_cols].astype(float)
        if means is None:
            num_scaled, means, stds = standardize(num_df)
        else:
            num_scaled = (num_df - means) / stds
    else:
        num_scaled = pd.DataFrame(index=df.index)

    if cat_cols:
        cat_df = df[cat_cols].astype(str)
        cat_dummies = pd.get_dummies(cat_df, drop_first=False)
        if dummy_cols is not None:
            cat_dummies = cat_dummies.reindex(columns=dummy_cols, fill_value=0)
        else:
            dummy_cols = cat_dummies.columns.tolist()
    else:
        cat_dummies = pd.DataFrame(index=df.index)

    return pd.concat([num_scaled, cat_dummies], axis=1), means, stds, dummy_cols


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def _weight_key(in_dim, hidden_sizes, out_dim):
    return "mlp_w_" + str(in_dim) + "_" + "_".join(str(h) for h in hidden_sizes) + "_" + str(out_dim)


def _make_weights(in_dim, hidden_sizes, out_dim):
    weights, prev = [], in_dim
    for h in hidden_sizes:
        scale = np.sqrt(2.0 / prev)
        weights.append((
            np.random.randn(h, prev) * scale,
            np.zeros((h, 1)),
        ))
        prev = h
    scale = np.sqrt(2.0 / prev)
    weights.append((
        np.random.randn(out_dim, prev) * scale,
        np.zeros((out_dim, 1)),
    ))
    return weights


def _get_weights(in_dim, hidden_sizes, out_dim):
    key = _weight_key(in_dim, hidden_sizes, out_dim)
    if key not in st.session_state:
        st.session_state[key] = _make_weights(in_dim, hidden_sizes, out_dim)
    return [
        (W.copy(), b.copy())
        for W, b in st.session_state[key]
    ]


# ══════════════════════════════════════════════════════════════════════════════
# FORWARD PASS
# ══════════════════════════════════════════════════════════════════════════════

def forward(X_T, weights, hidden_acts, output_act, task_type):
    """Forward pass. X_T: (n_features, n_samples)"""
    layer_A = [X_T]
    A = X_T
    for l_idx, (W, b) in enumerate(weights):
        Z = W @ A + b
        is_out = l_idx == len(weights) - 1
        if is_out:
            if task_type == "multiclass":
                A = softmax(Z)
            else:
                A = apply_act(Z, output_act)
        else:
            A = apply_act(Z, hidden_acts[l_idx])
        layer_A.append(A)
    return layer_A


# ══════════════════════════════════════════════════════════════════════════════
# BACKWARD PASS
# ══════════════════════════════════════════════════════════════════════════════

def backward(weights, layer_A, Y_T, hidden_acts, output_act, task_type, n_samples):
    """Backward pass. Y_T: (n_out, n_samples)"""
    grads = [None] * len(weights)
    y_pred = layer_A[-1]

    if task_type == "multiclass":
        dZ = (y_pred - Y_T) / n_samples
    else:
        dA = y_pred - Y_T
        dZ = dA * act_deriv(y_pred, output_act)
        dZ = dZ / n_samples

    for l_idx in range(len(weights) - 1, -1, -1):
        A_prev = layer_A[l_idx]
        W, _ = weights[l_idx]
        dW = dZ @ A_prev.T
        db = np.sum(dZ, axis=1, keepdims=True)
        grads[l_idx] = (dW, db)

        if l_idx > 0:
            dA_prev = W.T @ dZ
            dZ = dA_prev * act_deriv(layer_A[l_idx], hidden_acts[l_idx - 1])

    return grads


# ══════════════════════════════════════════════════════════════════════════════
# LOSS AND ACCURACY
# ══════════════════════════════════════════════════════════════════════════════

def compute_loss(y_pred_T, Y_T, task_type):
    y_pred = y_pred_T.T
    y_true = Y_T.T
    if task_type == "multiclass":
        return float(-np.mean(np.sum(y_true * np.log(np.clip(y_pred, 1e-9, 1)), axis=1)))
    else:
        return float(-np.mean(
            y_true * np.log(np.clip(y_pred, 1e-9, 1)) +
            (1 - y_true) * np.log(np.clip(1 - y_pred, 1e-9, 1))
        ))


def compute_accuracy(y_pred_T, Y_T, task_type):
    if task_type == "multiclass":
        pred_cls = np.argmax(y_pred_T, axis=0)
        true_cls = np.argmax(Y_T, axis=0)
    else:
        pred_cls = (y_pred_T[0] >= 0.5).astype(int)
        true_cls = Y_T[0].astype(int)
    return float(np.mean(pred_cls == true_cls)) * 100


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_loss_curve(train_losses, val_losses=None):
    fig = go.Figure()
    epochs = list(range(1, len(train_losses) + 1))
    fig.add_trace(go.Scatter(
        x=epochs, y=train_losses, mode="lines",
        line=dict(color="#2563EB", width=2),
        name="Train Loss",
        fill="tozeroy", fillcolor="rgba(37,99,235,0.06)"
    ))
    if val_losses:
        fig.add_trace(go.Scatter(
            x=epochs, y=val_losses, mode="lines",
            line=dict(color="#DC2626", width=2, dash="dash"),
            name="Val Loss"
        ))
    fig.update_layout(
        title=dict(text="Training Loss", font=dict(size=14, color="#0F172A")),
        xaxis=dict(title="Epoch", gridcolor="#E5E7EB"),
        yaxis=dict(title="Loss", gridcolor="#E5E7EB"),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"), margin=dict(t=50, b=40, l=50, r=20)
    )
    return fig


def plot_accuracy_curve(train_accs, val_accs=None):
    fig = go.Figure()
    epochs = list(range(1, len(train_accs) + 1))
    fig.add_trace(go.Scatter(
        x=epochs, y=train_accs, mode="lines",
        line=dict(color="#5227FF", width=2), name="Train Accuracy"
    ))
    if val_accs:
        fig.add_trace(go.Scatter(
            x=epochs, y=val_accs, mode="lines",
            line=dict(color="#9CA3AF", width=2, dash="dash"), name="Val Accuracy"
        ))
    fig.update_layout(
        title=dict(text="Training Accuracy", font=dict(size=14, color="#0F172A")),
        xaxis=dict(title="Epoch", gridcolor="#E5E7EB"),
        yaxis=dict(title="Accuracy (%)", gridcolor="#E5E7EB", range=[0, 105]),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"), margin=dict(t=50, b=40, l=50, r=20)
    )
    return fig


def count_parameters(layer_sizes):
    total = 0
    for i in range(len(layer_sizes) - 1):
        total += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
    return int(total)


def plot_architecture_graph(layer_sizes, layer_labels):
    fig = go.Figure()
    x_positions = np.linspace(0.08, 0.92, len(layer_sizes))
    layer_nodes = []

    for li, n_nodes in enumerate(layer_sizes):
        if n_nodes == 1:
            ys = np.array([0.5])
        else:
            ys = np.linspace(0.15, 0.85, n_nodes)
        layer_nodes.append([(x_positions[li], float(y)) for y in ys])

    for li in range(len(layer_sizes) - 1):
        left_nodes = layer_nodes[li]
        right_nodes = layer_nodes[li + 1]
        for x0, y0 in left_nodes:
            for x1, y1 in right_nodes:
                fig.add_trace(go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(color="rgba(148,163,184,0.35)", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                ))

    for li, nodes in enumerate(layer_nodes):
        xs = [p[0] for p in nodes]
        ys = [p[1] for p in nodes]
        color = "#5227FF" if li == 0 else ("#9CA3AF" if li == len(layer_nodes) - 1 else "#7C3AED")

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=20, color=color, line=dict(color="white", width=1.2)),
            hovertemplate=f"{layer_labels[li]}<extra></extra>",
            showlegend=False,
        ))

        fig.add_annotation(
            x=xs[0],
            y=0.05,
            text=f"<b>{layer_labels[li]}</b>",
            showarrow=False,
            font=dict(size=12, color="#334155"),
        )

    fig.update_layout(
        title=dict(text="Network Structure", font=dict(size=16, color="#0F172A")),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        height=360,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "mlp_trained": False, "mlp_train_losses": [], "mlp_val_losses": [],
        "mlp_train_accs": [], "mlp_val_accs": [], "mlp_weights": None,
        "mlp_task_type": None, "mlp_class_labels": None, "mlp_final_train_acc": None,
        "mlp_final_val_acc": None, "mlp_final_loss": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

def mlp_page():
    st.title("Multi-Layer Perceptron (MLP)")
    st.caption(
        "Train a neural network: forward pass, loss computation, backpropagation, weight update. "
        "Supports binary and multiclass classification with train/validation split."
    )

    _init_state()

    st.divider()
    st.subheader("Data Source")

    GATES = {
        "AND": [(0,0,0), (0,1,0), (1,0,0), (1,1,1)],
        "OR": [(0,0,0), (0,1,1), (1,0,1), (1,1,1)],
        "XOR": [(0,0,0), (0,1,1), (1,0,1), (1,1,0)],
    }

    data_source = st.radio("Source", list(GATES.keys()), horizontal=True, key="mlp_source")
    raw = GATES[data_source]
    X = np.array([[r[0], r[1]] for r in raw], dtype=float)
    Y = np.array([[r[2]] for r in raw], dtype=float)

    st.subheader(f"{data_source} Truth Table")
    st.dataframe(
        pd.DataFrame(raw, columns=["X1", "X2", "Output"]),
        hide_index=True, use_container_width=True
    )

    task_type = "binary"
    class_labels = [0, 1]

    st.divider()
    st.subheader("Network Architecture")

    in_dim = X.shape[1]
    out_dim = 1

    n_hidden_layers = st.slider("Hidden layers", 1, 3, 1, key="mlp_n_hidden")

    hidden_sizes = []
    ncols = st.columns(min(n_hidden_layers, 5))
    for l in range(n_hidden_layers):
        n = ncols[l % 5].slider(f"Layer {l+1}", 1, 16, 4, key=f"mlp_hl_{l}")
        hidden_sizes.append(n)

    layer_sizes = [in_dim] + hidden_sizes + [out_dim]
    layer_labels = ["Input"] + [f"Hidden {i+1}" for i in range(n_hidden_layers)] + ["Output"]
    total_params = count_parameters(layer_sizes)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Input", in_dim)
    s2.metric("Hidden Layers", n_hidden_layers)
    s3.metric("Output", out_dim)
    s4.metric("Parameters", total_params)

    arch_str = f"Input({in_dim}) -> " + " -> ".join([f"H{i+1}({h})" for i, h in enumerate(hidden_sizes)]) + f" -> Output({out_dim})"
    st.code(arch_str, language="text")
    st.plotly_chart(
        plot_architecture_graph(layer_sizes, layer_labels),
        use_container_width=True,
        key="mlp_arch_graph",
    )

    st.divider()
    st.subheader("Hyperparameters")

    h1, h2, h3, h4 = st.columns(4)
    with h1:
        learning_rate = st.number_input("Learning Rate", value=0.1, min_value=0.001, max_value=1.0, step=0.01, key="mlp_lr")
    with h2:
        epochs = st.slider("Epochs", 100, 5000, 1000, step=100, key="mlp_epochs")
    with h3:
        val_split = st.slider("Val split %", 0, 40, 20, step=5, key="mlp_val_split") / 100
    with h4:
        early_stop = st.number_input("Early stop patience", value=50, min_value=0, max_value=500, step=10, key="mlp_early_stop")

    h_act = st.selectbox("Hidden activation", list(ACTIVATIONS.keys()), index=0, key="mlp_h_act")
    o_act = st.selectbox("Output activation", ["Sigmoid", "Linear"], index=0, key="mlp_o_act")

    hidden_acts = [h_act] * n_hidden_layers

    st.divider()
    st.subheader("Train / Validation Split")

    n_samples = X.shape[0]
    if val_split > 0 and n_samples >= 4:
        n_val = max(1, int(n_samples * val_split))
        idx = np.random.RandomState(42).permutation(n_samples)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
        st.caption(f"Train: {len(train_idx)} samples | Val: {len(val_idx)} samples")
    else:
        X_train, Y_train = X, Y
        X_val, Y_val = None, None

    st.divider()

    btn_col, reset_col = st.columns([4, 1])
    with reset_col:
        if st.button("Reset", use_container_width=True, key="mlp_reset"):
            _init_state()
            st.rerun()

    with btn_col:
        train_clicked = st.button("Train MLP", type="primary", use_container_width=True, key="mlp_train")

    if train_clicked:
        X_tr_T = X_train.T
        Y_tr_T = Y_train.T
        X_vl_T = X_val.T if X_val is not None else None
        Y_vl_T = Y_val.T if Y_val is not None else None

        wkey = _weight_key(in_dim, hidden_sizes, out_dim)
        st.session_state[wkey] = _make_weights(in_dim, hidden_sizes, out_dim)
        weights = _get_weights(in_dim, hidden_sizes, out_dim)

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_val_loss = float("inf")
        patience_ctr = 0
        best_weights = None

        progress_bar = st.progress(0)
        status_text = st.empty()

        for epoch in range(epochs):
            layer_A = forward(X_tr_T, weights, hidden_acts, o_act, task_type)
            loss = compute_loss(layer_A[-1], Y_tr_T, task_type)
            acc = compute_accuracy(layer_A[-1], Y_tr_T, task_type)

            grads = backward(weights, layer_A, Y_tr_T, hidden_acts, o_act, task_type, X_train.shape[0])

            for l_idx, ((W, b), (dW, db)) in enumerate(zip(weights, grads)):
                weights[l_idx] = (W - learning_rate * dW, b - learning_rate * db)

            train_losses.append(loss)
            train_accs.append(acc)

            if X_vl_T is not None:
                vl_A = forward(X_vl_T, weights, hidden_acts, o_act, task_type)
                v_loss = compute_loss(vl_A[-1], Y_vl_T, task_type)
                v_acc = compute_accuracy(vl_A[-1], Y_vl_T, task_type)
                val_losses.append(v_loss)
                val_accs.append(v_acc)

                if early_stop > 0:
                    if v_loss < best_val_loss - 1e-6:
                        best_val_loss = v_loss
                        patience_ctr = 0
                        best_weights = [(W.copy(), b.copy()) for W, b in weights]
                    else:
                        patience_ctr += 1
                    if patience_ctr >= early_stop:
                        weights = best_weights
                        train_losses = train_losses[:epoch+1]
                        val_losses = val_losses[:epoch+1]
                        train_accs = train_accs[:epoch+1]
                        val_accs = val_accs[:epoch+1]
                        break

            if epoch % max(1, epochs // 50) == 0 or epoch == epochs - 1:
                progress_bar.progress(min((epoch + 1) / epochs, 1.0))
                v_str = f" | Val Acc: {v_acc:.1f}%" if X_vl_T is not None else ""
                status_text.text(f"Epoch {epoch+1}/{epochs} | Train Acc: {acc:.1f}%{v_str}")

        final_layer_A = forward(X_tr_T, weights, hidden_acts, o_act, task_type)
        final_train_acc = compute_accuracy(final_layer_A[-1], Y_tr_T, task_type)
        final_val_acc = None
        if X_vl_T is not None:
            vl_final = forward(X_vl_T, weights, hidden_acts, o_act, task_type)
            final_val_acc = compute_accuracy(vl_final[-1], Y_vl_T, task_type)

        st.session_state.mlp_trained = True
        st.session_state.mlp_train_losses = train_losses
        st.session_state.mlp_val_losses = val_losses
        st.session_state.mlp_train_accs = train_accs
        st.session_state.mlp_val_accs = val_accs
        st.session_state.mlp_weights = weights
        st.session_state.mlp_task_type = task_type
        st.session_state.mlp_class_labels = class_labels
        st.session_state.mlp_final_train_acc = final_train_acc
        st.session_state.mlp_final_val_acc = final_val_acc
        st.session_state.mlp_final_loss = train_losses[-1]

        progress_bar.empty()
        status_text.empty()
        st.success("Training complete!")

    if not st.session_state.mlp_trained:
        return

    st.divider()
    st.subheader("Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final Loss", f"{st.session_state.mlp_final_loss:.4f}")
    m2.metric("Train Accuracy", f"{st.session_state.mlp_final_train_acc:.1f}%")
    if st.session_state.mlp_final_val_acc is not None:
        m3.metric("Val Accuracy", f"{st.session_state.mlp_final_val_acc:.1f}%")
    else:
        m3.metric("Val Accuracy", "—")
    m4.metric("Epochs", len(st.session_state.mlp_train_losses))

    tab_loss, tab_acc = st.tabs(["Loss Curve", "Accuracy Curve"])

    with tab_loss:
        fig_loss = plot_loss_curve(st.session_state.mlp_train_losses, st.session_state.mlp_val_losses if st.session_state.mlp_val_losses else None)
        st.plotly_chart(fig_loss, use_container_width=True)

    with tab_acc:
        fig_acc = plot_accuracy_curve(st.session_state.mlp_train_accs, st.session_state.mlp_val_accs if st.session_state.mlp_val_accs else None)
        st.plotly_chart(fig_acc, use_container_width=True)
