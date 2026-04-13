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
        "formula": "sigmoid(z) = 1/(1+e^-z)",
    },
    "ReLU": {
        "fn": lambda z: np.maximum(0, z),
        "deriv": lambda a: (a > 0).astype(float),
        "formula": "ReLU(z) = max(0, z)",
    },
    "Tanh": {
        "fn": lambda z: np.tanh(z),
        "deriv": lambda a: 1 - a ** 2,
        "formula": "tanh(z) = (e^z-e^-z)/(e^z+e^-z)",
    },
    "Linear": {
        "fn": lambda z: z,
        "deriv": lambda a: np.ones_like(a),
        "formula": "f(z) = z",
    },
    "Step": {
        "fn": lambda z: (z > 0).astype(float),
        "deriv": lambda a: np.zeros_like(a),
        "formula": "f(z) = 1 if z>0 else 0",
    },
}


def apply_activation(z, name):
    return ACTIVATIONS[name]["fn"](z)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "perc_w1": 0.5,
        "perc_w2": 0.5,
        "perc_bias": 0.0,
        "perc_activation": "Sigmoid",
        "perc_threshold": 0.5,
        "perc_decision_data": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# PERCEPTRON COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_perceptron(x1, x2, w1, w2, bias, activation):
    z = w1 * x1 + w2 * x2 + bias
    a = apply_activation(np.array([z]), activation)[0]
    return z, a


def compute_decision_boundary(w1, w2, bias, activation, x_range, y_range):
    """Generate decision boundary data for visualization."""
    x_vals = np.linspace(x_range[0], x_range[1], 200)
    y_vals = np.linspace(y_range[0], y_range[1], 200)
    Z = np.zeros((len(y_vals), len(x_vals)))
    
    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            z = w1 * x + w2 * y + bias
            a = apply_activation(np.array([z]), activation)[0]
            Z[i, j] = a
    
    return x_vals, y_vals, Z


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_decision_boundary(w1, w2, bias, activation, x_range, y_range, threshold):
    """Plot neuron output as heatmap with decision boundary."""
    x_vals, y_vals, Z = compute_decision_boundary(w1, w2, bias, activation, x_range, y_range)
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        x=x_vals, y=y_vals, z=Z,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="Output"),
    ))
    
    # Add boundary line where output = threshold
    fig.add_trace(go.Contour(
        x=x_vals, y=y_vals, z=Z,
        contours=dict(
            start=threshold,
            end=threshold,
            size=1,
            coloring="none",
            showlines=True,
        ),
        line=dict(color="Red", width=2),
        showscale=False,
        hoverinfo="skip",
    ))
    
    fig.update_layout(
        title=f"Decision Boundary (Activation: {activation})",
        xaxis=dict(title="x1", gridcolor="#E5E7EB"),
        yaxis=dict(title="x2", gridcolor="#E5E7EB"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
        height=500,
        width=600,
    )
    return fig


def plot_activation_curve(activation, z_range):
    """Plot activation function."""
    z_vals = np.linspace(z_range[0], z_range[1], 200)
    a_vals = apply_activation(z_vals, activation)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=z_vals, y=a_vals,
        mode="lines",
        line=dict(color="#5227FF", width=3),
        name=activation,
        fill="tozeroy",
        fillcolor="rgba(82, 39, 255, 0.12)",
    ))
    
    fig.update_layout(
        title=f"{activation} Activation Function",
        xaxis=dict(title="z (pre-activation)", gridcolor="#E5E7EB"),
        yaxis=dict(title="a (post-activation)", gridcolor="#E5E7EB"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
        height=400,
        showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

def perceptron_page():
    st.title("Perceptron")
    st.caption(
        "Single neuron with two inputs, weights, bias, and non-linear activation. "
        "Visualize how weights shape the decision boundary."
    )
    
    _init_state()
    
    # ══════════════════════════════════════════════════════════════════════════
    # CONTROLS
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Neuron Parameters")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        w1 = st.slider(
            "Weight 1 (w1)", -2.0, 2.0, 0.5, 0.1,
            key="perc_w1_slider"
        )
    with c2:
        w2 = st.slider(
            "Weight 2 (w2)", -2.0, 2.0, 0.5, 0.1,
            key="perc_w2_slider"
        )
    with c3:
        bias = st.slider(
            "Bias (b)", -2.0, 2.0, 0.0, 0.1,
            key="perc_bias_slider"
        )
    
    st.caption(f"Neuron equation: z = {w1:.2f}*x1 + {w2:.2f}*x2 + {bias:.2f}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # ACTIVATION & THRESHOLD
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Activation Function")
    
    a1, a2 = st.columns(2)
    with a1:
        activation = st.selectbox(
            "Select activation", list(ACTIVATIONS.keys()),
            index=0, key="perc_act_select"
        )
    
    with a2:
        threshold = st.slider(
            "Decision threshold", 0.0, 1.0, 0.5, 0.05,
            key="perc_threshold_slider",
            help="Classes are separated at this output value"
        )
    
    st.caption(f"Formula: {ACTIVATIONS[activation]['formula']}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # TEST INPUTS
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Test Input")
    
    i1, i2 = st.columns(2)
    with i1:
        x1 = st.number_input(
            "Input x1", -1.0, 1.0, 0.3, 0.1,
            format="%.2f", key="perc_x1_input"
        )
    with i2:
        x2 = st.number_input(
            "Input x2", -1.0, 1.0, 0.3, 0.1,
            format="%.2f", key="perc_x2_input"
        )
    
    # ══════════════════════════════════════════════════════════════════════════
    # COMPUTE & DISPLAY
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    
    z_val, a_val = compute_perceptron(x1, x2, w1, w2, bias, activation)
    prediction = 1 if a_val >= threshold else 0
    
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("z (pre-activation)", f"{z_val:.4f}")
    m2.metric("a (post-activation)", f"{a_val:.4f}")
    m3.metric("Threshold", f"{threshold:.4f}")
    m4.metric("Class", f"{prediction}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # VISUALIZATIONS
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Visualizations")
    
    tab_boundary, tab_activation, tab_table = st.tabs([
        "Decision Boundary", "Activation Function", "Computation"
    ])
    
    with tab_boundary:
        st.caption("Heatmap shows neuron output across input space. Red line is decision boundary.")
        x_range = (x1 - 1, x1 + 1)
        y_range = (x2 - 1, x2 + 1)
        fig_boundary = plot_decision_boundary(w1, w2, bias, activation, x_range, y_range, threshold)
        st.plotly_chart(fig_boundary, use_container_width=True, key="perc_boundary")
    
    with tab_activation:
        st.caption("How the neuron transforms pre-activation (z) to post-activation (a).")
        z_range = (-5, 5) if activation in ["Sigmoid", "Tanh"] else (-3, 3)
        fig_activation = plot_activation_curve(activation, z_range)
        st.plotly_chart(fig_activation, use_container_width=True, key="perc_activation")
    
    with tab_table:
        st.caption("Step-by-step computation breakdown.")
        
        computation_data = {
            "Step": [
                "1. Input",
                "2. Weight sum",
                "3. Add bias",
                "4. Activation",
                "5. Decision",
            ],
            "Value": [
                f"x1={x1:.4f}, x2={x2:.4f}",
                f"w1*x1 + w2*x2 = {w1*x1:.4f} + {w2*x2:.4f}",
                f"z + b = {z_val - bias:.4f} + {bias:.4f}",
                f"a = {activation}({z_val:.4f}) = {a_val:.4f}",
                f"pred = 1 if a >= {threshold:.4f} else 0 = {prediction}",
            ],
        }
        st.dataframe(
            pd.DataFrame(computation_data),
            hide_index=True,
            use_container_width=True
        )
    
    # ══════════════════════════════════════════════════════════════════════════
    # LEARNING EXAMPLE
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Logic Gate Example")
    st.caption("Try to classify AND, OR, or XOR using the sliders above.")
    
    gate = st.selectbox("Select gate", ["AND", "OR"], key="perc_gate_select")
    
    gates_data = {
        "AND": [(0,0,0), (0,1,0), (1,0,0), (1,1,1)],
        "OR": [(0,0,0), (0,1,1), (1,0,1), (1,1,1)],
    }
    
    gate_samples = gates_data[gate]
    predictions = []
    correct = 0
    
    for x1_test, x2_test, y_true in gate_samples:
        z_t, a_t = compute_perceptron(x1_test, x2_test, w1, w2, bias, activation)
        pred_t = 1 if a_t >= threshold else 0
        predictions.append(pred_t)
        if pred_t == y_true:
            correct += 1
    
    accuracy = (correct / len(gate_samples)) * 100
    
    df_gate = pd.DataFrame(gate_samples, columns=["x1", "x2", "y_true"])
    df_gate["y_pred"] = predictions
    df_gate["Correct"] = df_gate["y_true"] == df_gate["y_pred"]
    
    st.dataframe(df_gate, hide_index=True, use_container_width=True)
    st.metric("Accuracy", f"{accuracy:.1f}%")
