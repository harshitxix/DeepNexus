import numpy as np
import plotly.graph_objects as go
import streamlit as st


def _apply_study_css():
    st.markdown(
        """
<style>
.study-fade {
    animation: studyFadeIn 260ms ease-out;
}
@keyframes studyFadeIn {
    from { opacity: 0; transform: translateY(-6px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""",
        unsafe_allow_html=True,
    )


def _plot_dl_use_cases():
    domains = ["Vision", "NLP", "Speech", "Healthcare", "Finance", "Robotics"]
    impact = [92, 88, 81, 76, 73, 69]

    fig = go.Figure(
        go.Bar(
            x=domains,
            y=impact,
            marker=dict(color=["#5227FF", "#7C3AED", "#9CA3AF", "#D1D5DB", "#E5E7EB", "#F3F4F6"]),
            text=[f"{v}%" for v in impact],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=dict(text="Where Deep Learning Creates Strong Impact", font=dict(size=16, color="#0F172A")),
        xaxis_title="Domain",
        yaxis_title="Relative Impact Score",
        yaxis=dict(range=[0, 100], gridcolor="#E5E7EB"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _plot_activation_shapes():
    x = np.linspace(-5, 5, 300)
    sigmoid = 1 / (1 + np.exp(-x))
    relu = np.maximum(0, x)
    tanh = np.tanh(x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=sigmoid, mode="lines", name="Sigmoid", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=x, y=relu, mode="lines", name="ReLU", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=x, y=tanh, mode="lines", name="Tanh", line=dict(width=2)))
    fig.update_layout(
        title=dict(text="Activation Functions", font=dict(size=16, color="#0F172A")),
        xaxis_title="z",
        yaxis_title="a",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
        height=330,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _plot_perceptron_decision_surface():
    x = np.linspace(-2.5, 2.5, 120)
    y = np.linspace(-2.5, 2.5, 120)
    X, Y = np.meshgrid(x, y)
    Z = 1 / (1 + np.exp(-(1.2 * X - 0.9 * Y + 0.25)))

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=x,
        y=y,
        z=Z,
        colorscale="Viridis",
        contours=dict(showlines=False),
        colorbar=dict(title="p(y=1)"),
        showscale=True,
    ))
    fig.add_trace(go.Contour(
        x=x,
        y=y,
        z=Z,
        contours=dict(start=0.5, end=0.5, size=1, coloring="none", showlines=True),
        line=dict(color="#5227FF", width=2),
        showscale=False,
    ))
    fig.update_layout(
        title=dict(text="Perceptron Decision Surface and Boundary", font=dict(size=16, color="#0F172A")),
        xaxis_title="x1",
        yaxis_title="x2",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _plot_forward_flow():
    layers = ["L1", "L2", "L3", "L4"]
    z_vals = [0.91, 0.44, 0.28, 0.13]
    a_vals = [0.71, 0.61, 0.57, 0.53]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=layers,
        y=z_vals,
        name="Pre-activation Z",
        marker_color="#5227FF",
    ))
    fig.add_trace(go.Bar(
        x=layers,
        y=a_vals,
        name="Post-activation A",
        marker_color="#9CA3AF",
    ))
    fig.update_layout(
        title=dict(text="Layer-wise Forward Values (Z vs A)", font=dict(size=16, color="#0F172A")),
        xaxis_title="Layer",
        yaxis_title="Value",
        barmode="group",
        yaxis=dict(range=[0, 1.1], gridcolor="#E5E7EB"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _plot_gradient_decay():
    depth = np.arange(1, 9)
    stable = np.exp(-0.25 * depth)
    vanishing = np.exp(-0.9 * depth)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=depth, y=stable, mode="lines+markers", name="Healthy Gradient", line=dict(width=3, color="#5227FF")))
    fig.add_trace(go.Scatter(x=depth, y=vanishing, mode="lines+markers", name="Vanishing Gradient", line=dict(width=3, color="#DC2626", dash="dash")))
    fig.update_layout(
        title=dict(text="Gradient Behavior Across Depth", font=dict(size=16, color="#0F172A")),
        xaxis_title="Layer Depth",
        yaxis_title="|Gradient|",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _plot_backprop_weight_update():
    weights = ["w1,1", "w1,2", "w1,3", "w2,1", "w2,2", "w2,3"]
    delta = [-0.000945, -0.026775, -0.213385, -0.000555, -0.11466, -0.022184]
    fig = go.Figure(go.Bar(
        x=weights,
        y=delta,
        marker=dict(color=["#DC2626" if d < 0 else "#5227FF" for d in delta]),
        text=[f"{d:.6f}" for d in delta],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="Example Weight Update Magnitudes (Delta w)", font=dict(size=16, color="#0F172A")),
        xaxis_title="Weights",
        yaxis_title="Update Value",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _plot_parameter_growth():
    widths = np.array([4, 8, 16, 32, 64])
    params = 2 * widths + widths * widths + widths + 1

    fig = go.Figure(
        go.Scatter(
            x=widths,
            y=params,
            mode="lines+markers",
            line=dict(width=3, color="#5227FF"),
            marker=dict(size=10),
        )
    )
    fig.update_layout(
        title=dict(text="MLP Parameter Growth With Hidden Width", font=dict(size=16, color="#0F172A")),
        xaxis_title="Hidden Neurons",
        yaxis_title="Approx Parameters",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _plot_mlp_train_profile():
    epochs = np.arange(1, 11)
    train_acc = np.array([0.62, 0.74, 0.80, 0.84, 0.87, 0.89, 0.90, 0.91, 0.92, 0.93])
    val_acc = np.array([0.60, 0.71, 0.77, 0.81, 0.84, 0.85, 0.86, 0.86, 0.86, 0.86])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_acc, mode="lines+markers", name="Train Accuracy", line=dict(color="#5227FF", width=3)))
    fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode="lines+markers", name="Validation Accuracy", line=dict(color="#9CA3AF", width=3, dash="dash")))
    fig.update_layout(
        title=dict(text="MLP Training Profile (Accuracy vs Epoch)", font=dict(size=16, color="#0F172A")),
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0.5, 1.0], gridcolor="#E5E7EB"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _plot_kernel_heatmap():
    kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ])

    fig = go.Figure(
        go.Heatmap(
            z=kernel,
            colorscale="RdBu",
            zmid=0,
            text=kernel,
            texttemplate="%{text}",
            textfont={"size": 14},
            showscale=True,
        )
    )
    fig.update_layout(
        title=dict(text="Edge-Detection Convolution Kernel", font=dict(size=16, color="#0F172A")),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _plot_cv_task_map():
    tasks = ["Classification", "Detection", "Segmentation"]
    complexity = [0.45, 0.72, 0.88]
    fig = go.Figure(go.Scatterpolar(
        r=complexity + [complexity[0]],
        theta=tasks + [tasks[0]],
        fill="toself",
        line=dict(color="#5227FF", width=3),
        marker=dict(color="#7C3AED", size=8),
    ))
    fig.update_layout(
        title=dict(text="Computer Vision Task Complexity Map", font=dict(size=16, color="#0F172A")),
        polar=dict(radialaxis=dict(range=[0, 1], visible=True)),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    return fig


def _plot_rnn_memory():
    t = np.arange(1, 11)
    short_memory = np.exp(-0.55 * t)
    gated_memory = np.exp(-0.20 * t)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=short_memory, mode="lines+markers", name="Vanilla RNN Memory", line=dict(width=3, color="#DC2626")))
    fig.add_trace(go.Scatter(x=t, y=gated_memory, mode="lines+markers", name="LSTM/GRU Memory", line=dict(width=3, color="#5227FF")))
    fig.update_layout(
        title=dict(text="How Memory Persists Across Time Steps", font=dict(size=16, color="#0F172A")),
        xaxis_title="Time Step",
        yaxis_title="Information Retained",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _plot_rnn_unrolling():
    t = ["t-2", "t-1", "t", "t+1"]
    state = [0.22, 0.47, 0.63, 0.71]
    fig = go.Figure(go.Scatter(
        x=t,
        y=state,
        mode="lines+markers+text",
        text=[f"h={v:.2f}" for v in state],
        textposition="top center",
        line=dict(color="#5227FF", width=3),
        marker=dict(size=11, color="#7C3AED"),
    ))
    fig.update_layout(
        title=dict(text="RNN Unrolling: Hidden State Across Time", font=dict(size=16, color="#0F172A")),
        xaxis_title="Time Step",
        yaxis_title="Hidden State Magnitude",
        yaxis=dict(range=[0, 1], gridcolor="#E5E7EB"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _render_topic(topic):
    if topic == "Perceptron":
        st.write(
            "Single Layer Perceptron (SLP) is inspired by biological neurons and is one of the earliest neural models. "
            "It takes inputs, processes them through weighted computation, and produces an output."
        )

        st.markdown("#### Biological Inspiration")
        st.markdown("- Receive signal from outside")
        st.markdown("- Process the signal and decide response")
        st.markdown("- Communicate output to next target")

        st.markdown("#### Single Layer Perceptron")
        st.write(
            "Proposed by Frank Rosenblatt (1958), perceptron is widely used as the foundational model for binary decision tasks "
            "such as logical gates (AND, OR, NOR)."
        )
        st.markdown("- Takes input from input layer")
        st.markdown("- Applies weights and sums signals")
        st.markdown("- Passes sum through nonlinear activation")

        st.latex(r"z = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b")
        st.latex(r"\hat{y} = f(z)")
        st.write("Activations can be sigmoid, tanh, or ReLU depending on task requirements.")
        st.latex(r"\hat{y}=\sigma(z)=\frac{1}{1+e^{-z}}")
        st.latex(r"\mathcal{L}_{BCE}=-\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log(\hat{y}_i)+(1-y_i)\log(1-\hat{y}_i)\right]")
        st.latex(r"w^Tx + b = 0")

        st.markdown("#### Implementation of Single-Layer Perceptron (TensorFlow)")
        st.markdown("##### Step 1: Import Libraries")
        st.code(
            "import tensorflow as tf\n"
            "from sklearn.datasets import make_classification\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.preprocessing import StandardScaler",
            language="python",
        )

        st.markdown("##### Step 2: Create and Split Dataset")
        st.code(
            "X, y = make_classification(\n"
            "    n_samples=1000,\n"
            "    n_features=2,\n"
            "    n_informative=2,\n"
            "    n_redundant=0,\n"
            "    n_repeated=0,\n"
            "    n_classes=2,\n"
            "    random_state=42\n"
            ")\n\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42\n"
            ")",
            language="python",
        )

        st.markdown("##### Step 3: Standardize Data")
        st.code(
            "scaler = StandardScaler()\n"
            "X_train = scaler.fit_transform(X_train)\n"
            "X_test = scaler.transform(X_test)",
            language="python",
        )

        st.markdown("##### Step 4: Build Model")
        st.code(
            "model = tf.keras.Sequential([\n"
            "    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(2,))\n"
            "])",
            language="python",
        )

        st.markdown("##### Step 5: Compile")
        st.code(
            "model.compile(optimizer='adam',\n"
            "              loss='binary_crossentropy',\n"
            "              metrics=['accuracy'])",
            language="python",
        )

        st.markdown("##### Step 6: Train")
        st.code(
            "history = model.fit(X_train, y_train,\n"
            "                    epochs=50,\n"
            "                    batch_size=16,\n"
            "                    validation_split=0.1,\n"
            "                    verbose=0)",
            language="python",
        )

        st.markdown("##### Step 7: Evaluate")
        st.code(
            "loss, accuracy = model.evaluate(X_test, y_test, verbose=0)\n"
            "print(f'Test Accuracy: {accuracy:.2f}')",
            language="python",
        )

        st.write("Example outcome: test accuracy around 0.88 for this simple setup.")
        st.write("Even a single-neuron model can perform well on linearly separable patterns; for richer patterns, deeper models are used.")

        st.markdown("#### Graph: Perceptron Activation Functions")
        st.write("Interpretation: compares sigmoid, ReLU, and tanh to show how output changes as input $z$ varies.")
        st.plotly_chart(_plot_activation_shapes(), use_container_width=True, key="study_perc_activation")

        st.markdown("#### Graph: Perceptron Decision Boundary")
        st.write("Interpretation: contour colors show class probability and the red line marks the 0.5 decision threshold.")
        st.plotly_chart(_plot_perceptron_decision_surface(), use_container_width=True, key="study_perc_boundary")

    elif topic == "Forward Propagation":
        st.write(
            "Forward propagation is the step-by-step process where input data moves through each layer to generate output predictions "
            "using weights, biases, and activation functions."
        )

        st.markdown("#### Key Points")
        st.markdown("- Computes intermediate values from input layer to output layer")
        st.markdown("- Each neuron applies weighted sum and activation")
        st.markdown("- Used in both training and inference (without weight update)")
        st.markdown("- Prediction quality depends on how well these transformations capture patterns")

        st.markdown("#### Working")
        st.markdown("##### 1. Input Layer")
        st.write(
            "The network receives raw features through input neurons. "
            "Data is usually normalized/standardized for more stable and faster training."
        )

        st.markdown("##### 2. Hidden Layers")
        st.write("Each hidden neuron computes linear transformation followed by nonlinearity.")
        st.latex(r"Z = W\times X + b")
        st.write("Where $W$ are weights, $X$ is input vector, and $b$ is bias.")
        st.write("Activation functions such as ReLU or sigmoid then produce outputs passed to the next layer.")

        st.markdown("##### 3. Output Layer")
        st.markdown("- Softmax: multi-class classification")
        st.markdown("- Sigmoid: binary classification")
        st.markdown("- Linear: regression")

        st.markdown("##### 4. Prediction")
        st.write(
            "The final output is compared with true values through a loss function. "
            "This error signal is then used by backpropagation to update parameters."
        )

        st.markdown("#### Mathematical Explanation")
        st.write("For a network with input layer, two hidden layers, and output layer:")
        st.latex(r"A^{[1]} = \sigma(W^{[1]}X + b^{[1]})")
        st.latex(r"A^{[n]} = \sigma(W^{[n]}A^{[n-1]} + b^{[n]})")
        st.latex(r"Y = \sigma(W^{[3]}A^{[2]} + b^{[3]})")
        st.latex(r"A^{[3]} = \sigma(\sigma(\sigma(XW^{[1]} + b^{[1]})W^{[2]} + b^{[2]})W^{[3]} + b^{[3]})")
        st.write("Here, weights determine feature importance, biases shift thresholds, and activations introduce nonlinearity.")

        st.markdown("#### Implementation Flow")
        st.markdown("##### 1. Import Libraries")
        st.code(
            "import numpy as np\nimport pandas as pd",
            language="python",
        )

        st.markdown("##### 2. Create Sample Dataset")
        st.code(
            "data = {'cgpa': [8.5, 9.2, 7.8], 'profile_score': [85, 92, 78], 'lpa': [10, 12, 8]}\n"
            "df = pd.DataFrame(data)\n"
            "X = df[['cgpa', 'profile_score']].values",
            language="python",
        )

        st.markdown("##### 3. Initialize Parameters")
        st.code(
            "def initialize_parameters():\n"
            "    np.random.seed(1)\n"
            "    W = np.random.randn(2, 1) * 0.01\n"
            "    b = np.zeros((1, 1))\n"
            "    return W, b",
            language="python",
        )

        st.markdown("##### 4. Define Forward Propagation")
        st.code(
            "def forward_propagation(X, W, b):\n"
            "    Z = np.dot(X, W) + b\n"
            "    A = 1 / (1 + np.exp(-Z))\n"
            "    return A",
            language="python",
        )

        st.markdown("##### 5. Execute")
        st.code(
            "W, b = initialize_parameters()\n"
            "A = forward_propagation(X, W, b)\n"
            "print('Final Output:', A)",
            language="python",
        )
        st.write("Output values are sigmoid probabilities in [0, 1], representing pre-training prediction scores.")
        st.markdown("#### Graph: Forward Layer Values (Z vs A)")
        st.write("Interpretation: compares pre-activation values ($Z$) and post-activation values ($A$) layer by layer.")
        st.plotly_chart(_plot_forward_flow(), use_container_width=True, key="study_fp_flow")

    elif topic == "Backward Propagation":
        st.write(
            "Backpropagation (Backward Propagation of Errors) is a core neural-network training algorithm that minimizes prediction error. "
            "It propagates errors backward, computes gradients using chain rule, and updates weights/biases iteratively."
        )

        st.markdown("#### Why Backpropagation Matters")
        st.markdown("- Efficient weight updates using loss gradients")
        st.markdown("- Scales to deep and multi-layer architectures")
        st.markdown("- Enables automated learning across epochs")

        st.markdown("#### Working of Backpropagation")
        st.markdown("##### 1. Forward Pass")
        st.write("Inputs move layer by layer, each neuron computes weighted sum plus bias, then activation.")
        st.latex(r"a_j=\sum_i w_{i,j}x_i")
        st.latex(r"o_j = f(a_j)")
        st.write("At output layer, activations like softmax/sigmoid produce predictions.")

        st.markdown("##### 2. Backward Pass")
        st.write("Error is propagated backward to compute gradient for each parameter.")
        st.latex(r"\mathrm{MSE}=(\hat{y}-y)^2")
        st.latex(r"\Delta w_{ij}=\eta\,\delta_j\,o_i")

        st.markdown("#### Worked Sigmoid Example (Conceptual)")
        st.write("For target 0.5 and predicted 0.67, error is:")
        st.latex(r"\mathrm{Error}=y_{target}-y_5=0.5-0.67=-0.17")
        st.write("Output delta (sigmoid):")
        st.latex(r"\delta_5=y_5(1-y_5)(y_{target}-y_5)=0.67(1-0.67)(-0.17)=-0.0376")
        st.write("Hidden-layer deltas are computed similarly using upstream delta and weights.")

        st.markdown("#### Gradient and Update Form")
        st.latex(r"\frac{\partial L}{\partial W^{[l]}}=\delta^{[l]}(a^{[l-1]})^T")
        st.latex(r"W^{[l]}\leftarrow W^{[l]}-\eta\frac{\partial L}{\partial W^{[l]}}")

        st.markdown("#### XOR Implementation Flow (Python)")
        st.markdown("- Define network: input=2, hidden=4, output=1")
        st.markdown("- Initialize weights and biases")
        st.markdown("- Feedforward: hidden activation -> output activation")
        st.markdown("- Backward: output error -> hidden error -> gradient updates")
        st.markdown("- Train across epochs (e.g., 10,000) with learning rate")
        st.markdown("- Evaluate predictions on XOR truth-table")
        st.code(
            "for epoch in range(epochs):\n"
            "    output = nn.feedforward(X)\n"
            "    nn.backward(X, y, learning_rate)\n"
            "    if epoch % 4000 == 0:\n"
            "        loss = np.mean(np.square(y - output))\n"
            "        print(epoch, loss)",
            language="python",
        )

        st.markdown("#### Advantages")
        st.markdown("- Beginner-friendly and widely applicable")
        st.markdown("- Flexible across feedforward/CNN/RNN setups")
        st.markdown("- Efficient, scalable, and supports generalization")

        st.markdown("#### Challenges")
        st.markdown("- Vanishing gradients in deep networks")
        st.markdown("- Exploding gradients can destabilize training")
        st.markdown("- Overfitting in overly complex models")

        st.write("Purpose: efficient end-to-end learning by iteratively reducing loss through gradient-based updates.")

        st.markdown("#### Graph: Gradient Stability Across Depth")
        st.write("Interpretation: contrasts healthy and vanishing gradients, showing why deep networks need careful initialization and activations.")
        st.plotly_chart(_plot_gradient_decay(), use_container_width=True, key="study_bp_decay")

        st.markdown("#### Graph: Backprop Weight Update Magnitudes")
        st.write("Interpretation: bar heights represent sample parameter updates ($\Delta w$), indicating how strongly each weight changes.")
        st.plotly_chart(_plot_backprop_weight_update(), use_container_width=True, key="study_bp_weight_update")

    elif topic == "Multilayer Perceptron":
        st.write(
            "Multi-Layer Perceptron (MLP) is a fully connected neural network made of input, hidden, and output layers. "
            "Its purpose is to model complex relationships between inputs and outputs."
        )

        st.markdown("#### Components of MLP")
        st.markdown("- Input layer: one neuron per input feature")
        st.markdown("- Hidden layers: one or more dense layers for feature transformation")
        st.markdown("- Output layer: generates final prediction")
        st.write("MLP is fully connected, so each neuron in one layer connects to each neuron in the next layer.")

        st.markdown("#### Working of MLP")
        st.markdown("##### 1. Forward Propagation")
        st.write("Each neuron computes weighted sum and applies a nonlinear activation.")
        st.latex(r"z = \sum_i w_i x_i + b")
        st.latex(r"a = f(z)")
        st.write("Common activations: sigmoid, ReLU, tanh.")
        st.latex(r"\sigma(z)=\frac{1}{1+e^{-z}},\quad \mathrm{ReLU}(z)=\max(0,z),\quad \tanh(z)=\frac{2}{1+e^{-2z}}-1")

        st.markdown("##### 2. Loss Function")
        st.write("Predictions are compared against true labels using a task-appropriate loss.")
        st.latex(r"\mathcal{L}_{BCE} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log(\hat{y}_i)+(1-y_i)\log(1-\hat{y}_i)\right]")
        st.latex(r"\mathcal{L}_{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2")

        st.markdown("##### 3. Backpropagation")
        st.markdown("- Compute gradients using chain rule")
        st.markdown("- Propagate error backward layer-by-layer")
        st.markdown("- Update weights to reduce loss")
        st.latex(r"w \leftarrow w - \eta\frac{\partial \mathcal{L}}{\partial w}")

        st.markdown("##### 4. Optimization")
        st.write("SGD and Adam are common optimizers.")
        st.latex(r"w \leftarrow w - \eta\frac{\partial \mathcal{L}}{\partial w}\quad \text{(SGD)}")
        st.latex(r"m_t = \beta_1m_{t-1} + (1-\beta_1)g_t")
        st.latex(r"v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2")

        st.markdown("#### Typical TensorFlow Implementation Flow")
        st.markdown("- Import TensorFlow/NumPy/Matplotlib and load dataset (e.g., MNIST)")
        st.markdown("- Normalize image data by dividing by 255")
        st.markdown("- Visualize sample inputs")
        st.markdown("- Build Sequential model: Flatten -> Dense -> Dense -> Output Dense")
        st.markdown("- Compile with Adam and sparse categorical crossentropy")
        st.markdown("- Train with validation split")
        st.markdown("- Evaluate on test set")
        st.markdown("- Plot training/validation accuracy and loss curves")
        st.code(
            "model = Sequential([\n"
            "    Flatten(input_shape=(28, 28)),\n"
            "    Dense(256, activation='sigmoid'),\n"
            "    Dense(128, activation='sigmoid'),\n"
            "    Dense(10, activation='softmax')\n"
            "])\n"
            "model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n"
            "history = model.fit(x_train, y_train, epochs=10, batch_size=2000, validation_split=0.2)",
            language="python",
        )

        st.markdown("#### Advantages")
        st.markdown("- Versatile for classification and regression")
        st.markdown("- Models nonlinear relationships")
        st.markdown("- Benefits from parallel GPU training")

        st.markdown("#### Disadvantages")
        st.markdown("- Can be computationally expensive")
        st.markdown("- Prone to overfitting without regularization")
        st.markdown("- Sensitive to feature scaling")

        st.markdown("#### Graph: MLP Parameter Growth")
        st.write("Interpretation: shows that parameter count grows quickly as hidden layer width increases.")
        st.plotly_chart(_plot_parameter_growth(), use_container_width=True, key="study_mlp_params")

        st.markdown("#### Graph: MLP Training vs Validation")
        st.write("Interpretation: compares train and validation accuracy trends to assess convergence and possible overfitting.")
        st.plotly_chart(_plot_mlp_train_profile(), use_container_width=True, key="study_mlp_train_profile")

    elif topic == "Computer Vision Concepts":
        st.write(
            "Computer Vision is a branch of AI that enables machines to interpret images and videos. "
            "It combines classical image processing and deep learning to detect objects, recognize patterns, and extract useful information."
        )

        st.markdown("#### Basics")
        st.markdown("- Introduction: goal is machine understanding of visual scenes")
        st.markdown("- Image Processing: enhancement, filtering, denoising, transformation")
        st.markdown("- Image Representation and Pixels: images as numeric grids/channels")
        st.markdown("- Image Analysis and Manipulation: extracting structure and editing content")

        st.markdown("#### Mathematical Prerequisites")
        st.markdown("##### 1. Linear Algebra")
        st.markdown("- Vectors: represent features or embeddings")
        st.markdown("- Matrices and Tensors: represent images, kernels, batches")
        st.markdown("- Eigenvalues/Eigenvectors: used in transformations and PCA-like methods")
        st.markdown("- SVD: low-rank approximations, compression, denoising")

        st.markdown("##### 2. Signal Processing")
        st.markdown("- Image filtering and convolution: local feature extraction")
        st.markdown("- DFT/FFT: frequency-domain analysis")
        st.markdown("- PCA: dimensionality reduction for visual features")
        st.latex(r"(I*K)(i,j)=\sum_m\sum_n I(i-m,j-n)K(m,n)")

        st.markdown("#### Key Concepts")
        st.markdown("##### 1. Image Transformation")
        st.write(
            "Image transformation changes the spatial or intensity structure of an image. "
            "Geometric transforms (rotate/scale/translate/warp) align viewpoints, while intensity transforms adjust brightness and dynamic range. "
            "Fourier-domain transforms help isolate periodic noise and frequency components."
        )

        st.markdown("##### 2. Image Enhancement")
        st.write(
            "Enhancement improves visual quality and downstream model performance. "
            "Histogram equalization and contrast enhancement expose hidden details, sharpening emphasizes boundaries, "
            "and color correction reduces illumination or sensor bias artifacts."
        )

        st.markdown("##### 3. Noise Reduction Techniques")
        st.write(
            "Noise reduction suppresses random pixel variation while preserving meaningful structure. "
            "Median filtering is strong for salt-and-pepper noise, bilateral filtering preserves edges during smoothing, "
            "and wavelet denoising removes multi-scale noise components."
        )

        st.markdown("##### 4. Morphological Operations")
        st.write(
            "Morphological operations treat images as shapes and apply set-like transformations. "
            "Erosion and dilation shrink/expand foreground, opening removes small noise blobs, closing fills small holes, "
            "and morphological gradient highlights object boundaries."
        )

        st.markdown("##### 5. Feature Extraction")
        st.write(
            "Feature extraction converts raw pixels into robust descriptors. "
            "Edge detectors (Canny/Sobel/LoG) find intensity discontinuities, corner detectors identify stable keypoints, "
            "and descriptors like SIFT, SURF, ORB, and HOG encode local appearance for matching or recognition."
        )

        st.markdown("#### Popular Libraries")
        st.markdown("- OpenCV: classical CV and image operations")
        st.markdown("- TensorFlow: deep learning model development")
        st.markdown("- PyTorch: research-friendly deep learning framework")
        st.markdown("- Scikit-learn: classical ML and evaluation pipelines")

        st.markdown("#### Deep Learning in Computer Vision")
        st.markdown("##### 1. Convolutional Neural Networks (CNNs)")
        st.markdown("- Convolutional layers")
        st.markdown("- Pooling layers")
        st.markdown("- Fully connected layers")

        st.markdown("##### 2. Generative Adversarial Networks (GANs)")
        st.markdown("- DCGAN")
        st.markdown("- cGAN")
        st.markdown("- CycleGAN")
        st.markdown("- SRGAN")
        st.markdown("- StyleGAN")

        st.markdown("##### 3. Variational Autoencoders (VAEs)")
        st.markdown("- VAE")
        st.markdown("- Denoising Autoencoder (DAE)")
        st.markdown("- Convolutional Autoencoder (CAE)")

        st.markdown("##### 4. Vision Transformers (ViT)")
        st.markdown("- ViT")
        st.markdown("- Swin Transformer")
        st.markdown("- CvT")

        st.markdown("##### 5. Vision-Language Models")
        st.markdown("- CLIP")
        st.markdown("- ALIGN")
        st.markdown("- BLIP")

        st.markdown("#### Applications")
        st.markdown("##### 1. Image Classification")
        st.markdown("- Methods: SVM, RandomForest, CNN, TensorFlow, PyTorch Lightning")
        st.markdown("- Types: multiclass, multilabel, zero-shot")

        st.markdown("##### 2. Object Detection")
        st.markdown("- Models: YOLO, SSD, R-CNN family (Fast/Faster/Mask)")
        st.markdown("- Core concepts: bounding box regression, IoU, RPN, NMS")
        st.latex(r"IoU=\frac{Area(B_{pred}\cap B_{gt})}{Area(B_{pred}\cup B_{gt})}")

        st.markdown("##### 3. Image Segmentation")
        st.markdown("- Methods: K-means segmentation, U-Net, Mask R-CNN")
        st.markdown("- Types: semantic segmentation, instance segmentation, panoptic segmentation")
        st.code(
            "import cv2\n"
            "img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)\n"
            "edges = cv2.Canny(img, 100, 200)\n"
            "_, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)\n"
            "contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)",
            language="python",
        )

        st.write(
            "In practice, modern pipelines combine classical preprocessing (OpenCV) with deep models "
            "for robust performance across real-world visual tasks."
        )
        st.markdown("#### Graph: Convolution Kernel Heatmap")
        st.write("Interpretation: kernel weights show how edge emphasis is achieved using a high positive center and negative neighbors.")
        st.plotly_chart(_plot_kernel_heatmap(), use_container_width=True, key="study_cv_kernel")
        st.markdown("#### Graph: CV Task Complexity Map")
        st.write("Interpretation: radar profile compares relative workflow complexity for classification, detection, and segmentation.")
        st.plotly_chart(_plot_cv_task_map(), use_container_width=True, key="study_cv_task_map")

    elif topic == "RNN & LSTM":
        st.write(
            "Recurrent Neural Networks (RNNs) are designed for sequential and temporal data. "
            "They retain information from previous steps using hidden state memory, making them useful where order and context matter."
        )

        st.markdown("#### Why RNNs")
        st.markdown("- Designed for sequential and temporal inputs")
        st.markdown("- Maintains memory of previous inputs")
        st.markdown("- Common in NLP, forecasting, and speech tasks")

        st.markdown("#### Intuition")
        st.write(
            "When predicting the next word in a sentence, we use previous words as context. "
            "RNNs work similarly by passing hidden state from one time step to the next."
        )

        st.markdown("#### Key Components")
        st.markdown("- Recurrent neurons: units with hidden state that carry sequence memory")
        st.markdown("- Unfolding (unrolling): expansion over time steps for learning dependencies")

        st.markdown("#### Core Architecture Equations")
        st.latex(r"h = \sigma(U\cdot X + W\cdot h_{t-1} + B)")
        st.latex(r"Y = O(V\cdot h + C)")
        st.latex(r"Y = f(X,h,W,U,V,B,C)")

        st.markdown("#### Hidden State Update")
        st.latex(r"h_t = f(h_{t-1}, x_t)")
        st.latex(r"h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t)")
        st.latex(r"y_t = W_{hy}h_t")

        st.markdown("#### Backpropagation Through Time (BPTT)")
        st.write("Because sequence states depend on earlier states, gradients are propagated backward through time steps.")
        st.latex(r"\frac{\partial L(\theta)}{\partial W}=\frac{\partial L(\theta)}{\partial h_3}\cdot\frac{\partial h_3}{\partial W}")
        st.latex(r"\frac{\partial L(\theta)}{\partial W}=\frac{\partial L(\theta)}{\partial h_3}\cdot\sum_{k=1}^{3}\frac{\partial h_3}{\partial h_k}\cdot\frac{\partial h_k}{\partial W}")

        st.markdown("#### RNN Input-Output Types")
        st.markdown("- One-to-One: single input to single output")
        st.markdown("- One-to-Many: single input to sequence output")
        st.markdown("- Many-to-One: sequence input to single output")
        st.markdown("- Many-to-Many: sequence input to sequence output")

        st.markdown("#### Variants")
        st.markdown("- Vanilla RNN: simple, short dependency learning")
        st.markdown("- Bidirectional RNN: forward + backward context")
        st.markdown("- LSTM: input/forget/output gates for long memory")
        st.markdown("- GRU: lighter gated version, often faster")

        st.markdown("#### RNN vs Feedforward")
        st.markdown("- Feedforward NN: one-way flow, no sequence memory")
        st.markdown("- RNN: feedback loops + hidden state memory")

        st.markdown("#### Typical Implementation Flow")
        st.markdown("- Prepare sequences and labels")
        st.markdown("- One-hot encode inputs and targets")
        st.markdown("- Build model (SimpleRNN + Dense(softmax))")
        st.markdown("- Train with categorical cross-entropy and Adam")
        st.markdown("- Generate next token/character iteratively")
        st.code(
            "model = Sequential()\n"
            "model.add(SimpleRNN(50, input_shape=(seq_length, vocab_size), activation='relu'))\n"
            "model.add(Dense(vocab_size, activation='softmax'))\n"
            "model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])\n"
            "model.fit(X_one_hot, y_one_hot, epochs=100)",
            language="python",
        )

        st.markdown("#### Advantages")
        st.markdown("- Strong sequence memory for temporal tasks")
        st.markdown("- Useful in text, speech, and forecasting")

        st.markdown("#### Limitations")
        st.markdown("- Vanishing gradients on long sequences")
        st.markdown("- Exploding gradients can destabilize training")

        st.markdown("#### Applications")
        st.markdown("- Time-series prediction")
        st.markdown("- NLP and language modeling")
        st.markdown("- Speech recognition")
        st.markdown("- Video and sequence understanding")

        st.markdown("#### Graph: RNN Memory Retention")
        st.write("Interpretation: compares memory decay in vanilla RNN against gated variants like LSTM/GRU.")
        st.plotly_chart(_plot_rnn_memory(), use_container_width=True, key="study_rnn_memory")

        st.markdown("#### Graph: RNN Unrolled Hidden State")
        st.write("Interpretation: hidden-state trajectory across time steps illustrates how temporal context is carried forward.")
        st.plotly_chart(_plot_rnn_unrolling(), use_container_width=True, key="study_rnn_unrolling")


def _render_intro():
    st.markdown("### Deep Learning and Neural Networks")
    st.write(
        "Deep learning is a machine-learning approach where a neural network learns multiple levels of representation directly from data. "
        "Each layer transforms the previous representation into a more useful one for the final task."
    )
    st.write(
        "Neural networks are parameterized functions. Training means finding parameters that minimize expected task loss over a data distribution. "
        "This is done with forward propagation, backpropagation, and iterative optimization."
    )
    st.latex(r"\hat{y} = f(x;\theta)")
    st.latex(r"\theta^* = \arg\min_{\theta}\; \mathbb{E}_{(x,y)\sim D}[L(f(x;\theta), y)]")
    st.markdown("#### Graph: Deep Learning Impact Map")
    st.write("Interpretation: highlights domains where deep learning currently delivers strong practical impact.")
    st.plotly_chart(_plot_dl_use_cases(), use_container_width=True, key="study_dl_usage")


def study_mode_page():
    _apply_study_css()

    topics = [
        "Perceptron",
        "Forward Propagation",
        "Backward Propagation",
        "Multilayer Perceptron",
        "Computer Vision Concepts",
        "RNN & LSTM",
    ]

    if "study_picker_open" not in st.session_state:
        st.session_state.study_picker_open = False
    if "study_selected_topic" not in st.session_state:
        st.session_state.study_selected_topic = None
    if "study_topic_select_intro" not in st.session_state:
        st.session_state.study_topic_select_intro = "Select module"

    selected_topic = st.session_state.study_selected_topic

    if selected_topic is not None:
        c1, c2 = st.columns([5, 1])
        with c1:
            st.subheader(selected_topic)
        with c2:
            if st.button("Choose", key="study_choose_again", use_container_width=True):
                st.session_state.study_picker_open = True

        if st.session_state.study_picker_open:
            st.markdown('<div class="study-fade">', unsafe_allow_html=True)
            new_topic = st.selectbox(
                "Select module",
                topics,
                index=topics.index(selected_topic),
                key="study_topic_select_active",
            )
            if new_topic != selected_topic:
                st.session_state.study_selected_topic = new_topic
                st.session_state.study_picker_open = False
                selected_topic = new_topic
            st.markdown("</div>", unsafe_allow_html=True)

        _render_topic(selected_topic)
        return

    st.title("Study Mode")
    st.subheader("Learn Deep, Think Neural")

    c1, c2 = st.columns([5, 1])
    with c1:
        st.subheader("Choose to study in depth")
    with c2:
        if st.button("Choose", key="study_choose_intro", use_container_width=True):
            st.session_state.study_picker_open = not st.session_state.study_picker_open

    if st.session_state.study_picker_open:
        st.markdown('<div class="study-fade">', unsafe_allow_html=True)
        intro_options = ["Select module"] + topics
        topic_intro = st.selectbox(
            "Select module",
            intro_options,
            index=0,
            key="study_topic_select_intro",
        )
        if topic_intro != "Select module":
            st.session_state.study_selected_topic = topic_intro
            st.session_state.study_picker_open = False
            st.subheader(topic_intro)
            _render_topic(topic_intro)
            return
        st.markdown("</div>", unsafe_allow_html=True)

    _render_intro()
