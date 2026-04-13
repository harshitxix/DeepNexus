import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE DATA - SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_TEXTS = {
    "Positive": [
        "I love this movie",
        "Amazing product quality",
        "Great customer service",
        "Absolutely wonderful experience",
        "Best decision ever",
    ],
    "Negative": [
        "This is terrible",
        "Very disappointed",
        "Worst experience ever",
        "I hate this",
        "Complete waste of time",
    ],
}

VOCAB = {
    "love": 0.8, "amazing": 0.9, "great": 0.7, "wonderful": 0.85, "best": 0.8,
    "hate": -0.9, "terrible": -0.85, "disappointed": -0.7, "worst": -0.95, "waste": -0.8,
    "good": 0.6, "bad": -0.7, "excellent": 0.85, "poor": -0.8,
}


# ══════════════════════════════════════════════════════════════════════════════
# SIMPLE RNN FOR SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════

class SimpleRNN:
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1
        self.W_h = np.random.randn(hidden_dim, embedding_dim + hidden_dim) * 0.1
        self.W_o = np.random.randn(1, hidden_dim) * 0.1
        self.b_h = np.zeros((hidden_dim, 1))
        self.b_o = np.zeros((1, 1))
    
    def forward(self, sequence):
        """Forward pass through RNN."""
        h_states = []
        h_prev = np.zeros((self.hidden_dim, 1))
        
        for token_idx in sequence:
            if token_idx < len(self.embeddings):
                x_t = self.embeddings[token_idx].reshape(-1, 1)
                
                concat = np.vstack([x_t, h_prev])
                h_t = np.tanh(self.W_h @ concat + self.b_h)
                h_states.append(h_t)
                h_prev = h_t
        
        if not h_states:
            h_final = np.zeros((self.hidden_dim, 1))
        else:
            h_final = h_states[-1]
        
        output = 1 / (1 + np.exp(-np.clip(self.W_o @ h_final + self.b_o, -500, 500)))
        
        return h_states, h_final, float(output[0][0])


def tokenize_text(text, vocab):
    """Convert text to token indices."""
    words = text.lower().split()
    tokens = []
    
    for word in words:
        if word in vocab:
            tokens.append(vocab[word])
        else:
            tokens.append(np.mean(list(vocab.values())))
    
    return np.array(tokens) if tokens else np.array([0])


def text_to_sequence(text, vocab, vocab_size=20):
    """Convert text to sequence of indices."""
    word_scores = tokenize_text(text, vocab)
    
    sequence = []
    words = text.lower().split()
    for i, word in enumerate(words):
        if word in vocab:
            idx = int((vocab[word] + 1) * (vocab_size / 2))
        else:
            idx = vocab_size // 2
        
        sequence.append(min(max(idx, 0), vocab_size - 1))
    
    return sequence if sequence else [vocab_size // 2]


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "rnn_text": "I love this",
        "rnn_prediction": None,
        "rnn_h_states": None,
        "rnn_sequence": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_hidden_states(h_states):
    """Plot hidden state values over time steps."""
    if not h_states:
        return None
    
    h_array = np.array([h.flatten() for h in h_states])
    timesteps = list(range(len(h_states)))
    
    fig = go.Figure()
    
    for neuron_idx in range(min(5, h_array.shape[1])):
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=h_array[:, neuron_idx],
            mode="lines+markers",
            name=f"Hidden Unit {neuron_idx+1}",
            line=dict(width=2),
        ))
    
    fig.update_layout(
        title="Hidden States Over Time Steps",
        xaxis=dict(title="Time Step", gridcolor="#E5E7EB"),
        yaxis=dict(title="Activation Value", gridcolor="#E5E7EB"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#1F2937"),
        height=400,
    )
    return fig


def plot_sentiment_probability(score):
    """Plot sentiment prediction as gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Sentiment Score (%)"},
        delta={"reference": 50, "suffix": " from neutral"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#5227FF"},
            "steps": [
                {"range": [0, 25], "color": "#DC2626"},
                {"range": [25, 50], "color": "#F97316"},
                {"range": [50, 75], "color": "#5227FF"},
                {"range": [75, 100], "color": "#7C3AED"},
            ],
            "threshold": {
                "line": {"color": "#5227FF", "width": 4},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    
    fig.update_layout(
        height=400,
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

def rnn_page():
    st.title("RNN - Sentiment Analysis")
    st.caption(
        "Recurrent Neural Network for sentiment analysis. "
        "Process sequences of words and predict sentiment (positive/negative)."
    )
    
    _init_state()
    
    # ══════════════════════════════════════════════════════════════════════════
    # NETWORK ARCHITECTURE
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("RNN Architecture")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        vocab_size = st.slider("Vocabulary size", 10, 50, 20, key="rnn_vocab_size")
    with c2:
        embedding_dim = st.slider("Embedding dimension", 2, 16, 4, key="rnn_embed_dim")
    with c3:
        hidden_dim = st.slider("Hidden dimension", 2, 16, 4, key="rnn_hidden_dim")
    
    st.markdown(f"Input Sequence -> Embedding({embedding_dim}) -> RNN({hidden_dim}) -> Output")
    st.caption(
        f"Network processes variable-length sequences. "
        f"Uses hidden state from time t-1 along with current input."
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # INPUT TEXT
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Input Text")
    
    input_source = st.radio("Text source", ["Type text", "Sample examples"], horizontal=True, key="rnn_source")
    
    if input_source == "Type text":
        text_input = st.text_input(
            "Enter text for sentiment analysis",
            placeholder="e.g., This movie is amazing",
            value="I love this",
            key="rnn_text_input"
        )
    else:
        col_pos, col_neg = st.columns(2)
        
        with col_pos:
            st.caption("Positive examples:")
            if st.button("I love this movie", key="rnn_sample_pos_1"):
                text_input = "I love this movie"
            if st.button("Amazing product quality", key="rnn_sample_pos_2"):
                text_input = "Amazing product quality"
            if st.button("Great experience", key="rnn_sample_pos_3"):
                text_input = "Great experience"
        
        with col_neg:
            st.caption("Negative examples:")
            if st.button("This is terrible", key="rnn_sample_neg_1"):
                text_input = "This is terrible"
            if st.button("Very disappointed", key="rnn_sample_neg_2"):
                text_input = "Very disappointed"
            if st.button("Worst experience ever", key="rnn_sample_neg_3"):
                text_input = "Worst experience ever"
        
        text_input = st.session_state.get("rnn_text_input", "I love this")
    
    st.session_state.rnn_text = text_input
    
    # ══════════════════════════════════════════════════════════════════════════
    # RUN INFERENCE
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    
    run_col, reset_col = st.columns([4, 1])
    
    with reset_col:
        if st.button("Reset", use_container_width=True, key="rnn_reset"):
            st.session_state.rnn_text = ""
            st.session_state.rnn_prediction = None
            st.rerun()
    
    with run_col:
        if st.button("Predict Sentiment", type="primary", use_container_width=True, key="rnn_predict"):
            rnn_model = SimpleRNN(vocab_size, embedding_dim, hidden_dim)
            
            sequence = text_to_sequence(text_input, VOCAB, vocab_size)
            h_states, h_final, prediction = rnn_model.forward(sequence)
            
            st.session_state.rnn_prediction = prediction
            st.session_state.rnn_h_states = h_states
            st.session_state.rnn_sequence = sequence
    
    # ══════════════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.rnn_prediction is None:
        st.info("Enter text and click 'Predict Sentiment' to analyze.")
        return
    
    prediction_score = st.session_state.rnn_prediction
    h_states = st.session_state.rnn_h_states
    sequence = st.session_state.rnn_sequence
    
    st.divider()
    st.subheader("Results")
    
    # Sentiment label
    if prediction_score >= 0.5:
        sentiment = "POSITIVE"
        color = "#5227FF"
    else:
        sentiment = "NEGATIVE"
        color = "#DC2626"
    
    col_metric, col_label = st.columns([1, 2])
    with col_metric:
        st.metric("Sentiment Score", f"{prediction_score:.4f}")
    with col_label:
        st.markdown(
            f"<h3 style='color:{color}; margin-top: 25px;'>{sentiment}</h3>",
            unsafe_allow_html=True
        )
    
    # ══════════════════════════════════════════════════════════════════════════
    # SEQUENCE BREAKDOWN
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Sequence Processing")
    
    col_text, col_tokens = st.columns(2)
    
    with col_text:
        st.caption("Input Text Words:")
        words = text_input.lower().split()
        st.write(" -> ".join(words) if words else "(empty)")
    
    with col_tokens:
        st.caption("Token Indices:")
        st.write(" -> ".join([str(t) for t in sequence]))
    
    st.caption(f"Sequence length: {len(sequence)} words")
    
    # ══════════════════════════════════════════════════════════════════════════
    # HIDDEN STATES VISUALIZATION
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Hidden State Evolution")
    
    if h_states and len(h_states) > 0:
        fig_h = plot_hidden_states(h_states)
        if fig_h:
            st.plotly_chart(fig_h, use_container_width=True)
        
        st.caption(
            "Each line represents one hidden unit's activation value across time steps. "
            "The RNN uses these states to maintain context as it processes the sequence."
        )
    else:
        st.info("No hidden states to visualize for single-word inputs.")
    
    # ══════════════════════════════════════════════════════════════════════════
    # SENTIMENT METER
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Sentiment Score")
    
    fig_gauge = plot_sentiment_probability(prediction_score)
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.caption(
        "Score near 1.0 indicates strong positive sentiment. "
        "Score near 0.0 indicates strong negative sentiment. "
        "Score near 0.5 indicates neutral."
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # WORD SENTIMENT ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Word Sentiment Contribution")
    
    words = text_input.lower().split()
    word_scores = []
    
    for word in words:
        if word in VOCAB:
            score = VOCAB[word]
            word_scores.append((word, score))
    
    if word_scores:
        df_words = pd.DataFrame(word_scores, columns=["Word", "Sentiment Score"])
        df_words["Impact"] = df_words["Sentiment Score"].apply(
            lambda x: "Positive" if x > 0 else "Negative"
        )
        
        st.dataframe(df_words.round(3), hide_index=True, use_container_width=True)
    else:
        st.info("No recognized sentiment words in the input text.")
    
    # ══════════════════════════════════════════════════════════════════════════
    # RNN EXPLANATION
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("How the RNN Works")
    
    with st.expander("RNN Processing Steps", expanded=False):
        st.write("""
        1. **Tokenization**: Convert words to numerical indices based on vocabulary.
        2. **Embedding**: Map each token to a dense embedding vector.
        3. **RNN Processing**: For each time step:
           - Combine embedded input with previous hidden state
           - Compute new hidden state using tanh activation
           - Maintain information from previous time steps
        4. **Output**: Use final hidden state to predict sentiment (0=negative, 1=positive)
        5. **Sequence Handling**: Unlike feedforward networks, RNNs process sequences step-by-step,
           allowing them to capture context and dependencies across the sequence.
        """)
    
    with st.expander("Vanishing Gradient Problem", expanded=False):
        st.write("""
        Traditional RNNs can suffer from vanishing gradients during backpropagation through time (BPTT).
        As gradients are multiplied through many time steps, they can become extremely small,
        making it hard to learn long-term dependencies.
        
        Solutions:
        - LSTM (Long Short-Term Memory) with cell states and gates
        - GRU (Gated Recurrent Unit) with update and reset gates
        - Gradient clipping
        - Better initialization strategies
        """)
