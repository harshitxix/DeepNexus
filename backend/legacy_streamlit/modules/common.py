"""
DESIGN SYSTEM & COMMON UTILITIES
================================

DESIGN SYSTEM FILES:
  - design_system.html  : Centralized CSS for all styling (colors, typography, components, etc.)
  - design_tokens.py    : Python constants for design values (use when coding features)

HOW TO USE:
  1. For HTML/CSS styling: All designs are in design_system.html and auto-loaded via apply_theme()
  2. For Python code: Import from design_tokens.py
     
     from design_tokens import (
         COLOR_PRIMARY, COLOR_BG_MAIN, COLOR_TEXT_PRIMARY,
         PLOTLY_LAYOUT_BASE, get_chart_color, SPACING, RADIUS_MD
     )

ADDING NEW COLORS/STYLES:
  1. Add to design_tokens.py (Python constant)
  2. Add to design_system.html CSS (CSS variable if CSS-only, or class if component)
  3. Always keep both files in sync!

CONSISTENCY CHECKLIST:
  ✓ All buttons use COLOR_PRIMARY (#5227FF)
  ✓ All backgrounds use COLOR_BG_MAIN (#FFFFFF)
  ✓ All text uses COLOR_TEXT_PRIMARY (#0F172A)
  ✓ All cards use predefined CARD_STYLE
  ✓ All charts use PLOTLY_LAYOUT_BASE
  ✓ All spacing/radius values come from SPACING/RADIUS constants
"""

import math
from typing import Dict, List

import numpy as np
import streamlit as st


MODULES: Dict[str, str] = {
    "perceptron": "Perceptron",
    "forward_prop": "Forward Propagation",
    "backward_prop": "Backward Propagation",
    "multilayer": "Multilayer Perceptron",
    "opencv": "OpenCV",
    "rnn": "RNN & LSTM",
    "study_mode": "Study Mode",
}

MODULE_DESCRIPTIONS = {
    "perceptron": "Single neuron with weights, bias, activation functions, and decision boundary visualization.",
    "forward_prop": "Trace forward signal propagation through neural network layers with value tracking.",
    "backward_prop": "Step-through backpropagation with computation graph and weight update animations.",
    "multilayer": "Build and experiment with multi-layer networks, layer inspector, and parameter analysis.",
    "opencv": "Image processing with convolution, pooling, Canny edge detection, and contour detection.",
    "rnn": "RNN unrolling, LSTM/GRU gates, hidden state animation, and vanishing gradient visualizer.",
    "study_mode": "In-depth theory, formulas, and concept charts for all modules in one guided study page.",
}


def apply_theme() -> None:
    """
    Apply the centralized design system to the Streamlit app.
    This function reads the design_system.html file and injects all CSS rules.
    All design tokens are defined in design_system.html and design_tokens.py.
    """
    import os
    
    # Path to the design system file
    design_system_path = os.path.join(os.path.dirname(__file__), "..", "design_system.html")
    
    try:
        # Read the design system HTML file
        with open(design_system_path, "r", encoding="utf-8") as f:
            design_css = f.read()
        
        # Inject the design system CSS into the Streamlit app
        st.markdown(design_css, unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback: Use inline CSS if file not found
        st.warning("Design system file not found. Using fallback styling.")
        st.markdown(
            """
<style>
:root {
    --color-primary: #5227FF;
    --color-bg-main: #FFFFFF;
    --color-text-primary: #0F172A;
    --color-text-secondary: #334155;
    --color-border-light: #E2E8F0;
}
.main { background: var(--color-bg-main) !important; }
h1, h2, h3, h4, h5, h6 { color: var(--color-text-primary) !important; }
h1 { color: var(--color-primary) !important; }
body, p, span, div { color: var(--color-text-primary) !important; }
button { background-color: var(--color-primary) !important; color: white !important; border: none !important; }
</style>
""",
            unsafe_allow_html=True
        )


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def softmax(x):
    shifted = x - np.max(x)
    expv = np.exp(shifted)
    return expv / np.sum(expv)


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)


def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def parse_float_list(text: str, fallback: List[float]) -> List[float]:
    try:
        vals = [float(v.strip()) for v in text.split(",") if v.strip() != ""]
        return vals if vals else fallback
    except ValueError:
        return fallback


def parse_int_list(text: str, fallback: List[int]) -> List[int]:
    try:
        vals = [int(v.strip()) for v in text.split(",") if v.strip() != ""]
        vals = [v for v in vals if v > 0]
        return vals if vals else fallback
    except ValueError:
        return fallback


def attention_scale(d: int) -> float:
    return math.sqrt(max(d, 1))
