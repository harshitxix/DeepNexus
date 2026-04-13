"""
DESIGN TOKENS FOR DEEP LEARNING DASHBOARD
Centralized design constants and color definitions for consistent styling across all modules.
Import these tokens in your modules to ensure consistency.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# COLOR PALETTE
# ═══════════════════════════════════════════════════════════════════════════════

# Primary Colors
COLOR_PRIMARY = "#5227FF"                    # Deep Purple - Main brand color
COLOR_PRIMARY_LIGHT = "#7C3AED"              # Light Purple - Hover states
COLOR_PRIMARY_LIGHTER = "#A78BFA"            # Very Light Purple - Disabled states

# Background Colors
COLOR_BG_MAIN = "#FFFFFF"                    # Pure white - Main background
COLOR_BG_PANEL = "#F8FAFC"                   # Soft white-grey - Secondary panels
COLOR_BG_SUBTLE = "#F1F5F9"                  # Light grey-blue - Tertiary panels
COLOR_SIDEBAR_BG = "#F0F4F9"                 # Sidebar background

# Text Colors
COLOR_TEXT_PRIMARY = "#0F172A"               # Dark slate - Primary text
COLOR_TEXT_SECONDARY = "#334155"             # Medium slate - Secondary text
COLOR_TEXT_MUTED = "#64748B"                 # Light slate - Disabled/muted text

# Border & Divider Colors
COLOR_BORDER_DARK = "#CBD5E1"                # Strong borders
COLOR_BORDER_LIGHT = "#E2E8F0"               # Subtle borders
COLOR_DIVIDER = "#E5E7EB"                    # Light dividers

# Semantic Colors
COLOR_SUCCESS = "#10B981"                    # Success/positive
COLOR_WARNING = "#F59E0B"                    # Warning
COLOR_ERROR = "#DC2626"                      # Error/danger
COLOR_INFO = "#3B82F6"                       # Information

# Accent & Highlights
COLOR_ACCENT = "#5227FF"                     # Same as primary
COLOR_HIGHLIGHT = "rgba(82, 39, 255, 0.1)"  # Soft highlight background
COLOR_HIGHLIGHT_BORDER = "rgba(82, 39, 255, 0.2)"  # Highlight border

# ═══════════════════════════════════════════════════════════════════════════════
# SPACING SCALE
# ═══════════════════════════════════════════════════════════════════════════════

SPACING_XS = "0.25rem"
SPACING_SM = "0.5rem"
SPACING_MD = "1rem"
SPACING_LG = "1.5rem"
SPACING_XL = "2rem"
SPACING_2XL = "3rem"

# ═══════════════════════════════════════════════════════════════════════════════
# TYPOGRAPHY
# ═══════════════════════════════════════════════════════════════════════════════

FONT_FAMILY = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

FONT_SIZE_SM = "0.875rem"
FONT_SIZE_BASE = "1rem"
FONT_SIZE_LG = "1.125rem"
FONT_SIZE_XL = "1.5rem"
FONT_SIZE_2XL = "2rem"

FONT_WEIGHT_NORMAL = 400
FONT_WEIGHT_MEDIUM = 500
FONT_WEIGHT_SEMIBOLD = 600
FONT_WEIGHT_BOLD = 700

# ═══════════════════════════════════════════════════════════════════════════════
# BORDER RADIUS
# ═══════════════════════════════════════════════════════════════════════════════

RADIUS_SM = "4px"
RADIUS_MD = "8px"
RADIUS_LG = "12px"
RADIUS_XL = "16px"

# ═══════════════════════════════════════════════════════════════════════════════
# SHADOWS
# ═══════════════════════════════════════════════════════════════════════════════

SHADOW_SM = "0 1px 2px 0 rgba(0, 0, 0, 0.05)"
SHADOW_MD = "0 4px 6px -1px rgba(0, 0, 0, 0.1)"
SHADOW_LG = "0 10px 15px -3px rgba(0, 0, 0, 0.1)"
SHADOW_ACCENT = "0 4px 12px rgba(82, 39, 255, 0.15)"
SHADOW_ACCENT_HOVER = "0 8px 20px rgba(82, 39, 255, 0.25)"

# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT STYLES - PLOTLY CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

# Common Plotly layout settings for consistent chart styling
PLOTLY_LAYOUT_BASE = {
    "plot_bgcolor": COLOR_BG_MAIN,
    "paper_bgcolor": COLOR_BG_MAIN,
    "font": {"color": COLOR_TEXT_PRIMARY, "family": FONT_FAMILY},
    "xaxis": {"gridcolor": COLOR_DIVIDER},
    "yaxis": {"gridcolor": COLOR_DIVIDER},
    "margin": {"l": 20, "r": 20, "t": 50, "b": 20},
}

# Color palette for chart elements
CHART_COLORS = [
    COLOR_PRIMARY,           # Purple
    COLOR_PRIMARY_LIGHT,     # Light purple
    COLOR_TEXT_SECONDARY,    # Slate
    COLOR_ERROR,             # Red
    COLOR_WARNING,           # Orange
    COLOR_INFO,              # Blue
]

# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT STYLES - BUTTONS
# ═══════════════════════════════════════════════════════════════════════════════

BUTTON_STYLE = {
    "background_color": COLOR_PRIMARY,
    "text_color": "white",
    "border_color": "none",
    "border_radius": RADIUS_MD,
    "padding": f"{SPACING_SM} {SPACING_LG}",
    "font_weight": "600",
}

BUTTON_STYLE_SECONDARY = {
    "background_color": COLOR_BG_SUBTLE,
    "text_color": COLOR_TEXT_PRIMARY,
    "border_color": COLOR_BORDER_LIGHT,
    "border_radius": RADIUS_MD,
    "padding": f"{SPACING_SM} {SPACING_LG}",
}

# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT STYLES - CARDS & CONTAINERS
# ═══════════════════════════════════════════════════════════════════════════════

CARD_STYLE = {
    "background_color": COLOR_BG_PANEL,
    "border_color": COLOR_BORDER_LIGHT,
    "border_radius": RADIUS_MD,
    "padding": SPACING_MD,
    "box_shadow": SHADOW_SM,
}

CONTAINER_STYLE = {
    "background_color": COLOR_BG_MAIN,
    "border_color": COLOR_BORDER_LIGHT,
    "border_radius": RADIUS_MD,
    "padding": SPACING_MD,
}

# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT STYLES - INPUTS & FORMS
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_STYLE = {
    "background_color": "white",
    "text_color": COLOR_TEXT_PRIMARY,
    "border_color": COLOR_BORDER_LIGHT,
    "border_radius": RADIUS_MD,
    "padding": f"{SPACING_SM} {SPACING_MD}",
    "focus_border_color": COLOR_PRIMARY,
}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_chart_color(index: int) -> str:
    """Get a chart color from the palette by index."""
    return CHART_COLORS[index % len(CHART_COLORS)]

def get_semantic_color(status: str) -> str:
    """Get a semantic color by status type."""
    colors = {
        "success": COLOR_SUCCESS,
        "warning": COLOR_WARNING,
        "error": COLOR_ERROR,
        "info": COLOR_INFO,
    }
    return colors.get(status.lower(), COLOR_PRIMARY)

def create_button_style(style_type: str = "primary") -> dict:
    """Create button style dictionary by type."""
    styles = {
        "primary": BUTTON_STYLE,
        "secondary": BUTTON_STYLE_SECONDARY,
    }
    return styles.get(style_type, BUTTON_STYLE)

def create_card_style(custom_bg: str = None, custom_border: str = None) -> dict:
    """Create card style with optional customization."""
    style = CARD_STYLE.copy()
    if custom_bg:
        style["background_color"] = custom_bg
    if custom_border:
        style["border_color"] = custom_border
    return style

# ═══════════════════════════════════════════════════════════════════════════════
# SPACING SHORTHAND
# ═══════════════════════════════════════════════════════════════════════════════

SPACING = {
    "xs": SPACING_XS,
    "sm": SPACING_SM,
    "md": SPACING_MD,
    "lg": SPACING_LG,
    "xl": SPACING_XL,
    "2xl": SPACING_2XL,
}

# ═══════════════════════════════════════════════════════════════════════════════
# COLOR GROUPS FOR EASY ACCESS
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "primary": COLOR_PRIMARY,
    "primary_light": COLOR_PRIMARY_LIGHT,
    "primary_lighter": COLOR_PRIMARY_LIGHTER,
    "bg_main": COLOR_BG_MAIN,
    "bg_panel": COLOR_BG_PANEL,
    "bg_subtle": COLOR_BG_SUBTLE,
    "text_primary": COLOR_TEXT_PRIMARY,
    "text_secondary": COLOR_TEXT_SECONDARY,
    "text_muted": COLOR_TEXT_MUTED,
    "border_dark": COLOR_BORDER_DARK,
    "border_light": COLOR_BORDER_LIGHT,
    "success": COLOR_SUCCESS,
    "warning": COLOR_WARNING,
    "error": COLOR_ERROR,
    "info": COLOR_INFO,
}

# ═══════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════

"""
# In modules, import and use like this:

from design_tokens import (
    COLOR_PRIMARY, COLOR_BG_MAIN, COLOR_TEXT_PRIMARY,
    PLOTLY_LAYOUT_BASE, CHART_COLORS, get_chart_color
)
import plotly.graph_objects as go

# Example 1: Using colors directly
st.markdown(f"<h1 style='color:{COLOR_PRIMARY}'>Title</h1>", unsafe_allow_html=True)

# Example 2: Using predefined layout for all charts
fig = go.Figure()
fig.update_layout(**PLOTLY_LAYOUT_BASE)

# Example 3: Using helper function for colors
color = get_chart_color(0)  # Returns first color from palette

# Example 4: Creating styles dynamically
button_style = create_button_style("primary")

"""
