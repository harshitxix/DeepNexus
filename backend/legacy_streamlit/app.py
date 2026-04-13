import streamlit as st
from modules.common import apply_theme, MODULES, MODULE_DESCRIPTIONS
from modules.perceptron import perceptron_page
from modules.forward_prop import forward_propagation_page
from modules.backward_prop import backward_propagation_page
from modules.multilayer_perceptron import mlp_page
from modules.opencv_lab import opencv_page
from modules.rnn import rnn_page
from modules.study_mode import study_mode_page

# Page config
st.set_page_config(
    page_title="Deep Learning Dashboard",
    page_icon="DL",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

# Initialize session state
if "current_module" not in st.session_state:
    st.session_state.current_module = None

# Allow deep-linking to a module from query parameters, used by React iframe embedding.
query_module = st.query_params.get("module", None)
if query_module in MODULES and query_module != st.session_state.current_module:
    st.session_state.current_module = query_module

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.markdown("### Deep Learning Modules")
st.sidebar.markdown("---")

# Module buttons
for module_key, module_name in MODULES.items():
    is_active = st.session_state.current_module == module_key
    button_class = ">" if is_active else " "
    
    if st.sidebar.button(
        f"{button_class} {module_name}",
        key=f"btn_{module_key}",
        use_container_width=True,
    ):
        if module_key == "study_mode":
            st.session_state.study_selected_topic = None
            st.session_state.study_picker_open = False
            st.session_state.study_topic_select_intro = "Select module"
        st.session_state.current_module = module_key

st.sidebar.markdown("---")
st.sidebar.caption(f"Active: {MODULES.get(st.session_state.current_module, 'None')}")

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================
current = st.session_state.current_module

if current is None:
    # Dashboard home
    st.markdown("# Deep Learning Educational Dashboard")
    st.markdown("""
    Select a module from the sidebar to explore neural networks, optimization, and computer vision.
    
    #### Available Modules:
    """)
    
    cols = st.columns(2)
    for idx, (key, name) in enumerate(MODULES.items()):
        with cols[idx % 2]:
            with st.container():
                st.markdown(f"**{name}**")
                st.caption(MODULE_DESCRIPTIONS.get(key, "Learn more about this topic."))
                if st.button("Open ->", key=f"home_{key}", use_container_width=True):
                    st.session_state.current_module = key
                    st.rerun()

elif current == "perceptron":
    perceptron_page()

elif current == "forward_prop":
    forward_propagation_page()

elif current == "backward_prop":
    backward_propagation_page()

elif current == "multilayer":
    mlp_page()

elif current == "opencv":
    opencv_page()

elif current == "rnn":
    rnn_page()

elif current == "study_mode":
    study_mode_page()

else:
    st.error(f"Module '{current}' not found.")
