import streamlit as st

from brugada.app_state import clear_uploads, ensure_app_state
from brugada.app_tabs import (
    render_chatbot_tab,
    render_clinical_report_tab,
    render_records_tab,
)
from brugada.ui.components import inject_custom_css
from brugada.ui.sidebar_panel import render_patient_input_panel


st.set_page_config(page_title="Brugada AI Assistant", page_icon="ECG", layout="wide")
inject_custom_css()
st.title("Brugada Syndrome Clinical AI Assistant")
st.caption("Multi-view deep feature stacking + meta-learner for single-patient ECG triage")

ensure_app_state()

left, right = st.columns([1, 2])

with left:
    input_ctx = render_patient_input_panel(clear_uploads)

pairs = input_ctx["pairs"]
uploaded_files = input_ctx["uploaded_files"]
run_btn = input_ctx["run_btn"]
patient_id = input_ctx["patient_id"]
batch_patient_id_map = input_ctx["batch_patient_id_map"]
is_batch = input_ctx["is_batch"]

with right:
    tab_report, tab_chatbot, tab_records = st.tabs(["Clinical Report", "AI Advisor", "Records Center"])

    with tab_report:
        single_result_to_show, current_view = render_clinical_report_tab(
            pairs=pairs,
            uploaded_files=uploaded_files,
            run_btn=run_btn,
            is_batch=is_batch,
            patient_id=patient_id,
            batch_patient_id_map=batch_patient_id_map,
        )

    with tab_chatbot:
        render_chatbot_tab(
            single_result_to_show=single_result_to_show,
            is_batch=is_batch,
            current_view=current_view,
        )

    with tab_records:
        render_records_tab()
