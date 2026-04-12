import streamlit as st

from brugada.services.chatbot import BrugadaChatbot
from brugada.storage.record_store import init_record_store


def ensure_app_state() -> None:
    """Initialize all Streamlit session keys used by the app."""
    if "record_store_ready" not in st.session_state:
        try:
            # Use temporary storage mode - clear data on app start
            init_record_store(clear_existing=True)
            st.session_state.record_store_ready = True
            st.session_state.record_store_error = ""
        except Exception as exc:  # noqa: BLE001
            st.session_state.record_store_ready = False
            st.session_state.record_store_error = str(exc)

    if "chatbot" not in st.session_state:
        try:
            st.session_state.chatbot = BrugadaChatbot()
            st.session_state.chatbot_ready = True
        except Exception as exc:  # noqa: BLE001
            st.session_state.chatbot_ready = False
            st.session_state.chatbot_error = str(exc)

    if "last_ml_result" not in st.session_state:
        st.session_state.last_ml_result = None

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    if "deleted_pairs" not in st.session_state:
        st.session_state.deleted_pairs = set()

    if "records_loaded_result" not in st.session_state:
        st.session_state.records_loaded_result = None

    if "persistence_notice" not in st.session_state:
        st.session_state.persistence_notice = ""

    if "batch_record_uid_map" not in st.session_state:
        st.session_state.batch_record_uid_map = {}


def clear_uploads() -> None:
    st.session_state.uploader_key += 1
    st.session_state.deleted_pairs = set()
    if "last_ml_result" in st.session_state:
        st.session_state.last_ml_result = None
    if "batch_results" in st.session_state:
        del st.session_state.batch_results
