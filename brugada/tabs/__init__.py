from .chatbot_tab import render_chatbot_tab
from .clinical_report_tab import render_clinical_report_tab
from .ecg_review_tab import render_ecg_review_tab
from .records_tab import render_records_tab

__all__ = [
    "render_clinical_report_tab",
    "render_ecg_review_tab",
    "render_chatbot_tab",
    "render_records_tab",
]