import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# UI Assets (SVGs & CSS)
# =============================================================================
SVG_SUCCESS = '''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#16A34A" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>'''
SVG_WARNING = '''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#D97706" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'''
SVG_ERROR = '''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#DC2626" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>'''
SVG_INFO = '''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0284C7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>'''
SVG_FOLDER = '''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>'''

def get_status_indicator_svg(is_detected: bool, is_urgent: bool = False, is_gray_zone: bool = False) -> str:
    # Detected & Urgent: Triangle warning
    if is_detected or is_urgent:
        color = "#EF4444"
        return f'''<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block; margin-left: 8px;"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'''
    # Gray zone (requires review)
    elif is_gray_zone:
        color = "#F59E0B"
        return f'''<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block; margin-left: 8px;"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>'''
    # Clear / Normal
    else:
        color = "#22C55E"
        return f'''<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block; margin-left: 8px;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>'''

def inject_custom_css():
    st.markdown("""
        <style>
            /* Larger dashed dropzone for file uploader */
            [data-testid="stFileUploaderDropzone"] {
                min-height: 200px;
                display: flex;
                align-items: center;
                justify-content: center;
                border: 2px dashed #94A3B8 !important;
                border-radius: 12px;
                background-color: transparent !important;
                transition: background-color 0.2s ease, border-color 0.2s ease;
            }
            [data-testid="stFileUploaderDropzone"]:hover {
                border-color: #3B82F6 !important;
                background-color: rgba(59, 130, 246, 0.08) !important;
            }
            [data-testid="stFileUploaderDropzone"] svg {
                width: 50px;
                height: 50px;
                color: inherit;
                opacity: 0.8;
                margin-bottom: 10px;
            }
            
            [data-testid="stFileUploaderDropzone"] > div > div > small {
                display: none !important;
            }
            
            /* Hide Streamlit's native awkward individual file uploader list */
            [data-testid="stUploadedFile"] {
                display: none !important;
            }
            
            /* Custom card styling for paired records */
            .record-card {
                padding: 0.4rem 0.8rem;
                margin-bottom: 0px;
                height: 42px;
                border: 1px solid rgba(150, 150, 150, 0.2);
                border-radius: 8px;
                background-color: rgba(150, 150, 150, 0.05);
                color: inherit;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .record-card-warning {
                padding: 0.4rem 0.8rem;
                margin-bottom: 0px;
                height: 42px;
                border: 1px solid rgba(245, 158, 11, 0.4);
                border-radius: 8px;
                background-color: rgba(245, 158, 11, 0.1);
                color: inherit;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .record-tag {
                background-color: rgba(59, 130, 246, 0.15);
                color: #3B82F6;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.85em;
                font-weight: 500;
                margin-left: 6px;
            }
            
            .record-tag-missing {
                background-color: rgba(239, 68, 68, 0.15);
                color: #EF4444;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.85em;
                font-weight: 500;
                margin-left: 6px;
            }
        </style>
    """, unsafe_allow_html=True)

DEFAULT_LEAD_NAMES = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

@st.cache_data(show_spinner=False)
def _seconds_axis(length: int, fs: float) -> np.ndarray:
    return np.arange(length) / fs

def _plot_12_lead(signal: np.ndarray, lead_names: list[str], fs: float, highlights: dict[str, list[tuple[int, int]]]):
    n_samples, n_leads = signal.shape
    lead_names = (lead_names + DEFAULT_LEAD_NAMES)[:n_leads]

    fig, axes = plt.subplots(6, 2, figsize=(14, 12), sharex=True)
    axes = axes.ravel()
    t = _seconds_axis(n_samples, fs)

    for i in range(min(12, n_leads)):
        ax = axes[i]
        lead = lead_names[i]
        ax.plot(t, signal[:, i], linewidth=0.8, color="#0F172A")
        ax.set_title(lead, fontsize=10, loc="left")
        ax.grid(alpha=0.15)

        if lead in {"V1", "V2", "V3"} and lead in highlights:
            for start, end in highlights[lead]:
                ax.axvspan(start / fs, end / fs, color="#EF4444", alpha=0.22)

    for j in range(min(12, n_leads), 12):
        axes[j].axis("off")

    fig.suptitle("12-lead ECG overview (V1-V3 highlighted attention zones)", fontsize=13, y=0.995)
    fig.supxlabel("Time (s)")
    fig.supylabel("Normalized amplitude")
    fig.tight_layout()
    return fig


def _plot_decision_margin(probability: float, threshold: float):
    fig, ax = plt.subplots(figsize=(8.0, 1.8))
    gray_upper = min(1.0, threshold + 0.01)
    ax.axvspan(0.0, threshold, color="#DCFCE7", alpha=0.9, label="Below threshold")
    ax.axvspan(threshold, 1.0, color="#FEE2E2", alpha=0.60, label="Above threshold")
    ax.axvspan(threshold, gray_upper, color="#FEF3C7", alpha=0.95, label="Borderline-positive zone")
    ax.axvline(threshold, color="#16A34A", linewidth=2.0, linestyle="--", label=f"Threshold {threshold:.3f}")
    ax.scatter([probability], [0.5], color="#DC2626", s=90, zorder=5, label=f"Prediction {probability:.3f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("Brugada probability")
    ax.set_title("Decision Margin View")
    ax.grid(alpha=0.2, axis="x")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def _plot_evidence_heatmap(evidence_df: pd.DataFrame):
    rows = []
    for lead in ["V1", "V2", "V3"]:
        sub = evidence_df[evidence_df["lead"] == lead]
        if sub.empty:
            rows.append([np.nan, np.nan, np.nan, np.nan])
            continue
        item = sub.iloc[0]
        rows.append(
            [
                float(item.get("j_height", 0.0)),
                float(item.get("st_slope", 0.0)),
                float(item.get("curvature", 0.0)),
                float(item.get("score", 0.0)),
            ]
        )

    matrix = np.array(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 2.8))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["J-height", "ST-slope", "Curvature", "Score"])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["V1", "V2", "V3"])
    ax.set_title("V1-V3 Morphology Evidence Heatmap")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isnan(matrix[i, j]):
                ax.text(j, i, "N/A", ha="center", va="center", color="#4B5563", fontsize=8)
            else:
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="#111827", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Feature intensity")
    fig.tight_layout()
    return fig

def _recommendation_banner(recommendation_tier: str, recommendation_text: str):
    if recommendation_tier in {"urgent_cardiology_review", "urgent_review_repeat_ecg_quality_check"}:
        st.markdown(f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fef3c7; color: #92400e; display: flex; align-items: center;'>{SVG_WARNING} {recommendation_text}</div>", unsafe_allow_html=True)
    elif recommendation_tier == "gray_zone_priority_review":
        st.markdown(f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #e0f2fe; color: #075985; display: flex; align-items: center;'>{SVG_INFO} {recommendation_text}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #dcfce7; color: #166534; display: flex; align-items: center;'>{SVG_SUCCESS} {recommendation_text}</div>", unsafe_allow_html=True)
