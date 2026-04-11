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
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

            /* Global Typography & Spacing */
            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
                line-height: 1.6;
            }

            /* Hide Streamlit Artifacts (Hamburger Menu, Footers, Deploy button) */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            [data-testid="stHeader"] {display: none;}
            
            /* Custom Card Styling */
            .custom-card {
                background-color: #ffffff;
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
                border: 1px solid #e2e8f0;
                margin-bottom: 1rem;
            }

            /* Streamlit Metrics Upgrade */
            [data-testid="stMetricValue"] {
                font-size: 2.2rem !important;
                font-weight: 700 !important;
                color: #0f172a !important;
                text-align: center;
            }
            [data-testid="stMetricLabel"] {
                font-size: 0.95rem !important;
                font-weight: 600 !important;
                color: #475569 !important;
                text-align: center;
                justify-content: center;
            }
            [data-testid="stMetricDelta"] {
                justify-content: center;
            }

            /* Primary Button Modernization */
            [data-testid="baseButton-primary"] {
                border-radius: 8px !important;
                font-weight: 600 !important;
                letter-spacing: 0.3px;
                padding: 0.5rem 1rem !important;
                transition: all 0.2s ease-in-out !important;
                box-shadow: 0 2px 4px rgba(37, 99, 235, 0.15) !important;
            }
            [data-testid="baseButton-primary"]:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
                filter: brightness(105%);
            }

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

            .metric-tile {
                background: rgba(255, 255, 255, 0.65);
                border: 1px solid rgba(148, 163, 184, 0.32);
                border-radius: 10px;
                padding: 0.65rem 0.75rem;
                min-height: 96px;
            }

            .metric-tile-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.5rem;
                margin-bottom: 0.35rem;
            }

            .metric-tile-label {
                color: #334155;
                font-size: 0.92rem;
                font-weight: 600;
                line-height: 1.2;
            }

            .metric-info-icon {
                color: #94a3b8;
                font-size: 0.98rem;
                font-weight: 700;
                cursor: pointer;
                line-height: 1;
                user-select: none;
            }

            .metric-info-details {
                position: relative;
                display: inline-block;
            }

            .metric-info-details > summary {
                list-style: none;
                outline: none;
            }

            .metric-info-details > summary::-webkit-details-marker {
                display: none;
            }

            .metric-info-details > summary:hover .metric-info-icon {
                color: #64748b;
            }

            .metric-info-panel {
                position: absolute;
                top: 1.25rem;
                right: 0;
                z-index: 30;
                width: 280px;
                max-width: min(280px, 42vw);
                background: #f8fafc;
                border: 1px solid rgba(148, 163, 184, 0.45);
                border-radius: 10px;
                box-shadow: 0 12px 24px rgba(15, 23, 42, 0.16);
                padding: 0.6rem 0.7rem;
                color: #334155;
                font-size: 0.78rem;
                font-weight: 500;
                line-height: 1.35;
                text-align: left;
            }

            .metric-info-panel-title {
                color: #0f172a;
                font-size: 0.76rem;
                font-weight: 700;
                letter-spacing: 0.2px;
                margin-bottom: 0.35rem;
            }

            .metric-tile-value {
                color: #0f172a;
                font-size: 2.05rem;
                font-weight: 700;
                line-height: 1.05;
                letter-spacing: 0.2px;
            }

            .decision-legend-panel {
                display: flex;
                flex-wrap: wrap;
                gap: 0.6rem 1rem;
                padding: 0.55rem 0.7rem;
                margin-bottom: 0.55rem;
                border: 1px solid rgba(148, 163, 184, 0.35);
                border-radius: 9px;
                background: rgba(248, 250, 252, 0.85);
            }

            .decision-legend-item {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                color: #334155;
                font-size: 0.78rem;
                font-weight: 600;
            }

            .decision-legend-swatch {
                width: 15px;
                height: 11px;
                border-radius: 3px;
                border: 1px solid rgba(15, 23, 42, 0.12);
            }

            .decision-legend-swatch-below {
                background: #dcfce7;
            }

            .decision-legend-swatch-above {
                background: #fee2e2;
            }

            .decision-legend-swatch-gray {
                background: #fef3c7;
            }

            .decision-legend-line {
                width: 18px;
                border-top: 2px dashed #16a34a;
            }

            .decision-legend-dot {
                width: 10px;
                height: 10px;
                border-radius: 999px;
                background: #dc2626;
                display: inline-block;
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


def _trim_trailing_quiet_tail(signal: np.ndarray, fs: float) -> np.ndarray:
    """Trim long low-energy tails (typically zero padding) for cleaner ECG display."""
    if signal.ndim != 2 or signal.shape[0] < 10:
        return signal

    envelope = np.max(np.abs(signal), axis=1)
    peak = float(np.max(envelope))
    if peak <= 0.0:
        return signal

    active_threshold = max(peak * 0.01, 1e-6)
    active_indices = np.where(envelope > active_threshold)[0]
    if active_indices.size == 0:
        return signal

    last_active = int(active_indices[-1])
    safety_tail = max(1, int(fs * 0.25))
    min_keep = min(signal.shape[0], max(200, int(fs * 1.5)))
    end_idx = min(signal.shape[0], max(last_active + safety_tail, min_keep))
    return signal[:end_idx, :]

def _plot_12_lead(signal: np.ndarray, lead_names: list[str], fs: float, highlights: dict[str, list[tuple[int, int]]]):
    signal = _trim_trailing_quiet_tail(signal, fs)
    n_samples, n_leads = signal.shape
    lead_names = (lead_names + DEFAULT_LEAD_NAMES)[:n_leads]

    fig, axes = plt.subplots(6, 2, figsize=(14, 12), sharex=True)
    axes = axes.ravel()
    t = _seconds_axis(n_samples, fs)
    max_time = t[-1] if n_samples > 1 else (1.0 / fs)

    for i in range(min(12, n_leads)):
        ax = axes[i]
        lead = lead_names[i]
        ax.plot(t, signal[:, i], linewidth=0.8, color="#0F172A")
        ax.set_title(lead, fontsize=10, loc="left")
        ax.grid(alpha=0.15)

        if lead in {"V1", "V2", "V3"} and lead in highlights:
            for start, end in highlights[lead]:
                start_t = max(0.0, start / fs)
                end_t = min(max_time, end / fs)
                if end_t > start_t:
                    ax.axvspan(start_t, end_t, color="#EF4444", alpha=0.22)

    for j in range(min(12, n_leads), 12):
        axes[j].axis("off")

    fig.suptitle("12-lead ECG overview (V1-V3 highlighted attention zones)", fontsize=13, y=0.995)
    fig.supxlabel("Time (s)")
    fig.supylabel("Normalized amplitude")
    fig.tight_layout()
    return fig


def _render_decision_margin_legend(probability: float, threshold: float):
    st.markdown(
        f"""
        <div class="decision-legend-panel">
            <div class="decision-legend-item">
                <span class="decision-legend-swatch decision-legend-swatch-below"></span>
                <span>Below decision boundary</span>
            </div>
            <div class="decision-legend-item">
                <span class="decision-legend-swatch decision-legend-swatch-above"></span>
                <span>Above decision boundary</span>
            </div>
            <div class="decision-legend-item">
                <span class="decision-legend-swatch decision-legend-swatch-gray"></span>
                <span>Near-boundary review zone</span>
            </div>
            <div class="decision-legend-item">
                <span class="decision-legend-line"></span>
                <span>Threshold {threshold:.3f}</span>
            </div>
            <div class="decision-legend-item">
                <span class="decision-legend-dot"></span>
                <span>Score {probability:.3f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _plot_decision_margin(probability: float, threshold: float, gray_zone_upper: float | None = None):
    fig, ax = plt.subplots(figsize=(9.0, 2.6))
    gray_upper = min(1.0, gray_zone_upper if gray_zone_upper is not None else (threshold + 0.01))
    ax.axvspan(0.0, threshold, color="#DCFCE7", alpha=0.9)
    ax.axvspan(threshold, 1.0, color="#FEE2E2", alpha=0.60)
    ax.axvspan(threshold, gray_upper, color="#FEF3C7", alpha=0.95)
    ax.axvline(threshold, color="#16A34A", linewidth=2.0, linestyle="--")
    ax.scatter([probability], [0.5], color="#DC2626", s=90, zorder=5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("Displayed risk score")
    ax.set_title("Decision Boundary View", pad=10)
    ax.grid(alpha=0.2, axis="x")
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
