import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.ticker import MultipleLocator
from plotly.subplots import make_subplots

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
                color: var(--text-color) !important;
                text-align: center;
            }
            [data-testid="stMetricLabel"] {
                font-size: 0.95rem !important;
                font-weight: 600 !important;
                color: color-mix(in srgb, var(--text-color) 78%, transparent) !important;
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


def _coerce_ecg_signal(signal) -> np.ndarray | None:
    """Normalize ECG payloads loaded from JSON into a 2D NumPy array."""
    if signal is None:
        return None

    try:
        arr = np.asarray(signal, dtype=float)
    except Exception:  # noqa: BLE001
        return None

    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    elif arr.ndim != 2:
        return None

    if arr.size == 0:
        return None

    # Some stored payloads may be lead-major; convert to sample-major.
    if arr.shape[0] <= 12 and arr.shape[1] > arr.shape[0]:
        arr = arr.T

    return arr


def _trim_trailing_quiet_tail(signal: np.ndarray, fs: float) -> np.ndarray:
    """Trim long low-energy tails (typically zero padding) for cleaner ECG display."""
    signal = _coerce_ecg_signal(signal)
    if signal is None:
        return np.zeros((0, 0), dtype=float)

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


def _lead_index_lookup(lead_names: list[str]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for idx, lead in enumerate(lead_names):
        lead_name = str(lead).strip()
        if lead_name:
            lookup[lead_name] = idx
    return lookup


def _ecg_ylim(signal: np.ndarray, lead_indices: list[int]) -> tuple[float, float]:
    if not lead_indices:
        return -1.0, 1.0

    selected = signal[:, lead_indices]
    y_min = float(np.nanmin(selected))
    y_max = float(np.nanmax(selected))

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return -1.0, 1.0

    span = max(y_max - y_min, 0.2)
    pad = max(0.15, span * 0.08)
    y_min = float(np.floor((y_min - pad) / 0.5) * 0.5)
    y_max = float(np.ceil((y_max + pad) / 0.5) * 0.5)

    if y_max - y_min < 1.0:
        center = 0.5 * (y_max + y_min)
        y_min = center - 0.5
        y_max = center + 0.5

    return y_min, y_max


def _apply_ecg_paper_grid(ax, x_max: float, y_min: float, y_max: float) -> None:
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.04))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(which="major", color="#E6A3A3", linewidth=0.60, alpha=0.85)
    ax.grid(which="minor", color="#F7D2D2", linewidth=0.35, alpha=0.85)


def _attention_time_spans(
    lead_name: str,
    highlights: dict[str, list[tuple[int, int]]],
    fs: float,
    x_max: float,
) -> list[tuple[float, float]]:
    if lead_name not in {"V1", "V2", "V3"}:
        return []

    lead_segments = highlights.get(lead_name, []) if isinstance(highlights, dict) else []
    if not isinstance(lead_segments, list):
        return []

    spans: list[tuple[float, float]] = []
    min_width = 0.02
    for segment in lead_segments:
        if not isinstance(segment, (list, tuple)) or len(segment) != 2:
            continue

        start, end = segment
        start_t = max(0.0, float(start) / fs)
        end_t = min(x_max, float(end) / fs)
        if end_t <= start_t:
            continue

        if end_t - start_t < min_width:
            center = 0.5 * (start_t + end_t)
            start_t = max(0.0, center - (min_width / 2.0))
            end_t = min(x_max, center + (min_width / 2.0))

        spans.append((start_t, end_t))

    return spans


def _render_attention_overlay(
    ax,
    lead_name: str,
    highlights: dict[str, list[tuple[int, int]]],
    fs: float,
    x_max: float,
) -> bool:
    rendered = False
    for start_t, end_t in _attention_time_spans(lead_name, highlights, fs, x_max):
        ax.axvspan(start_t, end_t, color="#FB923C", alpha=0.25, zorder=0.2)
        rendered = True

    return rendered

def _plot_12_lead(signal: np.ndarray, lead_names: list[str], fs: float, highlights: dict[str, list[tuple[int, int]]]):
    signal = _coerce_ecg_signal(signal)
    if signal is None or signal.size == 0:
        fig, ax = plt.subplots(figsize=(10, 2.6))
        ax.axis("off")
        ax.text(0.5, 0.5, "ECG signal unavailable for plotting.", ha="center", va="center", fontsize=11)
        return fig

    if fs <= 0:
        fs = 500.0

    signal = _trim_trailing_quiet_tail(signal, fs)
    if signal.size == 0 or signal.ndim != 2:
        fig, ax = plt.subplots(figsize=(10, 2.6))
        ax.axis("off")
        ax.text(0.5, 0.5, "ECG signal unavailable for plotting.", ha="center", va="center", fontsize=11)
        return fig

    n_samples, n_leads = signal.shape
    lead_names = (lead_names + DEFAULT_LEAD_NAMES)[:n_leads]
    lead_lookup = _lead_index_lookup(lead_names)

    t = _seconds_axis(n_samples, fs)
    max_time = float(t[-1]) if n_samples > 1 else (1.0 / float(fs))

    # Standard clinical display order (3 x 4) + long rhythm strip (Lead II).
    clinical_layout = [
        ["I", "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"],
    ]
    plotted_indices = [
        lead_lookup[lead]
        for row in clinical_layout
        for lead in row
        if lead in lead_lookup
    ]
    y_min, y_max = _ecg_ylim(signal, plotted_indices)

    fig = plt.figure(figsize=(18, 11.5), constrained_layout=True)
    gs = fig.add_gridspec(4, 4, height_ratios=[1.0, 1.0, 1.0, 1.2], hspace=0.12, wspace=0.08)

    highlighted_count = 0
    for row_idx, row in enumerate(clinical_layout):
        for col_idx, lead_name in enumerate(row):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            lead_idx = lead_lookup.get(lead_name)
            if lead_idx is None:
                ax.axis("off")
                ax.text(0.5, 0.5, f"{lead_name} unavailable", ha="center", va="center", fontsize=8)
                continue

            ax.plot(t, signal[:, lead_idx], linewidth=0.9, color="#334155", zorder=1.0)
            if _render_attention_overlay(ax, lead_name, highlights, fs, max_time):
                highlighted_count += 1

            _apply_ecg_paper_grid(ax, max_time, y_min, y_max)
            ax.set_title(lead_name, fontsize=10, fontweight="bold", loc="left", pad=2.0)
            ax.tick_params(axis="both", labelsize=7)

            if col_idx == 0:
                ax.set_ylabel("mV", fontsize=8)
            else:
                ax.set_ylabel("")

            if row_idx < 2:
                ax.tick_params(labelbottom=False)

    rhythm_ax = fig.add_subplot(gs[3, :])
    rhythm_idx = lead_lookup.get("II")
    if rhythm_idx is None and lead_lookup:
        rhythm_idx = next(iter(lead_lookup.values()))

    if rhythm_idx is not None:
        rhythm_lead = "II" if "II" in lead_lookup else lead_names[rhythm_idx]
        rhythm_ax.plot(t, signal[:, rhythm_idx], linewidth=0.95, color="#1F2937", zorder=1.0)
        _apply_ecg_paper_grid(rhythm_ax, max_time, y_min, y_max)
        rhythm_ax.set_title(f"Rhythm Strip - Lead {rhythm_lead}", fontsize=10, fontweight="bold", loc="left", pad=3.0)
        rhythm_ax.set_xlabel("Time (s)", fontsize=9)
        rhythm_ax.set_ylabel("mV", fontsize=9)
        rhythm_ax.tick_params(axis="both", labelsize=8)
    else:
        rhythm_ax.axis("off")
        rhythm_ax.text(0.5, 0.5, "Rhythm strip unavailable", ha="center", va="center", fontsize=10)

    if highlighted_count > 0:
        title = "12-lead ECG clinical layout (V1-V3 highlighted attention zones)"
    else:
        title = "12-lead ECG clinical layout"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    return fig


def _plot_12_lead_interactive(signal: np.ndarray, lead_names: list[str], fs: float, highlights: dict[str, list[tuple[int, int]]]):
    def _empty_figure(message: str):
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=message,
            showarrow=False,
            font=dict(size=15, color="#334155"),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(template="plotly_white", height=420, margin=dict(l=20, r=20, t=40, b=20))
        return fig

    signal = _coerce_ecg_signal(signal)
    if signal is None or signal.size == 0:
        return _empty_figure("ECG signal unavailable for plotting.")

    if fs <= 0:
        fs = 500.0

    signal = _trim_trailing_quiet_tail(signal, fs)
    if signal.size == 0 or signal.ndim != 2:
        return _empty_figure("ECG signal unavailable for plotting.")

    n_samples, n_leads = signal.shape
    lead_names = (lead_names + DEFAULT_LEAD_NAMES)[:n_leads]
    lead_lookup = _lead_index_lookup(lead_names)

    t = _seconds_axis(n_samples, fs)
    max_time = float(t[-1]) if n_samples > 1 else (1.0 / float(fs))

    clinical_layout = [
        ["I", "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"],
    ]
    plotted_indices = [
        lead_lookup[lead]
        for row in clinical_layout
        for lead in row
        if lead in lead_lookup
    ]
    y_min, y_max = _ecg_ylim(signal, plotted_indices)

    rhythm_idx = lead_lookup.get("II")
    if rhythm_idx is None and lead_lookup:
        rhythm_idx = next(iter(lead_lookup.values()))
    rhythm_lead = "II" if "II" in lead_lookup else (lead_names[rhythm_idx] if rhythm_idx is not None else "N/A")

    fig = make_subplots(
        rows=4,
        cols=4,
        specs=[
            [{}, {}, {}, {}],
            [{}, {}, {}, {}],
            [{}, {}, {}, {}],
            [{"colspan": 4}, None, None, None],
        ],
        row_heights=[0.22, 0.22, 0.22, 0.34],
        horizontal_spacing=0.02,
        vertical_spacing=0.06,
        subplot_titles=[lead for row in clinical_layout for lead in row] + [f"Rhythm Strip - Lead {rhythm_lead}"],
    )

    highlighted_count = 0
    for row_idx, row in enumerate(clinical_layout, start=1):
        for col_idx, lead_name in enumerate(row, start=1):
            lead_idx = lead_lookup.get(lead_name)
            if lead_idx is None:
                continue

            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=signal[:, lead_idx],
                    mode="lines",
                    line=dict(color="#334155", width=1.05),
                    showlegend=False,
                    hovertemplate=f"{lead_name}<br>Time: %{{x:.3f}} s<br>Amplitude: %{{y:.3f}} mV<extra></extra>",
                ),
                row=row_idx,
                col=col_idx,
            )

            spans = _attention_time_spans(lead_name, highlights, fs, max_time)
            for start_t, end_t in spans:
                fig.add_vrect(
                    x0=start_t,
                    x1=end_t,
                    fillcolor="#FB923C",
                    opacity=0.22,
                    line_width=0,
                    row=row_idx,
                    col=col_idx,
                )
            if spans:
                highlighted_count += 1

            fig.update_xaxes(
                range=[0.0, max_time],
                dtick=0.2,
                showgrid=True,
                gridcolor="#E6A3A3",
                gridwidth=0.65,
                showticklabels=(row_idx == 3),
                title_text="Time (s)" if row_idx == 3 else None,
                row=row_idx,
                col=col_idx,
            )
            fig.update_yaxes(
                range=[y_min, y_max],
                dtick=0.5,
                showgrid=True,
                gridcolor="#E6A3A3",
                gridwidth=0.65,
                title_text="mV" if col_idx == 1 else None,
                row=row_idx,
                col=col_idx,
            )

    if rhythm_idx is not None:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=signal[:, rhythm_idx],
                mode="lines",
                line=dict(color="#1F2937", width=1.1),
                showlegend=False,
                hovertemplate=f"Lead {rhythm_lead}<br>Time: %{{x:.3f}} s<br>Amplitude: %{{y:.3f}} mV<extra></extra>",
            ),
            row=4,
            col=1,
        )

        for start_t, end_t in _attention_time_spans(rhythm_lead, highlights, fs, max_time):
            fig.add_vrect(
                x0=start_t,
                x1=end_t,
                fillcolor="#FB923C",
                opacity=0.22,
                line_width=0,
                row=4,
                col=1,
            )

        fig.update_xaxes(
            range=[0.0, max_time],
            dtick=0.2,
            showgrid=True,
            gridcolor="#E6A3A3",
            gridwidth=0.65,
            title_text="Time (s)",
            row=4,
            col=1,
        )
        fig.update_yaxes(
            range=[y_min, y_max],
            dtick=0.5,
            showgrid=True,
            gridcolor="#E6A3A3",
            gridwidth=0.65,
            title_text="mV",
            row=4,
            col=1,
        )

    title_text = "12-lead ECG clinical layout"
    if highlighted_count > 0:
        title_text = "12-lead ECG clinical layout (V1-V3 highlighted attention zones)"

    fig.update_layout(
        template="plotly_white",
        height=980,
        dragmode="zoom",
        hovermode="x",
        title=dict(text=title_text, x=0.01, xanchor="left"),
        uirevision="ecg_zoom_stable",
        margin=dict(l=30, r=20, t=70, b=30),
    )

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
