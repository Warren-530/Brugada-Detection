import re

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

from brugada.ui.components import (
    DEFAULT_LEAD_NAMES,
    SVG_INFO,
    SVG_WARNING,
    _attention_time_spans,
    _coerce_ecg_signal,
    _ecg_ylim,
    _lead_index_lookup,
    _seconds_axis,
    _trim_trailing_quiet_tail,
)


def _get_val(result, key, default=None):
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _ordered_leads(available_leads: list[str]) -> list[str]:
    canonical_order = [
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

    seen = set()
    ordered = []

    for lead in canonical_order:
        if lead in available_leads and lead not in seen:
            ordered.append(lead)
            seen.add(lead)

    for lead in available_leads:
        if lead not in seen:
            ordered.append(lead)
            seen.add(lead)

    return ordered


def _plot_lead_preview(signal, t_axis, lead_idx: int, lead_name: str, fs: float, highlights, y_min: float, y_max: float):
    fig, ax = plt.subplots(figsize=(2.8, 1.4))
    ax.plot(t_axis, signal[:, lead_idx], linewidth=0.85, color="#334155")

    x_max = float(t_axis[-1]) if len(t_axis) > 1 else (1.0 / fs)
    for start_t, end_t in _attention_time_spans(lead_name, highlights, fs, x_max):
        ax.axvspan(start_t, end_t, color="#FB923C", alpha=0.24)

    ax.set_xlim(0.0, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(lead_name, fontsize=9, loc="left", pad=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.14)
    fig.tight_layout(pad=0.3)
    return fig


def _plot_lead_focus(signal, t_axis, lead_idx: int, lead_name: str, fs: float, highlights):
    y_min, y_max = _ecg_ylim(signal, [lead_idx])
    x_max = float(t_axis[-1]) if len(t_axis) > 1 else (1.0 / fs)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_axis,
            y=signal[:, lead_idx],
            mode="lines",
            line=dict(color="#1F2937", width=1.2),
            showlegend=False,
            hovertemplate=f"{lead_name}<br>Time: %{{x:.3f}} s<br>Amplitude: %{{y:.3f}} mV<extra></extra>",
        )
    )

    for start_t, end_t in _attention_time_spans(lead_name, highlights, fs, x_max):
        fig.add_vrect(
            x0=start_t,
            x1=end_t,
            fillcolor="#FB923C",
            opacity=0.22,
            line_width=0,
        )

    fig.update_xaxes(
        range=[0.0, x_max],
        dtick=0.2,
        showgrid=True,
        gridcolor="#E6A3A3",
        gridwidth=0.65,
        title_text="Time (s)",
    )
    fig.update_yaxes(
        range=[y_min, y_max],
        dtick=0.5,
        showgrid=True,
        gridcolor="#E6A3A3",
        gridwidth=0.65,
        title_text="mV",
    )

    fig.update_layout(
        template="plotly_white",
        height=560,
        dragmode="zoom",
        hovermode="x",
        uirevision=f"lead_focus_{lead_name}",
        title=dict(text=f"Focused Lead Review: {lead_name}", x=0.01, xanchor="left"),
        margin=dict(l=35, r=20, t=65, b=35),
    )
    return fig


def render_ecg_review_tab(single_result_to_show, is_batch: bool, current_view: str):
    st.subheader("ECG Lead Review")
    st.caption("Dedicated diagnostic reading page for per-lead magnified inspection.")

    if single_result_to_show is None:
        if is_batch and current_view == "Batch Summary":
            st.markdown(
                f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #e0f2fe; color: #075985; display: flex; align-items: center;'>{SVG_INFO} Select a specific record in Clinical Report first, then come back here for per-lead review.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #e0f2fe; color: #075985; display: flex; align-items: center;'>{SVG_INFO} Run diagnosis first to open the dedicated per-lead review page.</div>",
                unsafe_allow_html=True,
            )
        return

    signal = _get_val(single_result_to_show, "signal_for_plot", _get_val(single_result_to_show, "signal", None))
    fs = float(_get_val(single_result_to_show, "fs", 500.0) or 500.0)
    lead_names = _get_val(single_result_to_show, "lead_names", DEFAULT_LEAD_NAMES)
    highlights = _get_val(single_result_to_show, "highlighted_segments", {})
    label = str(_get_val(single_result_to_show, "label", "Unknown"))

    signal = _coerce_ecg_signal(signal)
    if signal is None or signal.size == 0:
        st.markdown(
            f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fef3c7; color: #92400e; display: flex; align-items: center;'>{SVG_WARNING} ECG signal is unavailable for lead-by-lead review.</div>",
            unsafe_allow_html=True,
        )
        return

    signal = _trim_trailing_quiet_tail(signal, fs)
    if signal.size == 0 or signal.ndim != 2:
        st.markdown(
            f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fef3c7; color: #92400e; display: flex; align-items: center;'>{SVG_WARNING} ECG signal quality is insufficient for detailed review.</div>",
            unsafe_allow_html=True,
        )
        return

    n_samples, n_leads = signal.shape
    if isinstance(lead_names, (list, tuple)):
        lead_names = [str(item) for item in lead_names]
    else:
        lead_names = list(DEFAULT_LEAD_NAMES)
    lead_names = (lead_names + DEFAULT_LEAD_NAMES)[:n_leads]

    lead_lookup = _lead_index_lookup(lead_names)
    if not lead_lookup:
        st.warning("No valid lead names are available for review.")
        return

    ordered_leads = _ordered_leads(list(lead_lookup.keys()))
    t_axis = _seconds_axis(n_samples, fs)
    all_indices = [lead_lookup[name] for name in ordered_leads]
    y_min, y_max = _ecg_ylim(signal, all_indices)

    review_scope = str(_get_val(single_result_to_show, "record_uid", _get_val(single_result_to_show, "record", "single")))
    review_scope = re.sub(r"[^0-9A-Za-z_-]", "_", review_scope)
    selected_lead_key = f"ecg_selected_lead_{review_scope}"

    if selected_lead_key not in st.session_state or st.session_state[selected_lead_key] not in lead_lookup:
        st.session_state[selected_lead_key] = "II" if "II" in lead_lookup else ordered_leads[0]

    st.markdown(
        f"<div style='margin-bottom: 0.8rem; padding: 0.7rem; border-radius: 0.5rem; background-color: #f8fafc; color: #334155;'>Current interpretation: <strong>{label}</strong> | Click any lead card below to open a magnified diagnostic view.</div>",
        unsafe_allow_html=True,
    )

    preview_cols = st.columns(4)
    for idx, lead_name in enumerate(ordered_leads):
        col = preview_cols[idx % 4]
        with col:
            preview_fig = _plot_lead_preview(
                signal=signal,
                t_axis=t_axis,
                lead_idx=lead_lookup[lead_name],
                lead_name=lead_name,
                fs=fs,
                highlights=highlights,
                y_min=y_min,
                y_max=y_max,
            )
            st.pyplot(preview_fig, clear_figure=True)
            if st.button(f"Inspect {lead_name}", key=f"inspect_lead_{review_scope}_{lead_name}", use_container_width=True):
                st.session_state[selected_lead_key] = lead_name

    selected_lead = st.session_state[selected_lead_key]
    selected_index = lead_lookup[selected_lead]

    st.markdown("### Magnified Lead View")
    st.caption(f"Focused lead: {selected_lead}. Drag to zoom, scroll to scale, double-click to reset.")
    focus_fig = _plot_lead_focus(
        signal=signal,
        t_axis=t_axis,
        lead_idx=selected_index,
        lead_name=selected_lead,
        fs=fs,
        highlights=highlights,
    )
    st.plotly_chart(
        focus_fig,
        use_container_width=True,
        key=f"lead_focus_plot_{review_scope}_{selected_lead}",
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "doubleClick": "reset",
            "responsive": True,
        },
    )
