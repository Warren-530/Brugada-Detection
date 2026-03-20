from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from inference import DEFAULT_LEAD_NAMES, predict_batch_from_folder, predict_from_record


st.set_page_config(page_title="Brugada AI Assistant", page_icon="ECG", layout="wide")
st.title("Brugada Syndrome Clinical AI Assistant")
st.caption("Multi-view deep feature stacking + meta-learner for single-patient ECG triage")


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


def _save_uploaded_pair(hea_file, dat_file) -> Path:
    if hea_file is None or dat_file is None:
        raise ValueError("Both .hea and .dat files are required")

    record_name = Path(hea_file.name).stem
    temp_dir = Path(tempfile.mkdtemp(prefix="brugada_record_"))

    hea_path = temp_dir / f"{record_name}.hea"
    dat_path = temp_dir / f"{record_name}.dat"

    hea_path.write_bytes(hea_file.getbuffer())
    dat_path.write_bytes(dat_file.getbuffer())

    return temp_dir / record_name


def _save_batch_folder(uploaded_files: list) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="brugada_batch_"))
    for f in uploaded_files:
        (temp_dir / f.name).write_bytes(f.getbuffer())
    return temp_dir


left, right = st.columns([1, 2])

with left:
    st.subheader("Patient Input")
    hea_upload = st.file_uploader("Upload .hea", type=["hea"], key="hea")
    dat_upload = st.file_uploader("Upload .dat", type=["dat"], key="dat")

    run_btn = st.button("Run Diagnosis", type="primary", use_container_width=True)

    st.markdown("---")
    st.subheader("Batch Evaluation (Optional)")
    batch_uploads = st.file_uploader(
        "Upload multiple .hea/.dat files",
        type=["hea", "dat"],
        accept_multiple_files=True,
        key="batch",
    )
    batch_btn = st.button("Run Batch Risk List", use_container_width=True)

with right:
    st.subheader("Clinical Report")

    if run_btn:
        try:
            with st.spinner("Running inference pipeline..."):
                record_base = _save_uploaded_pair(hea_upload, dat_upload)
                result = predict_from_record(record_base)

            if result.label == "Brugada Syndrome Detected":
                st.error(result.label)
            else:
                st.success(result.label)

            st.metric("Probability Score", f"{result.confidence_percent:.1f}%")
            st.write(result.explanation)

            feat_df = pd.DataFrame(
                {
                    "Extractor": list(result.feature_importance.keys()),
                    "Feature Count": list(result.feature_importance.values()),
                }
            ).sort_values("Feature Count", ascending=False)
            st.bar_chart(feat_df.set_index("Extractor"))

            ecg_fig = _plot_12_lead(
                signal=result.signal,
                lead_names=result.lead_names,
                fs=result.fs,
                highlights=result.highlighted_segments,
            )
            st.pyplot(ecg_fig, clear_figure=True)
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)

    if batch_btn:
        if not batch_uploads:
            st.warning("Please upload paired .hea/.dat files for batch evaluation.")
        else:
            with st.spinner("Scoring all uploaded records..."):
                batch_dir = _save_batch_folder(batch_uploads)
                batch_results = predict_batch_from_folder(batch_dir)

            if not batch_results:
                st.warning("No valid WFDB pairs found in batch uploads.")
            else:
                df = pd.DataFrame(batch_results).sort_values("probability", ascending=False)
                st.dataframe(df, use_container_width=True)

                high_risk = df[df["risk"] == "High"]
                st.subheader("High-Risk Patients")
                if high_risk.empty:
                    st.info("No high-risk records detected at current threshold.")
                else:
                    st.dataframe(high_risk[["record", "probability", "label"]], use_container_width=True)
