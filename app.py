from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from inference import predict_from_record


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


def _plot_decision_margin(probability: float, threshold: float):
    fig, ax = plt.subplots(figsize=(8.0, 1.8))
    ax.axvspan(0.0, threshold, color="#DCFCE7", alpha=0.9, label="Below threshold")
    ax.axvspan(threshold, max(0.20, probability + 0.05), color="#FEE2E2", alpha=0.65, label="Above threshold")
    ax.axvline(threshold, color="#16A34A", linewidth=2.0, linestyle="--", label=f"Threshold {threshold:.3f}")
    ax.scatter([probability], [0.5], color="#DC2626", s=90, zorder=5, label=f"Prediction {probability:.3f}")
    ax.set_xlim(0.0, max(0.20, probability + 0.05))
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
            rows.append([0.0, 0.0, 0.0, 0.0])
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
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="#111827", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Feature intensity")
    fig.tight_layout()
    return fig


def _tier_sort_value(tier: str) -> int:
    mapping = {
        "urgent_cardiology_review": 0,
        "urgent_review_repeat_ecg_quality_check": 1,
        "gray_zone_priority_review": 2,
        "routine_clinical_correlation": 3,
    }
    return mapping.get(str(tier), 9)


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


def _predict_batch_from_folder(folder_path: Path) -> list[dict]:
    results = []
    for hea_file in sorted(folder_path.glob("*.hea")):
        base = hea_file.with_suffix("")
        dat_file = base.with_suffix(".dat")
        if not dat_file.exists():
            continue

        try:
            pred = predict_from_record(str(base))
            if isinstance(pred, dict):
                probability = float(pred.get("probability", 0.0))
                label = pred.get("label", "Unknown")
                stability = float(pred.get("decision_stability", 0.0))
                gray_zone = bool(pred.get("gray_zone", False))
                decision_threshold = float(pred.get("decision_threshold", 0.05))
                clinician_explain = pred.get("clinician_explain", {}) or {}
            else:
                probability = float(getattr(pred, "probability", 0.0))
                label = getattr(pred, "label", "Unknown")
                stability = float(getattr(pred, "decision_stability", 0.0))
                gray_zone = bool(getattr(pred, "gray_zone", False))
                decision_threshold = float(getattr(pred, "decision_threshold", 0.05))
                clinician_explain = getattr(pred, "clinician_explain", {}) or {}

            evidence_counts = clinician_explain.get("evidence_counts", {}) if isinstance(clinician_explain, dict) else {}
            evidence_strength_summary = (
                f"S{int(evidence_counts.get('strong', 0))}/"
                f"M{int(evidence_counts.get('moderate', 0))}/"
                f"W{int(evidence_counts.get('weak', 0))}"
            )
            recommendation_tier = str(clinician_explain.get("recommendation_tier", "routine_clinical_correlation"))

            results.append(
                {
                    "record": base.name,
                    "label": label,
                    "probability": probability,
                    "decision_stability": stability,
                    "gray_zone": gray_zone,
                    "risk": "High" if probability >= decision_threshold else "Low",
                    "recommendation_tier": recommendation_tier,
                    "evidence_strength_summary": evidence_strength_summary,
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "record": base.name,
                    "label": "Inference Failed",
                    "probability": 0.0,
                    "risk": "Unknown",
                    "error": str(exc),
                }
            )
    return results


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

            if isinstance(result, dict):
                label = result.get("label", "Unknown")
                confidence_percent = float(result.get("confidence", 0.0))
                stability_percent = float(result.get("decision_stability", 0.0))
                probability = float(result.get("probability", 0.0))
                gray_zone = bool(result.get("gray_zone", False))
                decision_threshold = float(result.get("decision_threshold", 0.05))
                clinician_explain = result.get("clinician_explain", {}) or {}
                model_contributions = result.get("model_contributions", {}) or {}
                explanation = result.get(
                    "explanation",
                    "Model prediction generated. Clinical correlation is recommended.",
                )
                signal_plot = result.get("signal_for_plot")
                fs_plot = float(result.get("fs", 500.0))
                lead_names = result.get("lead_names", DEFAULT_LEAD_NAMES)
                highlights = result.get("highlighted_segments", {})
                clinical_evidence = result.get("clinical_evidence", [])
            else:
                label = result.label
                confidence_percent = float(result.confidence_percent)
                stability_percent = float(getattr(result, "decision_stability", 0.0))
                probability = float(result.probability)
                gray_zone = bool(getattr(result, "gray_zone", False))
                decision_threshold = float(getattr(result, "decision_threshold", 0.05))
                clinician_explain = getattr(result, "clinician_explain", {}) or {}
                model_contributions = getattr(result, "model_contributions", {}) or {}
                explanation = result.explanation
                signal_plot = result.signal
                fs_plot = float(result.fs)
                lead_names = result.lead_names
                highlights = result.highlighted_segments
                clinical_evidence = getattr(result, "clinical_evidence", [])

            if label == "Brugada Syndrome Detected":
                st.error(label)
            else:
                st.success(label)

            if gray_zone:
                st.warning("Gray-zone prediction: probability is close to the decision threshold and needs clinician review.")

            st.metric("AI Class Confidence", f"{confidence_percent:.1f}%")
            st.metric("Decision Stability", f"{stability_percent:.1f}%")
            st.metric("Brugada Risk Probability", f"{probability * 100.0:.2f}%")
            st.caption("AI Class Confidence: certainty of the predicted class, not the same as disease probability.")
            st.caption("Decision Stability: distance from decision threshold; lower values indicate a borderline call.")
            st.info("Interpretation: Probability quantifies Brugada risk, confidence quantifies class certainty, and stability quantifies boundary distance.")
            st.write(explanation)
            st.caption(f"Raw probability: {probability:.4f} | Threshold: {decision_threshold:.4f}")
            st.caption("Red ECG overlays are shown for Brugada-detected or gray-zone cases.")

            margin_fig = _plot_decision_margin(probability=probability, threshold=decision_threshold)
            st.pyplot(margin_fig, clear_figure=True)

            if label == "Brugada Syndrome Detected" and not gray_zone:
                st.warning("Triage suggestion: urgent cardiology review is recommended.")
            elif gray_zone:
                st.info("Triage suggestion: prioritize manual over-read and consider repeat ECG.")
            else:
                st.info("Triage suggestion: low AI risk; correlate with symptoms and routine clinical workflow.")

            if clinical_evidence:
                evidence_df = pd.DataFrame(clinical_evidence)
                preferred_cols = ["lead", "source", "tier", "reliability", "j_height", "st_slope", "curvature", "score", "segments"]
                show_cols = [c for c in preferred_cols if c in evidence_df.columns]
                if show_cols:
                    st.caption("Clinical Evidence (V1-V3):")
                    st.dataframe(evidence_df[show_cols], use_container_width=True)

                tier_cols = st.columns(3)
                for idx, lead in enumerate(["V1", "V2", "V3"]):
                    lead_row = evidence_df[evidence_df["lead"] == lead]
                    if lead_row.empty:
                        tier_text = "weak"
                        rel_text = "poor"
                    else:
                        tier_text = str(lead_row.iloc[0].get("tier", "weak"))
                        rel_text = str(lead_row.iloc[0].get("reliability", "poor"))
                    tier_cols[idx].metric(f"{lead} evidence", tier_text.title(), rel_text.title())

                heatmap_fig = _plot_evidence_heatmap(evidence_df)
                st.pyplot(heatmap_fig, clear_figure=True)

            if model_contributions:
                st.caption("Deep-view contribution share (embedding norm %):")
                contrib_df = pd.DataFrame(
                    {
                        "model": list(model_contributions.keys()),
                        "share": list(model_contributions.values()),
                    }
                ).sort_values("share", ascending=False)
                st.bar_chart(contrib_df.set_index("model"))

            rec_tier = str(clinician_explain.get("recommendation_tier", "routine_clinical_correlation"))
            rec_text = str(clinician_explain.get("recommendation_text", "Clinical correlation is recommended."))
            evidence_counts = clinician_explain.get("evidence_counts", {}) if isinstance(clinician_explain, dict) else {}
            next_actions = clinician_explain.get("next_actions", []) if isinstance(clinician_explain, dict) else []
            mismatch = bool(clinician_explain.get("morphology_model_mismatch", False)) if isinstance(clinician_explain, dict) else False

            st.subheader("Recommended Clinical Actions")
            st.warning(rec_text)
            st.caption(
                f"Tier: {rec_tier} | Evidence S/M/W: "
                f"{int(evidence_counts.get('strong', 0))}/"
                f"{int(evidence_counts.get('moderate', 0))}/"
                f"{int(evidence_counts.get('weak', 0))}"
            )
            for action in next_actions:
                st.write(f"- {action}")
            if mismatch:
                st.warning("Evidence mismatch: model decision and morphology strength are not strongly aligned. Prioritize manual review.")
            st.caption("Clinical note: AI output supports triage and does not replace physician diagnosis.")

            evidence_segments = sum(int(item.get("segments", 0)) for item in clinical_evidence)
            if probability >= decision_threshold and evidence_segments == 0:
                st.warning("High-risk prediction with no V1-V3 morphological evidence. Manual cardiology review is recommended.")

            st.caption(f"Predicted Brugada probability: {probability:.3f}")

            plot_highlights = highlights if (label == "Brugada Syndrome Detected" or gray_zone) else {}

            ecg_fig = _plot_12_lead(
                signal=signal_plot,
                lead_names=lead_names,
                fs=fs_plot,
                highlights=plot_highlights,
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
                batch_results = _predict_batch_from_folder(batch_dir)

            if not batch_results:
                st.warning("No valid WFDB pairs found in batch uploads.")
            else:
                df = pd.DataFrame(batch_results)
                if "recommendation_tier" in df.columns:
                    df["_tier_rank"] = df["recommendation_tier"].map(_tier_sort_value)
                    df = df.sort_values(["_tier_rank", "probability"], ascending=[True, False]).drop(columns=["_tier_rank"])
                else:
                    df = df.sort_values("probability", ascending=False)
                st.dataframe(df, use_container_width=True)

                if "recommendation_tier" in df.columns:
                    urgent = df[df["recommendation_tier"] == "urgent_cardiology_review"]
                else:
                    urgent = df.iloc[0:0]

                if "gray_zone" in df.columns:
                    gray_queue = df[df["gray_zone"] == True]
                else:
                    gray_queue = df.iloc[0:0]

                st.subheader("Urgent Review Queue")
                if urgent.empty:
                    st.info("No urgent-cardiology records in this batch.")
                else:
                    st.dataframe(urgent[["record", "probability", "decision_stability", "label", "evidence_strength_summary"]], use_container_width=True)

                st.subheader("Gray-Zone Priority Queue")
                if gray_queue.empty:
                    st.info("No gray-zone records detected in this batch.")
                else:
                    st.dataframe(gray_queue[["record", "probability", "decision_stability", "label", "recommendation_tier"]], use_container_width=True)
