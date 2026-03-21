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


def _tier_sort_value(tier: str) -> int:
    mapping = {
        "urgent_cardiology_review": 0,
        "urgent_review_repeat_ecg_quality_check": 1,
        "gray_zone_priority_review": 2,
        "routine_clinical_correlation": 3,
    }
    return mapping.get(str(tier), 9)


def _safe_int(value) -> int:
    try:
        return int(float(value))
    except Exception:  # noqa: BLE001
        return 0


def _normalize_clinician_explain(raw) -> dict:
    if not isinstance(raw, dict):
        return {}
    return raw


def _recommendation_banner(recommendation_tier: str, recommendation_text: str):
    if recommendation_tier in {"urgent_cardiology_review", "urgent_review_repeat_ecg_quality_check"}:
        st.warning(recommendation_text)
    elif recommendation_tier == "gray_zone_priority_review":
        st.info(recommendation_text)
    else:
        st.success(recommendation_text)


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
                clinician_explain = _normalize_clinician_explain(pred.get("clinician_explain", {}))
            else:
                probability = float(getattr(pred, "probability", 0.0))
                label = getattr(pred, "label", "Unknown")
                stability = float(getattr(pred, "decision_stability", 0.0))
                gray_zone = bool(getattr(pred, "gray_zone", False))
                decision_threshold = float(getattr(pred, "decision_threshold", 0.05))
                clinician_explain = _normalize_clinician_explain(getattr(pred, "clinician_explain", {}))

            evidence_counts = clinician_explain.get("evidence_counts", {})
            evidence_strength_summary = (
                f"S{_safe_int(evidence_counts.get('strong', 0))}/"
                f"M{_safe_int(evidence_counts.get('moderate', 0))}/"
                f"W{_safe_int(evidence_counts.get('weak', 0))}"
            )
            recommendation_tier = str(clinician_explain.get("recommendation_tier", "routine_clinical_correlation"))
            mismatch = bool(clinician_explain.get("morphology_model_mismatch", False))

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
                    "morphology_model_mismatch": mismatch,
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "record": base.name,
                    "label": "Inference Failed",
                    "probability": 0.0,
                    "decision_stability": 0.0,
                    "gray_zone": False,
                    "risk": "Unknown",
                    "recommendation_tier": "routine_clinical_correlation",
                    "evidence_strength_summary": "S0/M0/W0",
                    "morphology_model_mismatch": False,
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

    if run_btn and (hea_upload is None or dat_upload is None):
        st.error("Please upload both .hea and .dat files before running diagnosis.")
    elif run_btn:
        try:
            with st.spinner("Running inference pipeline..."):
                record_base = _save_uploaded_pair(hea_upload, dat_upload)
                result = predict_from_record(record_base)

            if isinstance(result, dict):
                label = result.get("label", "Unknown")
                confidence_percent = float(result.get("confidence", 0.0))
                stability_percent = float(result.get("decision_stability", 0.0))
                class_support_percent = float(result.get("class_support", 0.0))
                probability = float(result.get("probability", 0.0))
                gray_zone = bool(result.get("gray_zone", False))
                decision_threshold = float(result.get("decision_threshold", 0.05))
                clinician_explain = _normalize_clinician_explain(result.get("clinician_explain", {}))
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
                class_support_percent = float(getattr(result, "class_support", 0.0))
                probability = float(result.probability)
                gray_zone = bool(getattr(result, "gray_zone", False))
                decision_threshold = float(getattr(result, "decision_threshold", 0.05))
                clinician_explain = _normalize_clinician_explain(getattr(result, "clinician_explain", {}))
                model_contributions = getattr(result, "model_contributions", {}) or {}
                explanation = result.explanation
                signal_plot = result.signal
                fs_plot = float(result.fs)
                lead_names = result.lead_names
                highlights = result.highlighted_segments
                clinical_evidence = getattr(result, "clinical_evidence", [])

            if not isinstance(clinical_evidence, list):
                clinical_evidence = []
            clinical_evidence = [item for item in clinical_evidence if isinstance(item, dict)]

            rec_tier = str(clinician_explain.get("recommendation_tier", "routine_clinical_correlation"))
            rec_text = str(clinician_explain.get("recommendation_text", "Clinical correlation is recommended."))
            evidence_counts = clinician_explain.get("evidence_counts", {})
            next_actions = clinician_explain.get("next_actions", []) if isinstance(clinician_explain.get("next_actions", []), list) else []
            mismatch = bool(clinician_explain.get("morphology_model_mismatch", False))

            if label == "Brugada Syndrome Detected":
                st.error(label)
            else:
                st.success(label)

            if gray_zone:
                st.warning("Gray-zone prediction: probability is close to the decision threshold and needs clinician review.")
                st.caption(f"Borderline-positive zone policy: [{decision_threshold:.3f}, {decision_threshold + 0.01:.3f}].")

            if mismatch:
                st.warning("Discordant case: model decision and V1-V3 morphology strength are not strongly aligned. Prioritize manual review.")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Decision Confidence (derived)", f"{confidence_percent:.1f}%")
            m2.metric("Threshold Distance", f"{stability_percent:.2f} pp")
            m3.metric("Predicted-Class Support", f"{class_support_percent:.1f}%")
            m4.metric("Brugada Risk Probability", f"{probability * 100.0:.2f}%")
            st.caption("Decision Confidence (derived): monotonic transform of threshold distance for easier quick reading.")
            st.caption("Threshold Distance: absolute distance from threshold in percentage points; lower means more borderline.")
            st.caption("Predicted-Class Support: posterior support for the assigned label (can differ from risk probability under low-threshold policy).")
            st.info("Interpretation: prioritize Brugada Risk Probability for risk level and Threshold Distance for borderline assessment; Decision Confidence is the same boundary information in a normalized view.")
            st.write(explanation or "No explanation text returned by model. Please review metrics and clinical evidence.")
            st.caption(f"Raw probability: {probability:.4f} | Threshold: {decision_threshold:.4f}")
            st.caption("Red ECG overlays are shown for Brugada-detected or gray-zone cases.")

            margin_fig = _plot_decision_margin(probability=probability, threshold=decision_threshold)
            st.pyplot(margin_fig, clear_figure=True)

            _recommendation_banner(rec_tier, rec_text)

            if gray_zone or stability_percent <= 1.0:
                st.warning("Borderline Interpretation Protocol")
                st.caption(
                    "This record is close to the decision threshold. Use a conservative review workflow before final clinical action."
                )
                st.write("- Repeat ECG acquisition and verify V1-V3 lead placement.")
                st.write("- Prioritize manual cardiologist over-read for morphology confirmation.")
                st.write("- Correlate with symptoms, syncope history, and family history.")
                st.write("- If uncertainty remains, escalate to urgent specialist review pathway.")

            if clinical_evidence:
                evidence_df = pd.DataFrame(clinical_evidence)
                preferred_cols = ["lead", "source", "tier", "reliability", "j_height", "st_slope", "curvature", "score", "segments"]
                show_cols = [c for c in preferred_cols if c in evidence_df.columns]
                if show_cols:
                    st.caption("Clinical Evidence (V1-V3):")
                    display_df = evidence_df[show_cols].rename(
                        columns={
                            "tier": "evidence_strength",
                            "reliability": "extraction_reliability",
                        }
                    )
                    st.dataframe(display_df, use_container_width=True)

                tier_cols = st.columns(3)
                for idx, lead in enumerate(["V1", "V2", "V3"]):
                    lead_row = evidence_df[evidence_df["lead"] == lead]
                    if lead_row.empty:
                        tier_text = "weak"
                        rel_text = "poor"
                    else:
                        tier_text = str(lead_row.iloc[0].get("tier", "weak"))
                        rel_text = str(lead_row.iloc[0].get("reliability", "poor"))
                    tier_cols[idx].markdown(f"**{lead} Evidence Strength**")
                    tier_cols[idx].markdown(f"### {tier_text.title()}")
                    tier_cols[idx].caption(f"Extraction Reliability: {rel_text.title()}")

                heatmap_fig = _plot_evidence_heatmap(evidence_df)
                st.pyplot(heatmap_fig, clear_figure=True)
                st.caption("Evidence Strength legend: strong=clear morphology support, moderate=partial support, weak=limited support.")
                st.caption("Extraction Reliability legend: good=delineation robust, fair=usable but less robust, poor=insufficient delineation.")

            if model_contributions and sum(float(v) for v in model_contributions.values()) > 0:
                st.caption("Deep-view contribution share (embedding norm %):")
                contrib_df = pd.DataFrame(
                    {
                        "model": list(model_contributions.keys()),
                        "share": list(model_contributions.values()),
                    }
                ).sort_values("share", ascending=False)
                st.bar_chart(contrib_df.set_index("model"))

            st.subheader("Recommended Clinical Actions")
            st.caption("Action checklist")
            st.caption(
                f"Tier: {rec_tier} | Evidence S/M/W: "
                f"{_safe_int(evidence_counts.get('strong', 0))}/"
                f"{_safe_int(evidence_counts.get('moderate', 0))}/"
                f"{_safe_int(evidence_counts.get('weak', 0))}"
            )
            for action in next_actions:
                if isinstance(action, str) and action.strip():
                    st.write(f"- {action}")
            st.caption("Clinical note: AI output supports triage and does not replace physician diagnosis.")

            evidence_segments = sum(_safe_int(item.get("segments", 0)) for item in clinical_evidence)
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
                    df = df.sort_values(["_tier_rank", "probability", "decision_stability"], ascending=[True, False, False]).drop(columns=["_tier_rank"])
                else:
                    df = df.sort_values("probability", ascending=False)
                st.dataframe(df, use_container_width=True)

                if "recommendation_tier" in df.columns:
                    urgent = df[df["recommendation_tier"].isin(["urgent_cardiology_review", "urgent_review_repeat_ecg_quality_check"])].sort_values("probability", ascending=False)
                else:
                    urgent = df.iloc[0:0]

                if "gray_zone" in df.columns:
                    gray_queue = df[df["gray_zone"] == True]
                else:
                    gray_queue = df.iloc[0:0]

                if "morphology_model_mismatch" in df.columns:
                    discordant = df[df["morphology_model_mismatch"] == True]
                else:
                    discordant = df.iloc[0:0]

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

                st.subheader("Discordant Cases Queue")
                if discordant.empty:
                    st.info("No model-morphology discordant records detected in this batch.")
                else:
                    st.dataframe(discordant[["record", "probability", "decision_stability", "label", "recommendation_tier"]], use_container_width=True)
