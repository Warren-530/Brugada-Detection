from datetime import datetime, timezone
import pandas as pd
import re
import numpy as np
import streamlit as st

from brugada.analytics import find_similar_cases, normalize_result_snapshot
from brugada.export import build_batch_html_zip, build_single_case_html_report
from brugada.inference.pipeline import predict_from_record
from brugada.storage.record_store import get_record_payload, list_records, save_batch_results, save_record_result
from brugada.ui.components import (
    DEFAULT_LEAD_NAMES,
    SVG_ERROR,
    SVG_INFO,
    SVG_SUCCESS,
    SVG_WARNING,
    _plot_12_lead,
    _plot_decision_margin,
    _plot_evidence_heatmap,
    _recommendation_banner,
    _render_decision_margin_legend,
)
from brugada.ui.helpers import render_metric_with_info, format_recommendation_tier
from brugada.file_utils import (
    _normalize_clinician_explain,
    _predict_batch_from_folder,
    _safe_int,
    _save_batch_folder,
    _save_uploaded_pair,
    _tier_sort_value,
)


def render_clinical_report_tab(
    pairs: dict,
    uploaded_files,
    run_btn: bool,
    is_batch: bool,
    patient_id: str | None,
    batch_patient_id_map: dict[str, str] | None = None,
):
    st.subheader("Clinical Report")

    if st.session_state.persistence_notice:
        st.info(st.session_state.persistence_notice)
        st.session_state.persistence_notice = ""

    is_single = len(pairs) == 1

    if run_btn:
        if not pairs:
            st.markdown(
                f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fee2e2; color: #991b1b; display: flex; align-items: center;'>{SVG_ERROR} Please upload at least one valid .hea and .dat pair before running diagnosis.</div>",
                unsafe_allow_html=True,
            )
        elif is_single:
            stem = list(pairs.keys())[0]
            hea_upload = pairs[stem]["hea"]
            dat_upload = pairs[stem]["dat"]
            try:
                with st.spinner("Running inference pipeline..."):
                    record_base = _save_uploaded_pair(hea_upload, dat_upload)
                    result = predict_from_record(str(record_base))
                    st.session_state.records_loaded_result = None
                    st.session_state.last_ml_result = result

                    if st.session_state.record_store_ready:
                        try:
                            record_uid = save_record_result(
                                record_name=stem,
                                result=result,
                                source_mode="single",
                                patient_id=patient_id,
                            )
                            if isinstance(result, dict):
                                result["record_uid"] = record_uid
                            st.session_state.persistence_notice = f"Saved record '{stem}' to local Records Center."
                        except Exception as persist_exc:  # noqa: BLE001
                            st.session_state.persistence_notice = (
                                f"Diagnosis completed, but local save failed: {persist_exc}"
                            )

                    if st.session_state.chatbot_ready:
                        st.session_state.chatbot.reset_conversation()
                        st.session_state.conversation_history = []
                    st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)
        else:
            with st.spinner("Scoring all uploaded records..."):
                batch_dir = _save_batch_folder(uploaded_files)
                batch_results = _predict_batch_from_folder(batch_dir)
                st.session_state.records_loaded_result = None
                st.session_state.batch_results = batch_results

                if st.session_state.record_store_ready:
                    try:
                        uid_map = save_batch_results(
                            batch_results,
                            patient_id=patient_id,
                            patient_id_by_record=batch_patient_id_map,
                        )
                        st.session_state.batch_record_uid_map = uid_map
                        st.session_state.persistence_notice = (
                            f"Saved {len(uid_map)}/{len(batch_results)} batch records to local Records Center."
                        )
                    except Exception as persist_exc:  # noqa: BLE001
                        st.session_state.persistence_notice = (
                            f"Batch diagnosis completed, but local save failed: {persist_exc}"
                        )

                st.rerun()

    current_view = st.session_state.get("current_view", "Batch Summary")
    if current_view in st.session_state.deleted_pairs:
        current_view = "Batch Summary"
        st.session_state.current_view = current_view

    batch_results = st.session_state.get("batch_results")
    if isinstance(batch_results, list) and batch_results:
        record_views: list[str] = []
        seen_records: set[str] = set()
        for item in batch_results:
            if not isinstance(item, dict):
                continue
            record_name = str(item.get("record", "")).strip()
            if not record_name:
                continue
            if record_name in st.session_state.deleted_pairs:
                continue
            if record_name in seen_records:
                continue
            seen_records.add(record_name)
            record_views.append(record_name)

        if record_views:
            view_options = ["Batch Summary", *record_views]
            if current_view not in view_options:
                current_view = "Batch Summary"
                st.session_state.current_view = current_view

            selected_view = st.selectbox(
                "Batch record view",
                options=view_options,
                index=view_options.index(current_view),
                key="clinical_report_batch_view_selector",
                help="Switch between batch summary and individual record reports.",
            )
            if selected_view != current_view:
                current_view = selected_view
                st.session_state.current_view = current_view
            st.caption("Choose a specific record from the sidebar arrow keys or dropdown selector.")

    single_result_to_show = None
    has_results = False

    # Priority: if viewing batch results and user selected specific record, show that record
    batch_results = st.session_state.get("batch_results")
    if batch_results and current_view != "Batch Summary":
        for res in batch_results:
            if res["record"] == current_view:
                single_result_to_show = res.get("raw")
                has_results = True
                break
    
    # Fall back to loaded record result
    if single_result_to_show is None:
        loaded_record_result = st.session_state.get("records_loaded_result")
        if loaded_record_result is not None:
            single_result_to_show = loaded_record_result
            has_results = True
    
    # Fall back to single ML result (non-batch mode)
    if single_result_to_show is None and not is_batch and st.session_state.last_ml_result is not None:
        single_result_to_show = st.session_state.last_ml_result
        has_results = True
    
    # Fall back to any batch results (for Batch Summary view)
    if single_result_to_show is None and batch_results:
        has_results = True

    if not has_results:
        st.markdown(
            f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #e0f2fe; color: #075985; display: flex; align-items: center;'><span style='margin-right: 0.5rem; display: flex;'>{SVG_INFO}</span> Run diagnosis to see results in the clinical report tab.</div>",
            unsafe_allow_html=True,
        )

    if single_result_to_show is not None:
        result = single_result_to_show
        if isinstance(result, dict):
            label = result.get("label", "Unknown")
            operational_probability = float(result.get("probability", 0.0))
            operational_threshold = float(result.get("decision_threshold", 0.05))
            probability = float(result.get("display_probability", operational_probability))
            decision_threshold = float(result.get("display_threshold", 0.35))
            gray_zone_upper = float(result.get("display_gray_zone_upper", min(1.0, decision_threshold + 0.01)))
            confidence_percent = float(result.get("display_confidence", result.get("confidence", 0.0)))
            stability_percent = float(result.get("display_decision_stability", result.get("decision_stability", 0.0)))
            class_support_percent = float(result.get("class_support", 0.0))
            gray_zone = bool(result.get("gray_zone", False))
            clinician_explain = _normalize_clinician_explain(result.get("clinician_explain", {}))
            model_contributions = result.get("model_contributions", {}) or {}
            explanation = result.get("explanation", "Model prediction generated. Clinical correlation is recommended.")
            signal_plot = result.get("signal_for_plot")
            fs_plot = float(result.get("fs", 500.0))
            lead_names = result.get("lead_names", DEFAULT_LEAD_NAMES)
            highlights = result.get("highlighted_segments", {})
            clinical_evidence = result.get("clinical_evidence", [])
        else:
            label = result.label
            operational_probability = float(getattr(result, "probability", 0.0))
            operational_threshold = float(getattr(result, "decision_threshold", 0.05))
            probability = float(getattr(result, "display_probability", operational_probability))
            decision_threshold = float(getattr(result, "display_threshold", 0.35))
            gray_zone_upper = float(getattr(result, "display_gray_zone_upper", min(1.0, decision_threshold + 0.01)))
            confidence_percent = float(getattr(result, "display_confidence", getattr(result, "confidence_percent", 0.0)))
            stability_percent = float(getattr(result, "display_decision_stability", getattr(result, "decision_stability", 0.0)))
            class_support_percent = float(getattr(result, "class_support", 0.0))
            gray_zone = bool(getattr(result, "gray_zone", False))
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
        rec_tier = format_recommendation_tier(rec_tier)
        rec_text = str(clinician_explain.get("recommendation_text", "Clinical correlation is recommended."))
        evidence_counts = clinician_explain.get("evidence_counts", {})
        next_actions = clinician_explain.get("next_actions", []) if isinstance(clinician_explain.get("next_actions", []), list) else []
        mismatch = bool(clinician_explain.get("morphology_model_mismatch", False))

        # Display record number when viewing individual record from batch
        if current_view != "Batch Summary":
            st.markdown(
                f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #f0f9ff; color: #0c4a6e; font-weight: 600; font-size: 1.05em;'>Record: <span style='font-family: monospace; color: #0369a1;'>{current_view}</span></div>",
                unsafe_allow_html=True,
            )

        if label == "Brugada Syndrome Detected":
            st.markdown(
                f"<div style='margin-bottom: 1rem; padding: 1rem; border-radius: 0.5rem; background-color: #fee2e2; color: #991b1b; display: flex; align-items: center; font-weight: bold; font-size: 1.1em;'>{SVG_ERROR} {label}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='margin-bottom: 1rem; padding: 1rem; border-radius: 0.5rem; background-color: #dcfce7; color: #166534; display: flex; align-items: center; font-weight: bold; font-size: 1.1em;'>{SVG_SUCCESS} {label}</div>",
                unsafe_allow_html=True,
            )

        _recommendation_banner(rec_tier, rec_text)

        if gray_zone:
            st.markdown(
                f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fef3c7; color: #92400e; display: flex; align-items: center;'>{SVG_WARNING} Gray-zone prediction: score is close to the 0.35 boundary and requires cardiology review.</div>",
                unsafe_allow_html=True,
            )
        if mismatch:
            st.markdown(
                f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fef3c7; color: #92400e; display: flex; align-items: center;'>{SVG_WARNING} Discordant case: model decision and V1-V3 morphology strength are not strongly aligned. Prioritize manual review.</div>",
                unsafe_allow_html=True,
            )

        risk_pct = float(probability * 100.0)
        st.markdown("### Diagnostic Snapshot")
        m1, m2, m3 = st.columns(3)
        render_metric_with_info(
            m1,
            "Brugada Risk Score",
            f"{risk_pct:.2f}%",
            "**What it means**\n"
            "- Report-aligned risk score shown on the UI scale.\n\n"
            "**How it is derived**\n"
            "- Monotonic remap of raw model probability.\n"
            "- Keeps ordering and decision consistency while mapping 0.05 -> 0.35 for display.",
            "metric_info_risk",
        )
        render_metric_with_info(
            m2,
            "Decision Confidence",
            f"{confidence_percent:.1f}%",
            "**What it means**\n"
            "- Boundary-aware certainty of the current decision.\n\n"
            "**How it is derived**\n"
            "- Uses distance to the display threshold (0.35).\n"
            "- Near boundary -> closer to 50%; farther away -> closer to 100%.",
            "metric_info_confidence",
        )
        render_metric_with_info(
            m3,
            "Prediction Support",
            f"{class_support_percent:.1f}%",
            "**What it means**\n"
            "- Posterior support for the predicted class.\n\n"
            "**How it is derived**\n"
            "- If predicted Brugada: support = P(Brugada).\n"
            "- If predicted Normal: support = 1 - P(Brugada).",
            "metric_info_support",
        )

        threshold_relation = "Above 0.35 boundary" if probability >= decision_threshold else "Below 0.35 boundary"
        st.caption(f"{threshold_relation} | Boundary distance: {stability_percent:.2f} pp")
        if next_actions:
            st.info(f"Next action: {next_actions[0]}")

        snapshot = normalize_result_snapshot(
            result,
            fallback_record_name=current_view if current_view != "Batch Summary" else "Current Case",
        )
        export_record_name = snapshot.get("record_name", "Current Case")
        export_patient_id = snapshot.get("patient_id", "") or (patient_id or "")
        export_scope = snapshot.get("record_uid", "") or export_record_name
        export_scope = re.sub(r"[^0-9A-Za-z_-]", "_", str(export_scope))

        st.markdown("### Report Export")
        export_btn_col, export_dl_col = st.columns([1, 2])
        export_html_state_key = f"clinical_export_html_{export_scope}"

        with export_btn_col:
            if st.button("Prepare HTML Clinical Report", key=f"prepare_single_html_{export_scope}", use_container_width=True):
                html_report = build_single_case_html_report(
                    result=result,
                    record_name=export_record_name,
                    patient_id=export_patient_id,
                    generated_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                )
                st.session_state[export_html_state_key] = html_report.encode("utf-8")

        with export_dl_col:
            html_bytes = st.session_state.get(export_html_state_key)
            if isinstance(html_bytes, (bytes, bytearray)):
                safe_file_stem = re.sub(r"[^0-9A-Za-z_-]", "_", export_record_name).strip("_") or "clinical_report"
                st.download_button(
                    "Download HTML Clinical Report",
                    data=html_bytes,
                    file_name=f"{safe_file_stem}_clinical_report.html",
                    mime="text/html",
                    key=f"download_single_html_{export_scope}",
                    use_container_width=True,
                )
            else:
                st.caption("Prepare the report once, then download as printable HTML.")

        st.markdown("### Similar Local Cases")
        if st.session_state.record_store_ready:
            try:
                candidate_records = [
                    item
                    for item in list_records(status="all", limit=1000)
                    if isinstance(item, dict) and str(item.get("status", "")).lower() != "deleted"
                ]
            except Exception as retrieval_exc:  # noqa: BLE001
                candidate_records = []
                st.warning(f"Unable to query local case library: {retrieval_exc}")

            similar_cases = find_similar_cases(result, candidate_records, top_k=5)
            if similar_cases:
                similar_df = pd.DataFrame(similar_cases)
                display_df = similar_df.copy()
                display_df["risk_score_%"] = (display_df["probability_display"] * 100.0).round(2)
                display_df["similarity_%"] = display_df["similarity_score"].round(2)
                # Format recommendation_tier by removing underscores
                display_df["recommendation_tier"] = display_df["recommendation_tier"].apply(format_recommendation_tier)
                show_cols = [
                    "record_name",
                    "patient_id",
                    "similarity_%",
                    "risk_score_%",
                    "probability_delta",
                    "decision_stability_display",
                    "recommendation_tier",
                    "evidence_summary",
                    "created_at",
                ]
                st.dataframe(display_df[show_cols], use_container_width=True, hide_index=True)

                load_options = {
                    (
                        f"{row['record_name']} | similarity {row['similarity_score']:.1f}% | "
                        f"risk {row['probability_display'] * 100.0:.1f}%"
                    ): row["record_uid"]
                    for row in similar_cases
                    if row.get("record_uid")
                }
                if load_options:
                    option_labels = list(load_options.keys())
                    selected_label = st.selectbox(
                        "Load similar case",
                        options=option_labels,
                        key=f"similar_case_selector_{export_scope}",
                    )
                    if st.button("Load selected similar case", key=f"load_similar_case_{export_scope}"):
                        target_uid = load_options.get(selected_label, "")
                        payload = get_record_payload(target_uid)
                        if not isinstance(payload, dict):
                            st.error("Unable to load selected similar case payload from local storage.")
                        else:
                            st.session_state.records_loaded_result = payload
                            st.session_state.last_ml_result = payload
                            if "batch_results" in st.session_state:
                                del st.session_state.batch_results
                            st.session_state.current_view = "Batch Summary"
                            st.session_state.persistence_notice = "Loaded selected similar case into Clinical Report."
                            if st.session_state.chatbot_ready:
                                st.session_state.chatbot.reset_conversation()
                                st.session_state.conversation_history = []
                            st.rerun()
            else:
                st.info("No similar local cases found yet. Run or load more records into Records Center to activate retrieval.")
        else:
            st.info("Local record store is unavailable, so similar-case retrieval is currently disabled.")

        st.markdown("### Key Evidence")
        v1, v2 = st.columns(2)

        with v1:
            _render_decision_margin_legend(probability=probability, threshold=decision_threshold)
            margin_fig = _plot_decision_margin(
                probability=probability,
                threshold=decision_threshold,
                gray_zone_upper=gray_zone_upper,
            )
            st.pyplot(margin_fig, clear_figure=True)

        with v2:
            if clinical_evidence:
                evidence_df = pd.DataFrame(clinical_evidence)
                heatmap_fig = _plot_evidence_heatmap(evidence_df)
                st.pyplot(heatmap_fig, clear_figure=True)
            else:
                st.info("No extractable V1-V3 morphology evidence for this record.")

        with st.expander("View 12-lead ECG", expanded=False):
            # Optimize ECG rendering with session-level caching
            import hashlib
            signal_array = signal_plot
            if signal_array is not None:
                try:
                    signal_bytes = np.array(signal_array).tobytes()
                    signal_hash = hashlib.md5(signal_bytes).hexdigest()[:8]
                    cache_key = f"ecg_plot_{signal_hash}_{fs_plot}"
                    
                    if cache_key not in st.session_state:
                        # Generate and cache the plot
                        plot_highlights = highlights if isinstance(highlights, dict) else {}
                        ecg_fig = _plot_12_lead(
                            signal=signal_plot,
                            lead_names=lead_names,
                            fs=fs_plot,
                            highlights=plot_highlights,
                        )
                        st.session_state[cache_key] = ecg_fig
                    
                    ecg_fig = st.session_state[cache_key]
                    st.pyplot(ecg_fig, clear_figure=True)
                except Exception as ecg_err:  # noqa: BLE001
                    st.warning(f"ECG rendering optimized (cache error: {str(ecg_err)[:50]})")
                    plot_highlights = highlights if isinstance(highlights, dict) else {}
                    ecg_fig = _plot_12_lead(
                        signal=signal_plot,
                        lead_names=lead_names,
                        fs=fs_plot,
                        highlights=plot_highlights,
                    )
                    st.pyplot(ecg_fig, clear_figure=True)
            else:
                st.warning("ECG signal data unavailable")
            
            st.info("For stable magnified diagnosis, use the dedicated ECG Review tab for per-lead inspection.")

        with st.expander("Detailed Clinical Explanation & Evidence", expanded=False):
            st.write(explanation or "No explanation text returned by model.")
            st.caption(f"Risk score: {probability:.4f} | Decision boundary: {decision_threshold:.4f}")

            if gray_zone or stability_percent <= 1.0:
                st.markdown(
                    f"<div style='margin-top: 1rem; margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fef3c7; color: #92400e; display: flex; align-items: center;'>{SVG_WARNING} <strong>Borderline Interpretation Protocol</strong></div>",
                    unsafe_allow_html=True,
                )
                st.write("- Repeat ECG acquisition and verify V1-V3 lead placement.")
                st.write("- Prioritize manual cardiologist over-read for morphology confirmation.")
                st.write("- Correlate with symptoms, syncope history, and family history.")
                st.write("- If uncertainty remains, escalate to urgent specialist review pathway.")

            if clinical_evidence:
                preferred_cols = [
                    "lead",
                    "source",
                    "tier",
                    "reliability",
                    "j_height",
                    "st_slope",
                    "curvature",
                    "score",
                    "segments",
                ]
                show_cols = [c for c in preferred_cols if c in evidence_df.columns]
                if show_cols:
                    display_df = evidence_df[show_cols].rename(
                        columns={
                            "tier": "evidence_strength",
                            "reliability": "extraction_reliability",
                        }
                    )
                    st.dataframe(display_df, use_container_width=True)

            if model_contributions:
                contrib_df = (
                    pd.DataFrame(
                        [{"view": k, "contribution_percent": v} for k, v in model_contributions.items()]
                    )
                    .sort_values("contribution_percent", ascending=False)
                    .reset_index(drop=True)
                )
                st.subheader("Model Contribution Summary")
                st.dataframe(contrib_df, use_container_width=True)

            st.subheader("Recommended Clinical Actions")
            st.caption(
                f"Tier: {rec_tier} | Evidence S/M/W: "
                f"{_safe_int(evidence_counts.get('strong', 0))}/"
                f"{_safe_int(evidence_counts.get('moderate', 0))}/"
                f"{_safe_int(evidence_counts.get('weak', 0))}"
            )
            for action in next_actions:
                if isinstance(action, str) and action.strip():
                    st.write(f"- {action}")

            evidence_segments = sum(_safe_int(item.get("segments", 0)) for item in clinical_evidence)
            if operational_probability >= operational_threshold and evidence_segments == 0:
                st.markdown(
                    f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fef3c7; color: #92400e; display: flex; align-items: center;'>{SVG_WARNING} High-risk prediction with no V1-V3 morphological evidence. Manual cardiology review is recommended.</div>",
                    unsafe_allow_html=True,
                )

    if "batch_results" in st.session_state and current_view == "Batch Summary" and loaded_record_result is None:
        batch_results = st.session_state.batch_results
        if not batch_results:
            st.markdown(
                f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fef3c7; color: #92400e; display: flex; align-items: center;'>{SVG_WARNING} No valid WFDB pairs found in batch uploads.</div>",
                unsafe_allow_html=True,
            )
        else:
            df = pd.DataFrame(batch_results)
            batch_export_state_key = "clinical_export_batch_zip"
            b1, b2 = st.columns([1, 2])
            with b1:
                if st.button("Prepare Batch HTML ZIP", key="prepare_batch_html_zip", use_container_width=True):
                    batch_zip = build_batch_html_zip(batch_results, batch_name="batch_clinical_reports")
                    st.session_state[batch_export_state_key] = batch_zip
            with b2:
                batch_zip_bytes = st.session_state.get(batch_export_state_key)
                if isinstance(batch_zip_bytes, (bytes, bytearray)):
                    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
                    st.download_button(
                        "Download Batch HTML Reports (ZIP)",
                        data=batch_zip_bytes,
                        file_name=f"brugada_batch_reports_{ts}.zip",
                        mime="application/zip",
                        key="download_batch_html_zip",
                        use_container_width=True,
                    )
                else:
                    st.caption("Prepare batch package once, then download all case reports in a ZIP archive.")

            sort_prob_col = "probability_raw" if "probability_raw" in df.columns else "probability"
            sort_stability_col = "decision_stability_raw" if "decision_stability_raw" in df.columns else "decision_stability"
            if "recommendation_tier" in df.columns:
                df["_tier_rank"] = df["recommendation_tier"].map(_tier_sort_value)
                df = df.sort_values(["_tier_rank", sort_prob_col, sort_stability_col], ascending=[True, False, False]).drop(columns=["_tier_rank"])
            else:
                df = df.sort_values(sort_prob_col, ascending=False)

            hidden_cols = [
                c for c in ["raw", "probability_raw", "decision_stability_raw", "decision_threshold_raw"] if c in df.columns
            ]
            display_df = df.drop(columns=hidden_cols) if hidden_cols else df.copy()
            # Format recommendation_tier to remove underscores
            if "recommendation_tier" in display_df.columns:
                display_df["recommendation_tier"] = display_df["recommendation_tier"].apply(format_recommendation_tier)
            st.dataframe(display_df, use_container_width=True)

            st.subheader("Urgent Review Queue")
            urgent = (
                df[df["recommendation_tier"].isin(["urgent_cardiology_review", "urgent_review_repeat_ecg_quality_check"])].sort_values(sort_prob_col, ascending=False)
                if "recommendation_tier" in df.columns
                else df.iloc[0:0]
            )
            if urgent.empty:
                st.markdown(
                    f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #e0f2fe; color: #075985; display: flex; align-items: center;'>{SVG_INFO} No urgent-cardiology records in this batch.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.dataframe(urgent[["record", "probability", "decision_stability", "label", "evidence_strength_summary"]], use_container_width=True)

            st.subheader("Gray-Zone Priority Queue")
            gray_queue = df[df["gray_zone"] == True] if "gray_zone" in df.columns else df.iloc[0:0]
            if gray_queue.empty:
                st.markdown(
                    f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #e0f2fe; color: #075985; display: flex; align-items: center;'>{SVG_INFO} No gray-zone records detected in this batch.</div>",
                    unsafe_allow_html=True,
                )
            else:
                display_gray_queue = gray_queue[["record", "probability", "decision_stability", "label", "recommendation_tier"]].copy()
                display_gray_queue["recommendation_tier"] = display_gray_queue["recommendation_tier"].apply(format_recommendation_tier)
                st.dataframe(display_gray_queue, use_container_width=True)

    return single_result_to_show, current_view