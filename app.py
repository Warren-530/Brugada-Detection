import streamlit as st
import pandas as pd
import html
import re
from chatbot import BrugadaChatbot
from inference import predict_from_record
from record_store import (
    get_record_counts,
    get_record_payload,
    init_record_store,
    list_records,
    save_batch_results,
    save_record_result,
    update_record_status,
)

from file_utils import (
    _save_uploaded_pair,
    _save_batch_folder,
    _predict_batch_from_folder,
    _normalize_clinician_explain,
    _safe_int,
    _tier_sort_value,
    group_uploaded_files
)

from ui_components import (
    DEFAULT_LEAD_NAMES,
    SVG_SUCCESS,
    SVG_WARNING,
    SVG_ERROR,
    SVG_INFO,
    SVG_FOLDER,
    get_status_indicator_svg,
    inject_custom_css,
    _plot_12_lead,
    _render_decision_margin_legend,
    _plot_decision_margin,
    _plot_evidence_heatmap,
    _recommendation_banner
)

st.set_page_config(page_title="Brugada AI Assistant", page_icon="ECG", layout="wide")
inject_custom_css()
st.title("Brugada Syndrome Clinical AI Assistant")
st.caption("Multi-view deep feature stacking + meta-learner for single-patient ECG triage")

if "record_store_ready" not in st.session_state:
    try:
        init_record_store()
        st.session_state.record_store_ready = True
        st.session_state.record_store_error = ""
    except Exception as exc:  # noqa: BLE001
        st.session_state.record_store_ready = False
        st.session_state.record_store_error = str(exc)

# =============================================================================
# Chatbot & Session State Initialization
# =============================================================================
if "chatbot" not in st.session_state:
    try:
        st.session_state.chatbot = BrugadaChatbot()
        st.session_state.chatbot_ready = True
    except Exception as e:
        st.session_state.chatbot_ready = False
        st.session_state.chatbot_error = str(e)

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


def clear_uploads():
    st.session_state.uploader_key += 1
    st.session_state.deleted_pairs = set()
    if "last_ml_result" in st.session_state:
        st.session_state.last_ml_result = None
    if "batch_results" in st.session_state:
        del st.session_state.batch_results


def _render_metric_with_info(column, label: str, value: str, info_markdown: str, info_key: str):
    """Render metric as a custom card with an icon-only tooltip."""
    info_lines = []
    for raw_line in info_markdown.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"\*\*|`", "", line)
        if line.startswith("- "):
            line = f"• {line[2:].strip()}"
        info_lines.append(html.escape(line))

    info_html = "<br>".join(info_lines) if info_lines else html.escape("No additional details.")

    with column:
        st.markdown(
            f"""
            <div class="metric-tile" id="{info_key}">
                <div class="metric-tile-header">
                    <span class="metric-tile-label">{html.escape(label)}</span>
                    <details class="metric-info-details">
                        <summary>
                            <span class="metric-info-icon" title="More info">ℹ</span>
                        </summary>
                        <div class="metric-info-panel">
                            <div class="metric-info-panel-title">MORE INFO</div>
                            {info_html}
                        </div>
                    </details>
                </div>
                <div class="metric-tile-value">{html.escape(value)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

left, right = st.columns([1, 2])

with left:
    st.subheader("Patient Input")
    
    # Retrieve current files to determine expander state
    current_files = st.session_state.get(f"unified_upload_{st.session_state.uploader_key}")
    
    # Keep expander open if no files, or if there are incomplete pairs
    is_expanded = not bool(current_files)
    if current_files:
        _, mp = group_uploaded_files(current_files)
        # Check if there are missing pairs that haven't been deleted
        active_mp = [k for k in mp.keys() if k not in st.session_state.deleted_pairs]
        if active_mp:
            is_expanded = True
    
    with st.expander("Upload Records", expanded=is_expanded):
        st.markdown(f"<div style='margin-bottom: 0.5rem;'>Drop the entire window here or browse.</div>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload .hea and .dat files (Single pair or batch)",
            type=["hea", "dat"],
            accept_multiple_files=True,
            key=f"unified_upload_{st.session_state.uploader_key}",
            label_visibility="collapsed"
        )
        
        if st.button("Clear Uploads", on_click=clear_uploads, use_container_width=True):
            pass

    with st.expander("Optional Metadata", expanded=False):
        st.text_input(
            "Patient ID",
            key="patient_id_input",
            placeholder="e.g., PT-001",
            help="Optional identifier stored with saved records. Leave empty if unavailable.",
        )

    patient_id = st.session_state.get("patient_id_input", "").strip() or None

    st.write("") # Spacing
    run_btn = st.button("Run Diagnosis", type="primary", use_container_width=True)
    st.write("")
    
    pairs, missing_pairs = {}, {}
    if uploaded_files:
        pairs, missing_pairs = group_uploaded_files(uploaded_files)
        
        # Filter deleted pairs visually from the side panel
        pairs = {k: v for k, v in pairs.items() if k not in st.session_state.deleted_pairs}
        missing_pairs = {k: v for k, v in missing_pairs.items() if k not in st.session_state.deleted_pairs}
        
        is_batch = len(pairs) > 1
        
        # Display configured pairs as visual blocks
        if pairs or missing_pairs:
            st.markdown("##### Uploaded Record Pairs")
            
            if "batch_results" in st.session_state and pairs:
                if st.button("Batch Summary", key="nav_batch_summary", use_container_width=True):
                    st.session_state.current_view = "Batch Summary"
                    
                st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
                    
                for res in st.session_state.batch_results:
                    stem_name = res['record']
                    # Skip deleted items from showing up in batch nav if user deleted them while viewing results
                    if stem_name in st.session_state.deleted_pairs:
                        continue
                        
                    is_detected = res.get("label") == "Brugada Syndrome Detected"
                    is_urgent = res.get("recommendation_tier") in {"urgent_cardiology_review", "urgent_review_repeat_ecg_quality_check"}
                    is_gray = res.get("gray_zone", False)
                    indicator_svg = get_status_indicator_svg(is_detected, is_urgent, is_gray)
                    
                    if is_detected:
                        status_msg = "Brugada Syndrome Detected"
                        if is_urgent:
                            status_msg += " (Urgent)"
                        elif is_gray:
                            status_msg += " (Gray-zone)"
                        card_svg = f"<span title='{status_msg}' style='cursor: help;'>{SVG_WARNING}</span>"
                    else:
                        card_svg = f"<span title='No Brugada Syndrome Detected'>{SVG_FOLDER}</span>"
                    
                    col1, col2 = st.columns([6, 1])
                    with col1:
                        st.markdown(
                            f"<div class='record-card'>"
                            f"<div style='display: flex; align-items: center;'>{card_svg} <span style='font-weight: 600;'>{stem_name}</span></div>"
                            f"<div><span class='record-tag'>.hea</span><span class='record-tag'>.dat</span></div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    with col2:
                        clicked = st.button("→", key=f"nav_{stem_name}", use_container_width=True)
                        if clicked:
                            st.session_state.current_view = stem_name
            else:
                for stem in list(pairs.keys()):
                    card_svg = f"<span title='Valid pair ready for diagnosis' style='cursor: help;'>{SVG_FOLDER}</span>"
                    
                    if len(pairs) == 1 and st.session_state.get('last_ml_result') is not None:
                        result = st.session_state.last_ml_result
                        if isinstance(result, dict):
                            is_detected = result.get("label", "") == "Brugada Syndrome Detected"
                            is_urgent = result.get("recommendation_tier") in {"urgent_cardiology_review", "urgent_review_repeat_ecg_quality_check"}
                            is_gray = result.get("gray_zone", False)
                        else:
                            is_detected = getattr(result, "label", "") == "Brugada Syndrome Detected"
                            is_urgent = getattr(result, "recommendation_tier", "") in {"urgent_cardiology_review", "urgent_review_repeat_ecg_quality_check"}
                            is_gray = getattr(result, "gray_zone", False)
                        
                        if is_detected:
                            status_msg = "Brugada Syndrome Detected"
                            if is_urgent:
                                status_msg += " (Urgent)"
                            elif is_gray:
                                status_msg += " (Gray-zone)"
                            card_svg = f"<span title='{status_msg}' style='cursor: help;'>{SVG_WARNING}</span>"
                        else:
                            card_svg = f"<span title='No Brugada Syndrome Detected'>{SVG_FOLDER}</span>"
                            
                    col1, col2 = st.columns([6, 1])
                    with col1:
                        st.markdown(
                            f"<div class='record-card'>"
                            f"<div style='display: flex; align-items: center;'>{card_svg} <span style='font-weight: 600;'>{stem}</span></div>"
                            f"<div><span class='record-tag'>.hea</span><span class='record-tag'>.dat</span></div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    with col2:
                        if st.button("✕", key=f"del_{stem}", use_container_width=True):
                            st.session_state.deleted_pairs.add(stem)
                            st.rerun()

                for stem, missing in missing_pairs.items():
                    col1, col2 = st.columns([6, 1])
                    with col1:
                        tags = []
                        if "hea" in missing:
                            tags.append("<span class='record-tag-missing'>.hea</span>")
                        else:
                            tags.append("<span class='record-tag'>.hea</span>")
                            
                        if "dat" in missing:
                            tags.append("<span class='record-tag-missing'>.dat</span>")
                        else:
                            tags.append("<span class='record-tag'>.dat</span>")
                            
                        missing_text = " and ".join([f".{m}" for m in missing])
                        st.markdown(
                            f"<div class='record-card-warning'>"
                            f"<div style='display: flex; align-items: center;'><span title='Missing {missing_text} file(s)' style='cursor: help;'>{SVG_WARNING}</span> <span style='font-weight: 600;'>{stem}</span></div>"
                            f"<div>{''.join(tags)}</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    with col2:
                        if st.button("✕", key=f"del_mp_{stem}", use_container_width=True):
                            st.session_state.deleted_pairs.add(stem)
                            st.rerun()
            
            if st.button("Clear Uploads", key="clear_uploaded_pairs", on_click=clear_uploads, use_container_width=True):
                pass
            st.write("")

with right:
    tab_report, tab_chatbot, tab_records = st.tabs(["Clinical Report", "AI Advisor", "Records Center"])
    
    with tab_report:
        st.subheader("Clinical Report")

        if st.session_state.persistence_notice:
            st.info(st.session_state.persistence_notice)
            st.session_state.persistence_notice = ""

        # Determine mode
        is_batch = len(pairs) > 1
        is_single = len(pairs) == 1
        
        if run_btn:
            if not pairs:
                st.markdown(f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fee2e2; color: #991b1b; display: flex; align-items: center;'>{SVG_ERROR} Please upload at least one valid .hea and .dat pair before running diagnosis.</div>", unsafe_allow_html=True)
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

                        # Reset chat for a new record
                        if st.session_state.chatbot_ready:
                            st.session_state.chatbot.reset_conversation()
                            st.session_state.conversation_history = []
                        st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.exception(exc)
            else: # is_batch
                with st.spinner("Scoring all uploaded records..."):
                    batch_dir = _save_batch_folder(uploaded_files)
                    batch_results = _predict_batch_from_folder(batch_dir)
                    st.session_state.records_loaded_result = None
                    st.session_state.batch_results = batch_results

                    if st.session_state.record_store_ready:
                        try:
                            uid_map = save_batch_results(batch_results, patient_id=patient_id)
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

        single_result_to_show = None
        has_results = False

        loaded_record_result = st.session_state.get("records_loaded_result")
        if loaded_record_result is not None:
            single_result_to_show = loaded_record_result
            has_results = True
        
        elif not is_batch and st.session_state.last_ml_result is not None:
            single_result_to_show = st.session_state.last_ml_result
            has_results = True
        elif is_batch and "batch_results" in st.session_state:
            has_results = True
            if current_view != "Batch Summary":
                for res in st.session_state.batch_results:
                    if res["record"] == current_view:
                        single_result_to_show = res.get("raw")
                        break

        # --- Empty State Message ---
        if not has_results:
            st.markdown(f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #e0f2fe; color: #075985; display: flex; align-items: center;'><span style='margin-right: 0.5rem; display: flex;'>{SVG_INFO}</span> Run diagnosis to see results in the clinical report tab.</div>", unsafe_allow_html=True)

        # --- Single Record Display Logic ---
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
            rec_text = str(clinician_explain.get("recommendation_text", "Clinical correlation is recommended."))
            evidence_counts = clinician_explain.get("evidence_counts", {})
            next_actions = clinician_explain.get("next_actions", []) if isinstance(clinician_explain.get("next_actions", []), list) else []
            mismatch = bool(clinician_explain.get("morphology_model_mismatch", False))

            if label == "Brugada Syndrome Detected":
                st.markdown(f"<div style='margin-bottom: 1rem; padding: 1rem; border-radius: 0.5rem; background-color: #fee2e2; color: #991b1b; display: flex; align-items: center; font-weight: bold; font-size: 1.1em;'>{SVG_ERROR} {label}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='margin-bottom: 1rem; padding: 1rem; border-radius: 0.5rem; background-color: #dcfce7; color: #166534; display: flex; align-items: center; font-weight: bold; font-size: 1.1em;'>{SVG_SUCCESS} {label}</div>", unsafe_allow_html=True)

            _recommendation_banner(rec_tier, rec_text)
            
            if gray_zone:
                st.markdown(f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fef3c7; color: #92400e; display: flex; align-items: center;'>{SVG_WARNING} Gray-zone prediction: score is close to the 0.35 boundary and requires cardiology review.</div>", unsafe_allow_html=True)
            if mismatch:
                st.markdown(f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fef3c7; color: #92400e; display: flex; align-items: center;'>{SVG_WARNING} Discordant case: model decision and V1-V3 morphology strength are not strongly aligned. Prioritize manual review.</div>", unsafe_allow_html=True)

            # L1: concise decision snapshot
            risk_pct = float(probability * 100.0)
            st.markdown("### Diagnostic Snapshot")
            m1, m2, m3 = st.columns(3)
            _render_metric_with_info(
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
            _render_metric_with_info(
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
            _render_metric_with_info(
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
            
            # L2: compact evidence visuals
            st.markdown("### Key Evidence")
            v1, v2 = st.columns(2)
            
            with v1:
                _render_decision_margin_legend(
                    probability=probability,
                    threshold=decision_threshold,
                )
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
                plot_highlights = highlights if (label == "Brugada Syndrome Detected" or gray_zone) else {}
                ecg_fig = _plot_12_lead(
                    signal=signal_plot,
                    lead_names=lead_names,
                    fs=fs_plot,
                    highlights=plot_highlights,
                )
                st.pyplot(ecg_fig, clear_figure=True)

            # L3: detailed evidence and full explanation
            with st.expander("Detailed Clinical Explanation & Evidence", expanded=False):
                st.write(explanation or "No explanation text returned by model.")
                st.caption(f"Risk score: {probability:.4f} | Decision boundary: {decision_threshold:.4f}")
                
                if gray_zone or stability_percent <= 1.0:
                    st.markdown(f"<div style='margin-top: 1rem; margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fef3c7; color: #92400e; display: flex; align-items: center;'>{SVG_WARNING} <strong>Borderline Interpretation Protocol</strong></div>", unsafe_allow_html=True)
                    st.write("- Repeat ECG acquisition and verify V1-V3 lead placement.")
                    st.write("- Prioritize manual cardiologist over-read for morphology confirmation.")
                    st.write("- Correlate with symptoms, syncope history, and family history.")
                    st.write("- If uncertainty remains, escalate to urgent specialist review pathway.")
                
                if clinical_evidence:
                    preferred_cols = ["lead", "source", "tier", "reliability", "j_height", "st_slope", "curvature", "score", "segments"]
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
                    st.markdown(f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fef3c7; color: #92400e; display: flex; align-items: center;'>{SVG_WARNING} High-risk prediction with no V1-V3 morphological evidence. Manual cardiology review is recommended.</div>", unsafe_allow_html=True)

        # --- Batch Record Display Logic ---
        if 'batch_results' in st.session_state and is_batch and current_view == "Batch Summary" and loaded_record_result is None:
            batch_results = st.session_state.batch_results
            if not batch_results:
                st.markdown(f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fef3c7; color: #92400e; display: flex; align-items: center;'>{SVG_WARNING} No valid WFDB pairs found in batch uploads.</div>", unsafe_allow_html=True)
            else:
                df = pd.DataFrame(batch_results)
                sort_prob_col = "probability_raw" if "probability_raw" in df.columns else "probability"
                sort_stability_col = "decision_stability_raw" if "decision_stability_raw" in df.columns else "decision_stability"
                if "recommendation_tier" in df.columns:
                    df["_tier_rank"] = df["recommendation_tier"].map(_tier_sort_value)
                    df = df.sort_values(["_tier_rank", sort_prob_col, sort_stability_col], ascending=[True, False, False]).drop(columns=["_tier_rank"])
                else:
                    df = df.sort_values(sort_prob_col, ascending=False)
                # Drop raw before showing dataframe
                hidden_cols = [
                    c
                    for c in ["raw", "probability_raw", "decision_stability_raw", "decision_threshold_raw"]
                    if c in df.columns
                ]
                display_df = df.drop(columns=hidden_cols) if hidden_cols else df
                st.dataframe(display_df, use_container_width=True)

                st.subheader("Urgent Review Queue")
                urgent = (
                    df[df["recommendation_tier"].isin(["urgent_cardiology_review", "urgent_review_repeat_ecg_quality_check"])].sort_values(sort_prob_col, ascending=False)
                    if "recommendation_tier" in df.columns
                    else df.iloc[0:0]
                )
                if urgent.empty:
                    st.markdown(f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #e0f2fe; color: #075985; display: flex; align-items: center;'>{SVG_INFO} No urgent-cardiology records in this batch.</div>", unsafe_allow_html=True)
                else:
                    st.dataframe(urgent[["record", "probability", "decision_stability", "label", "evidence_strength_summary"]], use_container_width=True)

                st.subheader("Gray-Zone Priority Queue")
                gray_queue = df[df["gray_zone"] == True] if "gray_zone" in df.columns else df.iloc[0:0]
                if gray_queue.empty:
                    st.markdown(f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #e0f2fe; color: #075985; display: flex; align-items: center;'>{SVG_INFO} No gray-zone records detected in this batch.</div>", unsafe_allow_html=True)
                else:
                    st.dataframe(gray_queue[["record", "probability", "decision_stability", "label", "recommendation_tier"]], use_container_width=True)

    with tab_chatbot:
        st.subheader("AI Clinical Advisor")
        st.caption("Ask questions about the diagnosis and get evidence-based clinical guidance")

        advisor_target_result = single_result_to_show if single_result_to_show is not None else st.session_state.last_ml_result

        if is_batch and current_view == "Batch Summary" and advisor_target_result is None:
            st.markdown(f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #e0f2fe; color: #075985; display: flex; align-items: center;'>{SVG_INFO} The AI Advisor requires a specific patient record. Please select a record from the 'Uploaded Record Pairs' sidebar to view its clinical advice and ask questions.</div>", unsafe_allow_html=True)
        elif st.session_state.chatbot_ready:
            st.markdown(f"""
            <div style='margin-bottom: 1rem; padding: 1rem; border-radius: 0.5rem; background-color: #fef2f2; color: #991b1b; border: 1px solid #fecaca;'>
                <div style='display: flex; align-items: center; font-weight: bold; margin-bottom: 0.5rem;'>
                    {SVG_WARNING} <span style="margin-left: 0.3rem;">Important Disclaimer</span>
                </div>
                This AI-generated interpretation is a decision support tool for qualified physicians only and does not constitute a diagnostic, prognostic, or treatment recommendation. Clinical decisions must always be made by a physician based on comprehensive patient evaluation, clinical judgment, and current medical guidelines. The ultimate responsibility for patient care rests with the treating physician.
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Initial AI Advice", expanded=True):
                if advisor_target_result is None:
                    st.info("Run a diagnosis or load a saved record to generate AI advice.")
                else:
                    try:
                        with st.spinner("Analyzing clinical data and generating advice..."):
                            initial_advice = st.session_state.chatbot.get_advice(advisor_target_result)
                    
                        import re
                        # Parse sections strictly by ### 
                        sections = re.split(r'\n(?=### )', "\n" + initial_advice)
                        
                        for section in sections:
                            section = section.strip()
                            if not section: continue
                            
                            lower_section = section.lower()
                            if "consideration" in lower_section or "differential" in lower_section:
                                bg_col, border_col, svg_icon = "#fffbeb", "#fef08a", SVG_WARNING
                            elif "step" in lower_section or "action" in lower_section or "recommend" in lower_section:
                                bg_col, border_col, svg_icon = "#f0fdf4", "#bbf7d0", SVG_SUCCESS
                            else:
                                bg_col, border_col, svg_icon = "#f0f9ff", "#bae6fd", SVG_INFO

                            # Separate heading from body to inject custom styling for heading
                            lines = section.split('\n', 1)
                            heading = lines[0].replace('###', '').strip()
                            body = lines[1] if len(lines) > 1 else ""

                            st.markdown(f'''
<style>
.ai-advice-container p, .ai-advice-container li, .ai-advice-container span, .ai-advice-container strong {{
    color: #1e293b !important;
}}
</style>
<div class="ai-advice-container" style="background-color: {bg_col}; border: 1px solid {border_col}; border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem; color: #1e293b;">
<div style="display: flex; align-items: center; font-weight: bold; font-size: 1.1em; margin-bottom: 0.8rem; color: #1e293b;">
    {svg_icon} <span style="margin-left: 0.3rem;">{heading}</span>
</div>

{body}

</div>
''', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fee2e2; color: #991b1b; display: flex; align-items: center;'>{SVG_ERROR} Error generating advice: {str(e)}</div>", unsafe_allow_html=True)

            def send_message():
                user_q = st.session_state.user_question_input.strip()
                if user_q:
                    # Append user question immediately with None response to trigger loading state
                    st.session_state.conversation_history.append({
                        "user_q": user_q,
                        "response": None
                    })
                    # Clear using a clear_trigger key to force a re-render without hitting the widget exception
                    st.session_state.clear_input_trigger = True
            
            def reset_chat():
                st.session_state.chatbot.reset_conversation()
                st.session_state.conversation_history = []

            st.write("**Chat History:**")
            chat_container = st.container(border=True, height=400)
            
            with chat_container:
                if st.session_state.conversation_history:
                    for exchange in st.session_state.conversation_history:
                        with st.chat_message("user"):
                            st.markdown(exchange["user_q"])
                        
                        if exchange['response'] is None:
                            with st.spinner("Generating response..."):
                                try:
                                    response = st.session_state.chatbot.continue_conversation(exchange['user_q'])
                                    exchange['response'] = response
                                    st.rerun() # Refresh to show the new response
                                except Exception as e:
                                    exchange['response'] = f"**Error:** {str(e)}"
                                    st.rerun()
                        else:
                            with st.chat_message("assistant"):
                                st.markdown(exchange['response'])
                else:
                    st.caption("No questions yet. Ask something below!")

            st.write("**Ask follow-up questions:**")
            
            # Using a form ensures that hitting "Enter" in the text_input triggers the submit button
            
            # Clear logic via key manipulation to bypass Streamlit form bugs
            if st.session_state.get('clear_input_trigger', False):
                st.session_state.user_question_input = ""
                st.session_state.clear_input_trigger = False
                
            with st.form(key="chat_form", clear_on_submit=True):
                col_input, col_send = st.columns([4, 1])
                
                with col_input:
                    st.text_input(
                        "Your question:",
                        key="user_question_input",
                        placeholder="e.g., What should I look for in Lead V2?",
                        label_visibility="collapsed"
                    )
                
                with col_send:
                    submitted = st.form_submit_button("Send ↩︎", use_container_width=True, type="primary")
                    
                if submitted:
                    send_message()
                    st.rerun()
            
            st.button("Reset Chat", on_click=reset_chat, use_container_width=True)
            
        else:
            unavailable_msg = st.session_state.get('chatbot_error', 'Check API Key')
            error_html = f"""
            <div style='margin-bottom: 1rem; padding: 1rem; border-radius: 0.5rem; background-color: #fee2e2; color: #991b1b; display: flex; align-items: flex-start;'>
                <div style='margin-top: 2px; margin-right: 0.5rem;'>{SVG_ERROR}</div>
                <div>
                    <strong>AI Advisor Unavailable</strong><br>
                    An issue occurred connecting to the Gemini AI: {unavailable_msg}<br><br>
                    <strong>Troubleshooting steps:</strong><br>
                    1. <strong>Missing API Key:</strong> Ensure a valid <code>GEMINI_API_KEY</code> is set in your environment variables or in <code>.streamlit/secrets.toml</code>.<br>
                    2. <strong>Quota Reached:</strong> If you're encountering quota limits, please try again in a few minutes.<br>
                    3. <strong>Connectivity/Config:</strong> Verify your network connection and ensure you're using a valid model configuration.
                </div>
            </div>
            """
            st.markdown(error_html, unsafe_allow_html=True)

    with tab_records:
        st.subheader("Records Center")
        st.caption("Persistent local registry for record history, retrieval, and lifecycle management.")

        if not st.session_state.record_store_ready:
            store_error = st.session_state.get("record_store_error", "Unknown storage initialization error")
            st.markdown(
                f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fee2e2; color: #991b1b; display: flex; align-items: center;'>{SVG_ERROR} Local record store unavailable: {store_error}</div>",
                unsafe_allow_html=True,
            )
        else:
            try:
                counts = get_record_counts()
            except Exception as count_exc:  # noqa: BLE001
                st.error(f"Unable to read record counters: {count_exc}")
                counts = {"active": 0, "archived": 0, "deleted": 0}

            c1, c2, c3 = st.columns(3)
            c1.metric("Active", str(counts.get("active", 0)))
            c2.metric("Archived", str(counts.get("archived", 0)))
            c3.metric("Deleted", str(counts.get("deleted", 0)))

            status_display = st.radio(
                "Status Filter",
                ["Active", "Archived", "Deleted", "All"],
                horizontal=True,
                key="records_status_filter",
            )
            status_map = {
                "Active": "active",
                "Archived": "archived",
                "Deleted": "deleted",
                "All": "all",
            }
            selected_status = status_map[status_display]

            search_query = st.text_input(
                "Search Records",
                key="records_search_query",
                placeholder="Record name, patient ID, or label",
            )

            try:
                records = list_records(status=selected_status, search=search_query, limit=500)
            except Exception as list_exc:  # noqa: BLE001
                st.error(f"Unable to load records: {list_exc}")
                records = []

            if not records:
                st.markdown(
                    f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #e0f2fe; color: #075985; display: flex; align-items: center;'>{SVG_INFO} No records found for the selected filter.</div>",
                    unsafe_allow_html=True,
                )
            else:
                records_df = pd.DataFrame(records)
                records_df["risk_score_pct"] = (records_df["probability_display"] * 100.0).round(2)
                records_df["boundary_distance_pp"] = records_df["decision_stability_display"].round(2)

                display_columns = [
                    "created_at",
                    "record_name",
                    "patient_id",
                    "label",
                    "risk_score_pct",
                    "boundary_distance_pp",
                    "recommendation_tier",
                    "status",
                    "evidence_summary",
                ]
                st.dataframe(
                    records_df[display_columns].rename(
                        columns={
                            "created_at": "timestamp_utc",
                            "record_name": "record",
                            "patient_id": "patient_id",
                            "risk_score_pct": "risk_score_%",
                            "boundary_distance_pp": "decision_stability_pp",
                        }
                    ),
                    use_container_width=True,
                )

                option_map = {}
                for item in records:
                    record_line = (
                        f"{item['created_at']} | {item['record_name']} | "
                        f"{item['label']} | {item['probability_display'] * 100.0:.2f}%"
                    )
                    option_map[record_line] = item["record_uid"]

                selected_label = st.selectbox(
                    "Select Record",
                    options=list(option_map.keys()),
                    key="records_selected_label",
                )
                selected_uid = option_map[selected_label]
                selected_item = next((item for item in records if item["record_uid"] == selected_uid), None)

                if selected_item is not None:
                    a1, a2, a3 = st.columns(3)

                    with a1:
                        if st.button("Load Into Clinical Report", key=f"records_load_{selected_uid}", use_container_width=True):
                            payload = get_record_payload(selected_uid)
                            if payload is None:
                                st.error("Unable to load selected record payload from local storage.")
                            else:
                                st.session_state.records_loaded_result = payload
                                st.session_state.last_ml_result = payload
                                st.session_state.persistence_notice = (
                                    f"Loaded saved record '{selected_item['record_name']}' into Clinical Report."
                                )
                                if st.session_state.chatbot_ready:
                                    st.session_state.chatbot.reset_conversation()
                                    st.session_state.conversation_history = []
                                st.rerun()

                    with a2:
                        if selected_item["status"] == "active":
                            if st.button("Archive", key=f"records_archive_{selected_uid}", use_container_width=True):
                                if update_record_status(selected_uid, "archived"):
                                    st.session_state.persistence_notice = "Record archived."
                                st.rerun()
                        elif selected_item["status"] == "archived":
                            if st.button("Restore", key=f"records_restore_{selected_uid}", use_container_width=True):
                                if update_record_status(selected_uid, "active"):
                                    st.session_state.persistence_notice = "Record restored to active status."
                                st.rerun()
                        else:
                            st.caption("Archive action unavailable for deleted records.")

                    with a3:
                        if selected_item["status"] != "deleted":
                            if st.button("Soft Delete", key=f"records_delete_{selected_uid}", use_container_width=True):
                                if update_record_status(selected_uid, "deleted"):
                                    st.session_state.persistence_notice = "Record moved to deleted status."
                                st.rerun()
                        else:
                            if st.button("Recover", key=f"records_recover_{selected_uid}", use_container_width=True):
                                if update_record_status(selected_uid, "active"):
                                    st.session_state.persistence_notice = "Record recovered from deleted status."
                                st.rerun()

                    with st.expander("Selected Record Summary", expanded=False):
                        st.write(f"Record UID: {selected_item['record_uid']}")
                        st.write(f"Label: {selected_item['label']}")
                        st.write(f"Recommendation Tier: {selected_item['recommendation_tier']}")
                        st.write(f"Evidence Summary: {selected_item['evidence_summary']}")
                        st.write(f"Recommendation: {selected_item['recommendation_text']}")
