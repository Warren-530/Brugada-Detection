import streamlit as st
import pandas as pd
from chatbot import BrugadaChatbot
from inference import predict_from_record

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
    _plot_decision_margin,
    _plot_evidence_heatmap,
    _recommendation_banner
)

st.set_page_config(page_title="Brugada AI Assistant", page_icon="ECG", layout="wide")
inject_custom_css()
st.title("Brugada Syndrome Clinical AI Assistant")
st.caption("Multi-view deep feature stacking + meta-learner for single-patient ECG triage")

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

def clear_uploads():
    st.session_state.uploader_key += 1
    st.session_state.deleted_pairs = set()
    if "last_ml_result" in st.session_state:
        st.session_state.last_ml_result = None
    if "batch_results" in st.session_state:
        del st.session_state.batch_results

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
    tab_report, tab_chatbot = st.tabs(["Clinical Report", "AI Advisor"])
    
    with tab_report:
        st.subheader("Clinical Report")

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
                        st.session_state.last_ml_result = result
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
                    st.session_state.batch_results = batch_results
                    st.rerun()

        current_view = st.session_state.get("current_view", "Batch Summary")
        if current_view in st.session_state.deleted_pairs:
            current_view = "Batch Summary"
            st.session_state.current_view = current_view

        single_result_to_show = None
        has_results = False
        
        if not is_batch and st.session_state.last_ml_result is not None:
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
                st.markdown(f"<div style='margin-bottom: 1rem; padding: 1rem; border-radius: 0.5rem; background-color: #fee2e2; color: #991b1b; display: flex; align-items: center; font-weight: bold; font-size: 1.1em;'>{SVG_ERROR} {label}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='margin-bottom: 1rem; padding: 1rem; border-radius: 0.5rem; background-color: #dcfce7; color: #166534; display: flex; align-items: center; font-weight: bold; font-size: 1.1em;'>{SVG_SUCCESS} {label}</div>", unsafe_allow_html=True)

            _recommendation_banner(rec_tier, rec_text)
            
            if gray_zone:
                st.markdown(f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fef3c7; color: #92400e; display: flex; align-items: center;'>{SVG_WARNING} Gray-zone prediction: probability is close to the decision threshold and needs clinician review.</div>", unsafe_allow_html=True)
            if mismatch:
                st.markdown(f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fef3c7; color: #92400e; display: flex; align-items: center;'>{SVG_WARNING} Discordant case: model decision and V1-V3 morphology strength are not strongly aligned. Prioritize manual review.</div>", unsafe_allow_html=True)

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Decision Confidence", f"{confidence_percent:.1f}%")
            m2.metric("Threshold Distance", f"{stability_percent:.2f} pp")
            m3.metric("Predicted-Class Support", f"{class_support_percent:.1f}%")
            m4.metric("Brugada Risk Probability", f"{probability * 100.0:.2f}%")
            
            # Visuals
            st.markdown("### Visualizations")
            
            margin_fig = _plot_decision_margin(probability=probability, threshold=decision_threshold)
            st.pyplot(margin_fig, clear_figure=True)

            if clinical_evidence:
                evidence_df = pd.DataFrame(clinical_evidence)
                heatmap_fig = _plot_evidence_heatmap(evidence_df)
                st.pyplot(heatmap_fig, clear_figure=True)
            
            plot_highlights = highlights if (label == "Brugada Syndrome Detected" or gray_zone) else {}
            ecg_fig = _plot_12_lead(
                signal=signal_plot,
                lead_names=lead_names,
                fs=fs_plot,
                highlights=plot_highlights,
            )
            st.pyplot(ecg_fig, clear_figure=True)

            # Long Text Elements
            with st.expander("Detailed Clinical Explanation & Evidence", expanded=False):
                st.write(explanation or "No explanation text returned by model.")
                st.caption(f"Raw probability: {probability:.4f} | Threshold: {decision_threshold:.4f}")
                
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
                if probability >= decision_threshold and evidence_segments == 0:
                    st.markdown(f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fef3c7; color: #92400e; display: flex; align-items: center;'>{SVG_WARNING} High-risk prediction with no V1-V3 morphological evidence. Manual cardiology review is recommended.</div>", unsafe_allow_html=True)

        # --- Batch Record Display Logic ---
        if 'batch_results' in st.session_state and is_batch and current_view == "Batch Summary":
            batch_results = st.session_state.batch_results
            if not batch_results:
                st.markdown(f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fef3c7; color: #92400e; display: flex; align-items: center;'>{SVG_WARNING} No valid WFDB pairs found in batch uploads.</div>", unsafe_allow_html=True)
            else:
                df = pd.DataFrame(batch_results)
                if "recommendation_tier" in df.columns:
                    df["_tier_rank"] = df["recommendation_tier"].map(_tier_sort_value)
                    df = df.sort_values(["_tier_rank", "probability", "decision_stability"], ascending=[True, False, False]).drop(columns=["_tier_rank"])
                else:
                    df = df.sort_values("probability", ascending=False)
                # Drop raw before showing dataframe
                display_df = df.drop(columns=["raw"]) if "raw" in df.columns else df
                st.dataframe(display_df, use_container_width=True)

                st.subheader("Urgent Review Queue")
                urgent = df[df["recommendation_tier"].isin(["urgent_cardiology_review", "urgent_review_repeat_ecg_quality_check"])].sort_values("probability", ascending=False) if "recommendation_tier" in df.columns else df.iloc[0:0]
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

        if st.session_state.chatbot_ready:
            with st.expander("Initial AI Advice", expanded=True):
                try:
                    initial_advice = st.session_state.chatbot.get_advice(single_result_to_show if single_result_to_show else st.session_state.last_ml_result)
                    st.markdown(f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #e0f2fe; color: #075985; display: flex; align-items: flex-start;'><span style='margin-top: 2px;'>{SVG_INFO}</span> <div>{initial_advice}</div></div>", unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fee2e2; color: #991b1b; display: flex; align-items: center;'>{SVG_ERROR} Error generating advice: {str(e)}</div>", unsafe_allow_html=True)

            def send_message():
                user_q = st.session_state.user_question_input.strip()
                if user_q:
                    try:
                        response = st.session_state.chatbot.continue_conversation(user_q)
                        st.session_state.conversation_history.append({
                            "user_q": user_q,
                            "response": response
                        })
                    except Exception as e:
                        st.session_state.conversation_history.append({
                            "user_q": user_q,
                            "response": f"❌ Error: {str(e)}"
                        })
            
            def reset_chat():
                st.session_state.chatbot.reset_conversation()
                st.session_state.conversation_history = []

            st.write("**Chat History:**")
            chat_container = st.container(border=True, height=400)
            
            with chat_container:
                if st.session_state.conversation_history:
                    for i, exchange in enumerate(st.session_state.conversation_history):
                        st.markdown(f"**You:** {exchange['user_q']}")
                        st.markdown(f"**Advisor:** {exchange['response']}")
                        if i < len(st.session_state.conversation_history) - 1:
                            st.divider()
                else:
                    st.caption("No questions yet. Ask something below!")

            st.write("**Ask follow-up questions:**")
            col_input, col_send = st.columns([4, 1])
            
            with col_input:
                st.text_input(
                    "Your question:",
                    key="user_question_input",
                    placeholder="e.g., What should I look for in Lead V2?",
                    on_change=None
                )
            
            with col_send:
                st.write("")
                st.button("Send ➤", on_click=send_message, use_container_width=True, type="primary")
            
            st.button("🔄 Reset Chat", on_click=reset_chat, use_container_width=True)
            
        else:
            unavailable_msg = st.session_state.get('chatbot_error', 'Check API Key')
            st.markdown(f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fef3c7; color: #92400e; display: flex; align-items: center;'>{SVG_WARNING} Chatbot unavailable: {unavailable_msg}</div>", unsafe_allow_html=True)
