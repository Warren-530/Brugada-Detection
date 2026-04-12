import pandas as pd
import streamlit as st

from brugada.storage.record_store import (
    get_record_counts,
    get_record_payload,
    list_records,
    update_record_feedback,
    update_record_patient_id,
    update_record_status_bulk,
)
from brugada.ui.components import SVG_ERROR, SVG_INFO


def render_records_tab():
    st.subheader("Records Center")
    st.caption("Persistent local registry for record history, retrieval, and lifecycle management.")

    if not st.session_state.record_store_ready:
        store_error = st.session_state.get("record_store_error", "Unknown storage initialization error")
        st.markdown(
            f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #fee2e2; color: #991b1b; display: flex; align-items: center;'>{SVG_ERROR} Local record store unavailable: {store_error}</div>",
            unsafe_allow_html=True,
        )
        return

    try:
        counts = get_record_counts()
    except Exception as count_exc:  # noqa: BLE001
        st.error(f"Unable to read record counters: {count_exc}")
        counts = {"active": 0, "archived": 0, "deleted": 0}

    with st.expander("Registry Overview", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Active", str(counts.get("active", 0)))
        c2.metric("Archived", str(counts.get("archived", 0)))
        c3.metric("Deleted", str(counts.get("deleted", 0)))

    status_map = {
        "Active": "active",
        "Archived": "archived",
        "Deleted": "deleted",
        "All": "all",
    }
    status_display = str(st.session_state.get("records_status_filter", "Active"))
    selected_status = status_map.get(status_display, "active")
    search_query = str(st.session_state.get("records_search_query", ""))

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
        return

    records_df = pd.DataFrame(records)
    records_df["risk_score_pct"] = (records_df["probability_display"] * 100.0).round(2)
    records_df["boundary_distance_pp"] = records_df["decision_stability_display"].round(2)

    label_series = records_df["label"].fillna("").astype(str)
    sick_mask = label_series.str.contains("Brugada Syndrome Detected", case=False, na=False)
    gray_zone_mask = records_df["gray_zone"].fillna(False).astype(bool)

    total_in_view = int(len(records_df))
    sick_count = int(sick_mask.sum())
    not_sick_count = int(total_in_view - sick_count)
    gray_zone_count = int(gray_zone_mask.sum())

    # Get patient_id from sidebar Optional Metadata
    sidebar_patient_id = st.session_state.get("patient_id_input", "").strip() or None

    # Auto-fill empty patient_id cells with sidebar patient_id
    if sidebar_patient_id:
        records_df["patient_id"] = records_df["patient_id"].fillna(sidebar_patient_id)
        records_df["patient_id"] = records_df["patient_id"].apply(
            lambda x: sidebar_patient_id if (x == "" or pd.isna(x)) else x
        )

    editor_df = pd.DataFrame(
        {
            "select": False,
            "record_uid": records_df["record_uid"],
            "record": records_df["record_name"],
            "patient_id": records_df["patient_id"],
            "doctor_feedback": records_df["doctor_feedback"],
            "doctor_feedback_note": records_df["doctor_feedback_note"],
            "label": records_df["label"],
            "risk_score_%": records_df["risk_score_pct"],
            "recommendation_tier": records_df["recommendation_tier"],
            "status": records_df["status"],
            "timestamp_utc": records_df["created_at"],
            "decision_stability_pp": records_df["boundary_distance_pp"],
            "evidence_summary": records_df["evidence_summary"],
        }
    )

    with st.expander("Patient Status Summary and Sick-Flag Reasons", expanded=True):
        st.caption("Summary for records in the current filter and search view.")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Patients in View", str(total_in_view))
        s2.metric("Sick (Brugada Detected)", str(sick_count))
        s3.metric("Not Sick", str(not_sick_count))
        s4.metric("Gray-Zone", str(gray_zone_count))
        st.divider()
        st.write("Why patients are flagged as sick")
        if sick_count == 0:
            st.markdown(
                f"<div style='margin-bottom: 1rem; padding: 0.8rem; border-radius: 0.5rem; background-color: #e0f2fe; color: #075985; display: flex; align-items: center;'>{SVG_INFO} No records labeled as Brugada Syndrome Detected in the current view.</div>",
                unsafe_allow_html=True,
            )
        else:
            sick_df = records_df[sick_mask].copy()

            tier_summary = (
                sick_df.groupby("recommendation_tier", dropna=False)
                .agg(
                    patients=("record_uid", "count"),
                    mean_risk_score_pct=("risk_score_pct", "mean"),
                    mean_boundary_distance_pp=("boundary_distance_pp", "mean"),
                )
                .reset_index()
                .sort_values(["patients", "mean_risk_score_pct"], ascending=[False, False])
            )
            tier_summary["recommendation_tier"] = tier_summary["recommendation_tier"].fillna("unknown")
            tier_summary["mean_risk_score_pct"] = tier_summary["mean_risk_score_pct"].round(2)
            tier_summary["mean_boundary_distance_pp"] = tier_summary["mean_boundary_distance_pp"].round(2)

            evidence_summary = (
                sick_df.groupby("evidence_summary", dropna=False)
                .agg(patients=("record_uid", "count"))
                .reset_index()
                .sort_values("patients", ascending=False)
                .head(5)
            )
            evidence_summary["evidence_summary"] = evidence_summary["evidence_summary"].fillna("S0/M0/W0")

            top_recommendations = (
                sick_df["recommendation_text"]
                .fillna("Clinical correlation is recommended.")
                .replace("", "Clinical correlation is recommended.")
                .value_counts()
                .head(3)
            )

            st.caption("Recommendation tiers and risk profile among sick patients.")
            st.dataframe(tier_summary, use_container_width=True, hide_index=True)

            rs1, rs2 = st.columns(2)
            with rs1:
                st.caption("Most common evidence strength patterns (S/M/W).")
                st.dataframe(evidence_summary, use_container_width=True, hide_index=True)
            with rs2:
                st.caption("Top recommendation reasons returned by the model.")
                for recommendation_text, n_patients in top_recommendations.items():
                    st.write(f"- {int(n_patients)} patient(s): {recommendation_text}")
    selected_items = []
    summary_selected_uid = None
    with st.expander("Records Table, Patient ID Edits, and Bulk Actions", expanded=True):
        # Show sidebar patient_id status
        if sidebar_patient_id:
            st.info(f"📋 **Current Patient ID from Optional Metadata**: `{sidebar_patient_id}` - Empty patient_id fields will use this value")
        
        status_display = st.radio(
            "Status Filter",
            ["Active", "Archived", "Deleted", "All"],
            horizontal=True,
            key="records_status_filter",
        )
        search_query = st.text_input(
            "Search Records",
            key="records_search_query",
            placeholder="Record name, patient ID, or label",
        )

        edited_df = st.data_editor(
            editor_df,
            use_container_width=True,
            hide_index=True,
            key="records_table_editor",
            column_config={
                "select": st.column_config.CheckboxColumn("Select", help="Select one or more records for bulk actions."),
                "record_uid": None,
                "patient_id": st.column_config.TextColumn("patient_id", help="Editable patient identifier."),
                "doctor_feedback": st.column_config.SelectboxColumn(
                    "doctor_feedback",
                    options=["", "agree", "disagree"],
                    help="Doctor review label used for future model training.",
                ),
                "doctor_feedback_note": st.column_config.TextColumn(
                    "doctor_feedback_note",
                    help="Optional rationale supporting agree/disagree feedback.",
                ),
            },
            disabled=[
                "record_uid",
                "timestamp_utc",
                "record",
                "label",
                "risk_score_%",
                "decision_stability_pp",
                "recommendation_tier",
                "status",
                "evidence_summary",
            ],
        )

        original_patient_by_uid = {
            str(uid): str(pid or "")
            for uid, pid in zip(records_df["record_uid"], records_df["patient_id"])
        }
        original_feedback_by_uid = {
            str(item["record_uid"]): str(item.get("doctor_feedback", "") or "")
            for item in records
        }
        original_feedback_note_by_uid = {
            str(item["record_uid"]): str(item.get("doctor_feedback_note", "") or "")
            for item in records
        }
        patient_changes: list[tuple[str, str]] = []
        feedback_changes: list[tuple[str, str, str]] = []
        for _, row in edited_df.iterrows():
            row_uid = str(row["record_uid"])
            edited_patient_id = str(row.get("patient_id", "") or "").strip()
            if edited_patient_id != original_patient_by_uid.get(row_uid, ""):
                patient_changes.append((row_uid, edited_patient_id))

            edited_feedback = str(row.get("doctor_feedback", "") or "").strip().lower()
            if edited_feedback not in {"agree", "disagree"}:
                edited_feedback = ""
            edited_feedback_note = str(row.get("doctor_feedback_note", "") or "").strip()

            if (
                edited_feedback != original_feedback_by_uid.get(row_uid, "")
                or edited_feedback_note != original_feedback_note_by_uid.get(row_uid, "")
            ):
                feedback_changes.append((row_uid, edited_feedback, edited_feedback_note))

        selected_df = edited_df[edited_df["select"] == True]
        selected_uids = [str(uid) for uid in selected_df["record_uid"].tolist()]
        selected_items = [item for item in records if item["record_uid"] in selected_uids]

        e1, e2, e3 = st.columns([1, 1, 2])
        with e1:
            if patient_changes:
                if st.button("Save Patient ID Edits", use_container_width=True, key="records_save_patient_id"):
                    updated_count = 0
                    for row_uid, edited_patient_id in patient_changes:
                        if update_record_patient_id(row_uid, edited_patient_id):
                            updated_count += 1
                    st.session_state.persistence_notice = f"Updated patient_id for {updated_count} record(s)."
                    st.rerun()
        with e2:
            if feedback_changes:
                if st.button("Save Doctor Feedback", use_container_width=True, key="records_save_doctor_feedback"):
                    updated_count = 0
                    for row_uid, edited_feedback, edited_feedback_note in feedback_changes:
                        if update_record_feedback(row_uid, edited_feedback, edited_feedback_note):
                            updated_count += 1
                    st.session_state.persistence_notice = f"Updated doctor feedback for {updated_count} record(s)."
                    st.rerun()
        with e3:
            notices = []
            if patient_changes:
                notices.append("Unsaved patient_id edits detected.")
            if feedback_changes:
                notices.append("Unsaved doctor_feedback edits detected.")

            if notices:
                st.caption(" ".join(notices) + " Use the save buttons to persist changes.")
            else:
                st.caption("Tip: edit patient_id and doctor_feedback fields directly in the table to enable save actions.")

        st.divider()
        a1, a2, a3, a4 = st.columns(4)

        with a1:
            if st.button("Load Selected", key="records_load_selected", use_container_width=True):
                if not selected_items:
                    st.warning("Select at least one record to load.")
                elif len(selected_items) == 1:
                    selected_item = selected_items[0]
                    payload = get_record_payload(selected_item["record_uid"])
                    if payload is None:
                        st.error("Unable to load selected record payload from local storage.")
                    else:
                        st.session_state.records_loaded_result = payload
                        st.session_state.last_ml_result = payload
                        if "batch_results" in st.session_state:
                            del st.session_state.batch_results
                        st.session_state.persistence_notice = (
                            f"Loaded saved record '{selected_item['record_name']}' into Clinical Report."
                        )
                        if st.session_state.chatbot_ready:
                            st.session_state.chatbot.reset_conversation()
                            st.session_state.conversation_history = []
                        st.rerun()
                else:
                    batch_loaded = []
                    for item in selected_items:
                        payload = get_record_payload(item["record_uid"])
                        if not isinstance(payload, dict):
                            continue
                        batch_loaded.append(
                            {
                                "record": item["record_name"],
                                "label": payload.get("label", item["label"]),
                                "probability": float(payload.get("display_probability", item["probability_display"])),
                                "decision_stability": float(
                                    payload.get("display_decision_stability", item["decision_stability_display"])
                                ),
                                "gray_zone": bool(payload.get("gray_zone", item["gray_zone"])),
                                "recommendation_tier": payload.get("recommendation_tier", item["recommendation_tier"]),
                                "evidence_strength_summary": item["evidence_summary"],
                                "raw": payload,
                            }
                        )

                    if not batch_loaded:
                        st.error("Unable to load any selected payloads from local storage.")
                    else:
                        st.session_state.records_loaded_result = None
                        st.session_state.last_ml_result = None
                        st.session_state.batch_results = batch_loaded
                        st.session_state.current_view = "Batch Summary"
                        st.session_state.persistence_notice = f"Loaded {len(batch_loaded)} records into Clinical Report batch view."
                        if st.session_state.chatbot_ready:
                            st.session_state.chatbot.reset_conversation()
                            st.session_state.conversation_history = []
                        st.rerun()

        with a2:
            if st.button("Archive Selected", key="records_archive_selected", use_container_width=True):
                if not selected_uids:
                    st.warning("Select at least one record to archive.")
                else:
                    changed = update_record_status_bulk(selected_uids, "archived")
                    st.session_state.persistence_notice = f"Archived {changed} record(s)."
                    st.rerun()

        with a3:
            if st.button("Delete Selected", key="records_delete_selected", use_container_width=True):
                if not selected_uids:
                    st.warning("Select at least one record to delete.")
                else:
                    changed = update_record_status_bulk(selected_uids, "deleted")
                    st.session_state.persistence_notice = f"Moved {changed} record(s) to deleted status."
                    st.rerun()

        with a4:
            if st.button("Restore Selected", key="records_restore_selected", use_container_width=True):
                if not selected_uids:
                    st.warning("Select at least one record to restore.")
                else:
                    changed = update_record_status_bulk(selected_uids, "active")
                    st.session_state.persistence_notice = f"Restored {changed} record(s) to active status."
                    st.rerun()

    if selected_items:
        summary_options = {
            (
                f"{item['created_at']} | {item['record_name']} | "
                f"{item['label']} | {item['probability_display'] * 100.0:.2f}%"
            ): item["record_uid"]
            for item in selected_items
        }
        summary_option_labels = list(summary_options.keys())

        previous_summary_uid = st.session_state.get("records_summary_uid", "")
        default_summary_idx = 0
        if previous_summary_uid:
            for i, label in enumerate(summary_option_labels):
                if summary_options[label] == previous_summary_uid:
                    default_summary_idx = i
                    break

        selected_summary_label = st.selectbox(
            "Summary Record",
            options=summary_option_labels,
            index=default_summary_idx,
            key="records_summary_selector",
            help="Choose which selected record to show in the summary panel.",
        )
        summary_selected_uid = summary_options[selected_summary_label]
        st.session_state.records_summary_uid = summary_selected_uid

    if selected_items:
        selected_item = next((item for item in selected_items if item["record_uid"] == summary_selected_uid), selected_items[0])
        with st.expander("Selected Record Summary", expanded=False):
            st.write(f"Record UID: {selected_item['record_uid']}")
            st.write(f"Label: {selected_item['label']}")
            st.write(f"Recommendation Tier: {selected_item['recommendation_tier']}")
            st.write(f"Evidence Summary: {selected_item['evidence_summary']}")
            st.write(f"Recommendation: {selected_item['recommendation_text']}")
            st.write(f"Doctor Feedback: {selected_item.get('doctor_feedback', '') or 'not set'}")
            st.write(f"Doctor Feedback Note: {selected_item.get('doctor_feedback_note', '') or 'none'}")