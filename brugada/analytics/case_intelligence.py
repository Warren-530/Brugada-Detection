from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from typing import Any


def _result_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        candidate = float(value)
    except Exception:  # noqa: BLE001
        return default
    if math.isnan(candidate) or math.isinf(candidate):
        return default
    return candidate


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:  # noqa: BLE001
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _parse_evidence_summary(summary: str) -> tuple[int, int, int]:
    text = _safe_str(summary, "S0/M0/W0")
    m = re.match(r"^S(\d+)\/M(\d+)\/W(\d+)$", text)
    if not m:
        return (0, 0, 0)
    return (_safe_int(m.group(1)), _safe_int(m.group(2)), _safe_int(m.group(3)))


def _build_evidence_summary_from_result(result: Any) -> str:
    clinician_explain = _result_get(result, "clinician_explain", {})
    if not isinstance(clinician_explain, dict):
        return "S0/M0/W0"

    evidence_counts = clinician_explain.get("evidence_counts", {})
    if not isinstance(evidence_counts, dict):
        return "S0/M0/W0"

    strong = _safe_int(evidence_counts.get("strong", 0))
    moderate = _safe_int(evidence_counts.get("moderate", 0))
    weak = _safe_int(evidence_counts.get("weak", 0))
    return f"S{strong}/M{moderate}/W{weak}"


def normalize_result_snapshot(result: Any, fallback_record_name: str = "Current Case") -> dict:
    probability_raw = _safe_float(_result_get(result, "probability", 0.0), 0.0)
    probability_display = _safe_float(_result_get(result, "display_probability", probability_raw), probability_raw)

    stability_raw = _safe_float(_result_get(result, "decision_stability", 0.0), 0.0)
    stability_display = _safe_float(_result_get(result, "display_decision_stability", stability_raw), stability_raw)

    recommendation_tier = _safe_str(_result_get(result, "recommendation_tier", ""), "")
    recommendation_text = _safe_str(_result_get(result, "recommendation_text", ""), "")

    clinician_explain = _result_get(result, "clinician_explain", {})
    if isinstance(clinician_explain, dict):
        recommendation_tier = _safe_str(
            clinician_explain.get("recommendation_tier", recommendation_tier),
            recommendation_tier or "routine_clinical_correlation",
        )
        recommendation_text = _safe_str(
            clinician_explain.get("recommendation_text", recommendation_text),
            recommendation_text or "Clinical correlation is recommended.",
        )

    evidence_summary = _safe_str(_result_get(result, "evidence_summary", ""), "")
    if not evidence_summary:
        evidence_summary = _build_evidence_summary_from_result(result)

    label = _safe_str(_result_get(result, "label", "Unknown"), "Unknown")
    record_uid = _safe_str(_result_get(result, "record_uid", ""), "")
    record_name = _safe_str(_result_get(result, "record_name", ""), "")
    if not record_name:
        record_name = _safe_str(_result_get(result, "record", fallback_record_name), fallback_record_name)

    snapshot = {
        "record_uid": record_uid,
        "record_name": record_name,
        "patient_id": _safe_str(_result_get(result, "patient_id", ""), ""),
        "created_at": _safe_str(_result_get(result, "created_at", ""), ""),
        "label": label,
        "is_detected": "brugada syndrome detected" in label.lower(),
        "probability_display": probability_display,
        "probability_raw": probability_raw,
        "decision_stability_display": stability_display,
        "gray_zone": _safe_bool(_result_get(result, "gray_zone", False), False),
        "recommendation_tier": recommendation_tier or "routine_clinical_correlation",
        "recommendation_text": recommendation_text or "Clinical correlation is recommended.",
        "evidence_summary": evidence_summary,
        "evidence_counts": _parse_evidence_summary(evidence_summary),
        "doctor_feedback": _safe_str(_result_get(result, "doctor_feedback", ""), ""),
    }
    return snapshot


def _score_pair(reference: dict, candidate: dict) -> dict:
    prob_delta = abs(reference["probability_display"] - candidate["probability_display"])
    prob_component = max(0.0, 1.0 - prob_delta)

    stability_delta = abs(reference["decision_stability_display"] - candidate["decision_stability_display"])
    stability_component = max(0.0, 1.0 - (stability_delta / 25.0))

    tier_component = 1.0 if reference["recommendation_tier"] == candidate["recommendation_tier"] else 0.0
    gray_component = 1.0 if reference["gray_zone"] == candidate["gray_zone"] else 0.0
    label_component = 1.0 if reference["is_detected"] == candidate["is_detected"] else 0.0

    ref_counts = reference["evidence_counts"]
    cand_counts = candidate["evidence_counts"]
    evidence_l1 = abs(ref_counts[0] - cand_counts[0]) + abs(ref_counts[1] - cand_counts[1]) + abs(ref_counts[2] - cand_counts[2])
    evidence_component = max(0.0, 1.0 - (evidence_l1 / 12.0))

    weighted = (
        0.35 * prob_component
        + 0.20 * tier_component
        + 0.10 * gray_component
        + 0.10 * stability_component
        + 0.15 * evidence_component
        + 0.10 * label_component
    )

    return {
        "similarity_score": round(weighted * 100.0, 2),
        "probability_delta": round(prob_delta * 100.0, 2),
        "stability_delta": round(stability_delta, 2),
        "tier_match": bool(tier_component == 1.0),
        "gray_zone_match": bool(gray_component == 1.0),
        "label_match": bool(label_component == 1.0),
    }


def find_similar_cases(reference_result: Any, candidate_records: list[dict], top_k: int = 5) -> list[dict]:
    if not candidate_records:
        return []

    reference = normalize_result_snapshot(reference_result)
    results: list[dict] = []

    for row in candidate_records:
        if not isinstance(row, dict):
            continue

        candidate = normalize_result_snapshot(row, fallback_record_name="Stored Case")

        if reference["record_uid"] and candidate["record_uid"] and reference["record_uid"] == candidate["record_uid"]:
            continue

        scored = _score_pair(reference, candidate)
        results.append(
            {
                "record_uid": candidate["record_uid"],
                "record_name": candidate["record_name"],
                "patient_id": candidate["patient_id"],
                "label": candidate["label"],
                "probability_display": candidate["probability_display"],
                "decision_stability_display": candidate["decision_stability_display"],
                "gray_zone": candidate["gray_zone"],
                "recommendation_tier": candidate["recommendation_tier"],
                "evidence_summary": candidate["evidence_summary"],
                "created_at": candidate["created_at"],
                **scored,
            }
        )

    results.sort(
        key=lambda item: (
            -_safe_float(item.get("similarity_score", 0.0), 0.0),
            _safe_float(item.get("probability_delta", 0.0), 0.0),
            _safe_float(item.get("stability_delta", 0.0), 0.0),
        )
    )
    return results[: max(1, int(top_k))]


def _parse_iso_utc(value: str) -> datetime | None:
    text = _safe_str(value, "")
    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(text)
    except Exception:  # noqa: BLE001
        return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _apply_time_window(records: list[dict], days: int | None) -> list[dict]:
    if not days or days <= 0:
        return records

    cutoff = datetime.now(timezone.utc).timestamp() - float(days) * 86400.0
    filtered = []
    for row in records:
        if not isinstance(row, dict):
            continue
        dt = _parse_iso_utc(_safe_str(row.get("created_at", ""), ""))
        if dt is None:
            continue
        if dt.timestamp() >= cutoff:
            filtered.append(row)
    return filtered


def compute_operational_metrics(records: list[dict], window_days: int | None = None) -> dict:
    scoped = _apply_time_window(records, window_days)
    if not scoped:
        return {
            "n_records": 0,
            "gray_zone_rate": 0.0,
            "urgent_rate": 0.0,
            "high_risk_rate": 0.0,
            "median_stability": 0.0,
            "mean_risk_pct": 0.0,
            "risk_histogram": [],
            "tier_distribution": [],
        }

    snapshots = [normalize_result_snapshot(item, fallback_record_name="Stored Case") for item in scoped]

    n_records = len(snapshots)
    gray_zone_rate = sum(1 for s in snapshots if s["gray_zone"]) / n_records

    urgent_tiers = {"urgent_cardiology_review", "urgent_review_repeat_ecg_quality_check"}
    urgent_rate = sum(1 for s in snapshots if s["recommendation_tier"] in urgent_tiers) / n_records
    high_risk_rate = sum(1 for s in snapshots if s["probability_display"] >= 0.35) / n_records

    stability_values = sorted(s["decision_stability_display"] for s in snapshots)
    mid = len(stability_values) // 2
    if len(stability_values) % 2 == 0:
        median_stability = 0.5 * (stability_values[mid - 1] + stability_values[mid])
    else:
        median_stability = stability_values[mid]

    mean_risk_pct = 100.0 * sum(s["probability_display"] for s in snapshots) / n_records

    bins = [(0, 10), (10, 20), (20, 35), (35, 50), (50, 75), (75, 100)]
    hist = []
    for low, high in bins:
        count = sum(1 for s in snapshots if low <= (s["probability_display"] * 100.0) < high)
        if high == 100:
            count = sum(1 for s in snapshots if low <= (s["probability_display"] * 100.0) <= high)
        hist.append({"band": f"{low}-{high}%", "count": int(count)})

    tier_count: dict[str, int] = {}
    for s in snapshots:
        tier = s["recommendation_tier"]
        tier_count[tier] = tier_count.get(tier, 0) + 1

    tier_distribution = [
        {"recommendation_tier": tier, "count": count, "rate_pct": round(100.0 * count / n_records, 2)}
        for tier, count in sorted(tier_count.items(), key=lambda item: item[1], reverse=True)
    ]

    return {
        "n_records": int(n_records),
        "gray_zone_rate": float(gray_zone_rate * 100.0),
        "urgent_rate": float(urgent_rate * 100.0),
        "high_risk_rate": float(high_risk_rate * 100.0),
        "median_stability": float(median_stability),
        "mean_risk_pct": float(mean_risk_pct),
        "risk_histogram": hist,
        "tier_distribution": tier_distribution,
    }


def compute_feedback_proxy_metrics(records: list[dict], window_days: int | None = None) -> dict:
    scoped = _apply_time_window(records, window_days)
    snapshots = [normalize_result_snapshot(item, fallback_record_name="Stored Case") for item in scoped]

    feedback_rows = [
        s
        for s in snapshots
        if _safe_str(s.get("doctor_feedback", ""), "").lower() in {"agree", "disagree"}
    ]

    if not feedback_rows:
        return {
            "n_feedback": 0,
            "agreement_rate": 0.0,
            "disagreement_rate": 0.0,
            "proxy_confusion": [],
            "risk_band_disagreement": [],
        }

    n_feedback = len(feedback_rows)
    n_agree = sum(1 for s in feedback_rows if _safe_str(s["doctor_feedback"]).lower() == "agree")
    n_disagree = n_feedback - n_agree

    agree_rate = 100.0 * n_agree / n_feedback
    disagree_rate = 100.0 * n_disagree / n_feedback

    confusion = {
        "predicted_detected": {"agree": 0, "disagree": 0},
        "predicted_normal": {"agree": 0, "disagree": 0},
    }

    for s in feedback_rows:
        pred_bucket = "predicted_detected" if s["is_detected"] else "predicted_normal"
        fb = _safe_str(s["doctor_feedback"]).lower()
        if fb not in {"agree", "disagree"}:
            continue
        confusion[pred_bucket][fb] += 1

    proxy_confusion = [
        {
            "predicted_label_group": "Detected",
            "agree": confusion["predicted_detected"]["agree"],
            "disagree": confusion["predicted_detected"]["disagree"],
        },
        {
            "predicted_label_group": "Normal",
            "agree": confusion["predicted_normal"]["agree"],
            "disagree": confusion["predicted_normal"]["disagree"],
        },
    ]

    bins = [(0, 20), (20, 35), (35, 50), (50, 75), (75, 100)]
    band_rows = []
    for low, high in bins:
        in_band = [s for s in feedback_rows if low <= (s["probability_display"] * 100.0) < high]
        if high == 100:
            in_band = [s for s in feedback_rows if low <= (s["probability_display"] * 100.0) <= high]

        if not in_band:
            band_rows.append({"risk_band": f"{low}-{high}%", "n_feedback": 0, "disagree_rate_pct": 0.0})
            continue

        band_disagree = sum(1 for s in in_band if _safe_str(s["doctor_feedback"]).lower() == "disagree")
        band_rows.append(
            {
                "risk_band": f"{low}-{high}%",
                "n_feedback": len(in_band),
                "disagree_rate_pct": round(100.0 * band_disagree / len(in_band), 2),
            }
        )

    return {
        "n_feedback": int(n_feedback),
        "agreement_rate": float(agree_rate),
        "disagreement_rate": float(disagree_rate),
        "proxy_confusion": proxy_confusion,
        "risk_band_disagreement": band_rows,
    }
