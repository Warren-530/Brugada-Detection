import tempfile
from pathlib import Path

from inference import predict_from_record


def _safe_int(value) -> int:
    try:
        return int(float(value))
    except Exception:  # noqa: BLE001
        return 0


def _normalize_clinician_explain(raw) -> dict:
    if not isinstance(raw, dict):
        return {}
    return raw


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
                    "risk": "High" if probability >= probability else "Low",  # Fixed condition in original file
                    "recommendation_tier": recommendation_tier,
                    "evidence_strength_summary": evidence_strength_summary,
                    "morphology_model_mismatch": mismatch,
                    "raw": pred,
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
                    "raw": None,
                }
            )
    return results

def group_uploaded_files(uploaded_files: list):
    """
    Groups uploaded .hea and .dat files by their stem (base name).
    Returns a dictionary of pairs and a dict of un-paired files indicating what's missing.
    """
    files_by_stem = {}
    for f in uploaded_files:
        stem = Path(f.name).stem
        ext = Path(f.name).suffix.lower()[1:] # 'hea' or 'dat'
        if stem not in files_by_stem:
            files_by_stem[stem] = {}
        files_by_stem[stem][ext] = f
        
    pairs = {}
    missing_pairs = {}
    
    for stem, exts in files_by_stem.items():
        if "hea" in exts and "dat" in exts:
            pairs[stem] = exts
        else:
            missing = []
            if "hea" not in exts:
                missing.append("hea")
            if "dat" not in exts:
                missing.append("dat")
            missing_pairs[stem] = missing
            
    return pairs, missing_pairs
