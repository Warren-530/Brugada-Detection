from __future__ import annotations

import numpy as np
import scipy.signal as signal
import wfdb
from pathlib import Path

from brugada.inference.features import (
    _build_clinician_explain,
    _build_explanation,
    extract_clinical_package,
    generate_cwt_scalograms,
    remap_probability_for_display,
)
from brugada.inference.models import (
    DECISION_THRESHOLD,
    DISPLAY_THRESHOLD,
    MODELS,
    UPPER_BOUND,
    LeadSpatialAttention,
    load_all_models,
)


def extract_patient_metadata(record_path: str) -> dict:
    """Extract patient metadata from WFDB record header."""
    record_path = Path(record_path)

    # First try to read metadata directly from .hea file
    hea_path = record_path.with_suffix('.hea')
    patient_info = {}
    record_name = record_path.name

    if hea_path.exists():
        try:
            with open(hea_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if line.startswith('#') and ':' in line:
                    # Parse comment lines like "# Patient: PT-001"
                    comment_content = line[1:].strip()  # Remove #
                    if ":" in comment_content:
                        key, value = comment_content.split(":", 1)
                        key = key.strip().lower()
                        value = value.strip()
                        patient_info[key] = value
        except Exception:
            pass  # Ignore file reading errors

    # Try to use WFDB to get additional metadata (without reading signal data)
    try:
        # Use rdheader to avoid reading signal data
        header = wfdb.rdheader(str(record_path))

        metadata = {
            "record_name": header.record_name,
            "n_sig": header.n_sig,
            "fs": header.fs,
            "sig_len": header.sig_len if hasattr(header, 'sig_len') else None,
            "units": header.units,
            "sig_name": header.sig_name,
            "comments": header.comments,
        }

        # Add any comments from WFDB
        if header.comments:
            for comment in header.comments:
                comment = comment.strip()
                if ":" in comment:
                    key, value = comment.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    patient_info[key] = value

    except Exception:
        # Fallback metadata if WFDB fails
        metadata = {
            "record_name": record_name,
            "error": "WFDB header read failed",
        }

    metadata["patient_info"] = patient_info if patient_info else None

    # Try to extract patient ID from various sources
    patient_id = None

    # Check parsed patient info for patient ID
    for key in ["patient", "id", "patient_id", "subject", "patient id"]:
        if key in patient_info:
            patient_id = patient_info[key]
            break

    # Check if record name looks like a patient ID
    if not patient_id and record_name:
        # If record name contains numbers or looks like an ID, use it
        if any(char.isdigit() for char in record_name):
            patient_id = record_name

    # Fallback to record name if no patient ID found
    if not patient_id:
        patient_id = record_name

    metadata["extracted_patient_id"] = patient_id

    return metadata


def preprocess_signal(record_path: str) -> tuple[np.ndarray, float]:
    """Read WFDB, apply 3rd-order bandpass, and pad or truncate."""
    record = wfdb.rdrecord(str(record_path))
    raw_signal = record.p_signal
    fs = record.fs

    nyquist = 0.5 * fs
    b, a = signal.butter(3, [0.5 / nyquist, 40.0 / nyquist], btype="band")
    clean_signal = signal.filtfilt(b, a, raw_signal, axis=0)

    target_length = 1200
    curr_len = clean_signal.shape[0]

    if curr_len >= target_length:
        standardized_signal = clean_signal[:target_length, :]
    else:
        pad_len = target_length - curr_len
        standardized_signal = np.pad(clean_signal, ((0, pad_len), (0, 0)), mode="constant")

    return standardized_signal, fs


def predict_from_record(record_path: str) -> dict:
    """Core multi-view stacking inference pipeline."""
    load_all_models()

    required_runtime_keys = {
        "resnet_feat",
        "eegnet_feat",
        "blstm_feat",
        "cwt_feat",
        "scaler",
        "selector",
        "meta",
    }
    missing_runtime_keys = sorted(required_runtime_keys.difference(MODELS.keys()))
    if missing_runtime_keys:
        raise RuntimeError(
            "Model pipeline is not fully initialized. Missing keys: "
            f"{', '.join(missing_runtime_keys)}. "
            "Restart the app and ensure scikit-learn==1.6.1 with all model artifacts in models/."
        )

    base_signal, fs = preprocess_signal(record_path)

    signal_1d = np.expand_dims(base_signal, axis=0).astype(np.float32)
    signal_2d = generate_cwt_scalograms(base_signal).astype(np.float32)

    feat_resnet = MODELS["resnet_feat"].predict(signal_1d, verbose=0)
    feat_eegnet = MODELS["eegnet_feat"].predict(signal_1d, verbose=0)
    feat_blstm = MODELS["blstm_feat"].predict(signal_1d, verbose=0)
    feat_cwt = MODELS["cwt_feat"].predict(signal_2d, verbose=0)

    emb_norms = {
        "resnet": float(np.linalg.norm(feat_resnet)),
        "eegnet": float(np.linalg.norm(feat_eegnet)),
        "blstm": float(np.linalg.norm(feat_blstm)),
        "cwt_cnn": float(np.linalg.norm(feat_cwt)),
    }
    total_norm = sum(emb_norms.values()) + 1e-8
    model_contributions = {k: float(v / total_norm * 100.0) for k, v in emb_norms.items()}

    clinical_feat, highlighted_segments, evidence = extract_clinical_package(base_signal, fs=fs)
    feat_clinical = np.expand_dims(clinical_feat, axis=0)

    final_features = np.concatenate(
        [feat_clinical, feat_resnet, feat_eegnet, feat_blstm, feat_cwt],
        axis=1,
    )

    expected_n = getattr(MODELS["scaler"], "n_features_in_", None)
    if expected_n is not None and final_features.shape[1] != int(expected_n):
        raise ValueError(f"Feature dimension mismatch: got {final_features.shape[1]}, expected {int(expected_n)}")

    scaled_features = MODELS["scaler"].transform(final_features)
    selected_features = MODELS["selector"].transform(scaled_features)

    probabilities = MODELS["meta"].predict_proba(selected_features)[0]
    brugada_proba = probabilities[1]

    is_detected = brugada_proba >= DECISION_THRESHOLD
    in_gray_zone = DECISION_THRESHOLD <= brugada_proba <= UPPER_BOUND

    class_support = float(brugada_proba if is_detected else (1.0 - brugada_proba))

    max_margin = max(DECISION_THRESHOLD, 1.0 - DECISION_THRESHOLD)
    decision_margin = abs(brugada_proba - DECISION_THRESHOLD)
    margin_ratio = float(np.clip(decision_margin / max_margin, 0.0, 1.0))
    decision_confidence = float(0.5 + 0.5 * margin_ratio)
    confidence_percent = decision_confidence * 100.0

    stability_percent = float(decision_margin * 100.0)
    class_support_percent = class_support * 100.0

    display_probability = remap_probability_for_display(brugada_proba)
    display_threshold = DISPLAY_THRESHOLD
    display_gray_zone_upper = remap_probability_for_display(UPPER_BOUND)
    display_margin = abs(display_probability - display_threshold)
    display_max_margin = max(display_threshold, 1.0 - display_threshold)
    display_margin_ratio = float(np.clip(display_margin / display_max_margin, 0.0, 1.0))
    display_confidence = float((0.5 + 0.5 * display_margin_ratio) * 100.0)
    display_stability = float(display_margin * 100.0)

    explanation = _build_explanation(
        display_probability=display_probability,
        display_threshold=display_threshold,
        is_detected=is_detected,
        in_gray_zone=in_gray_zone,
        evidence=evidence,
    )
    clinician_explain = _build_clinician_explain(brugada_proba, is_detected, in_gray_zone, evidence)

    return {
        "status": "success",
        "label": "Brugada Syndrome Detected" if is_detected else "Normal ECG Pattern",
        "risk": "High" if is_detected else "Low",
        "probability": float(brugada_proba),
        "confidence": float(confidence_percent),
        "decision_stability": float(stability_percent),
        "class_support": float(class_support_percent),
        "decision_threshold": float(DECISION_THRESHOLD),
        "display_probability": float(display_probability),
        "display_threshold": float(display_threshold),
        "display_confidence": float(display_confidence),
        "display_decision_stability": float(display_stability),
        "display_gray_zone_upper": float(display_gray_zone_upper),
        "gray_zone": bool(in_gray_zone),
        "highlighted_segments": highlighted_segments,
        "lead_names": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "explanation": explanation,
        "clinical_evidence": evidence,
        "model_contributions": model_contributions,
        "clinician_explain": clinician_explain,
        "signal_for_plot": base_signal,
        "fs": fs,
    }


__all__ = [
    "LeadSpatialAttention",
    "MODELS",
    "DECISION_THRESHOLD",
    "UPPER_BOUND",
    "DISPLAY_THRESHOLD",
    "load_all_models",
    "preprocess_signal",
    "predict_from_record",
    "remap_probability_for_display",
]
