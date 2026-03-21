from __future__ import annotations
import os
import joblib
import numpy as np
import wfdb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, Dense, Reshape
import scipy.signal as signal
import scipy.stats as stats
import cv2  
import pywt 
import neurokit2 as nk

# =============================================================================
# Leads Attention Mechanism (with serialization decorator)
# =============================================================================
@keras.utils.register_keras_serializable()
class LeadSpatialAttention(Layer):
    def __init__(self, **kwargs):
        super(LeadSpatialAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_leads = input_shape[-1]
        self.dense1 = Dense(self.num_leads // 2, activation='relu')
        self.dense2 = Dense(self.num_leads, activation='sigmoid')
        super(LeadSpatialAttention, self).build(input_shape)

    def call(self, inputs):
        squeeze = tf.reduce_mean(inputs, axis=1)
        excitation = self.dense1(squeeze)
        attention_weights = self.dense2(excitation)
        attention_weights = Reshape((1, self.num_leads))(attention_weights)
        return inputs * attention_weights

# =============================================================================
# Global Model Cache
# =============================================================================
MODELS = {}

FEATURE_LAYER_BY_MODEL = {
    'resnet': 'nature_resnet_feature',
    'blstm': 'bilstm_feature',
    'eegnet': 'eegnet_feature',
    'cwt_cnn': 'cwt_feature',
}

# Notebook-aligned deployment thresholds
# Clinical safety override to preserve maximal recall in deployment.
DECISION_THRESHOLD = 0.050
LOWER_BOUND = 0.040
UPPER_BOUND = 0.060


def _evidence_tier(score: float, segments: int, source: str) -> str:
    if source == "none" or segments <= 0:
        return "weak"
    if score >= 0.18 and segments >= 2:
        return "strong"
    if score >= 0.08:
        return "moderate"
    return "weak"


def _evidence_reliability(segments: int, source: str) -> str:
    if source == "none" or segments <= 0:
        return "poor"
    if source == "dwt" and segments >= 2:
        return "good"
    return "fair"


def _build_clinician_explain(probability: float, is_detected: bool, in_gray_zone: bool, evidence: list[dict]) -> dict:
    strong_count = sum(1 for e in evidence if e.get("tier") == "strong")
    moderate_count = sum(1 for e in evidence if e.get("tier") == "moderate")
    weak_count = sum(1 for e in evidence if e.get("tier") == "weak")

    reliable_count = sum(1 for e in evidence if e.get("reliability") in {"good", "fair"})
    morphology_score = float(np.mean([float(e.get("score", 0.0)) for e in evidence])) if evidence else 0.0
    dominant_lead = max(evidence, key=lambda x: float(x.get("score", 0.0))).get("lead", "N/A") if evidence else "N/A"

    if in_gray_zone:
        recommendation_tier = "gray_zone_priority_review"
        recommendation_text = "Prioritize manual over-read, correlate with symptoms, and consider repeat ECG acquisition."
    elif is_detected and strong_count > 0:
        recommendation_tier = "urgent_cardiology_review"
        recommendation_text = "Urgent cardiology review is recommended due to high AI risk with robust V1-V3 morphology evidence."
    elif is_detected:
        recommendation_tier = "urgent_review_repeat_ecg_quality_check"
        recommendation_text = "Urgent review is recommended; repeat ECG and lead-quality check are advised due to weak morphology evidence."
    else:
        recommendation_tier = "routine_clinical_correlation"
        recommendation_text = "Low AI risk; continue routine clinical correlation and follow standard workflow."

    morphology_model_mismatch = bool((is_detected and strong_count == 0) or ((not is_detected) and strong_count >= 2))

    next_actions = [
        "Verify V1-V3 lead placement and signal quality.",
        "Correlate with symptoms, syncope history, and family history.",
    ]
    if in_gray_zone or is_detected:
        next_actions.append("Consider repeat ECG for morphology confirmation.")
    if is_detected:
        next_actions.append("Escalate to cardiology review pathway.")

    return {
        "recommendation_tier": recommendation_tier,
        "recommendation_text": recommendation_text,
        "evidence_counts": {
            "strong": int(strong_count),
            "moderate": int(moderate_count),
            "weak": int(weak_count),
            "reliable": int(reliable_count),
        },
        "morphology_score": morphology_score,
        "dominant_lead": dominant_lead,
        "morphology_model_mismatch": morphology_model_mismatch,
        "next_actions": next_actions,
    }


def _extract_stat_features(sig_2d: np.ndarray) -> list[float]:
    stat_feat: list[float] = []
    for lead in range(12):
        sig = sig_2d[:, lead]
        stat_feat.extend([
            np.max(sig),
            np.min(sig),
            np.std(sig),
            np.var(sig),
            stats.skew(sig),
            stats.kurtosis(sig),
            np.sqrt(np.mean(sig**2)),
        ])
    return stat_feat


def _merge_segments(segments: list[tuple[int, int]], merge_gap: int = 2) -> list[tuple[int, int]]:
    if not segments:
        return []
    segments = sorted(segments, key=lambda x: x[0])
    merged = [segments[0]]
    for start, end in segments[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + merge_gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _extract_expert_features_and_highlights(
    sig_2d: np.ndarray,
    fs: float,
) -> tuple[list[float], dict[str, list[tuple[int, int]]], list[dict]]:
    expert_feat: list[float] = []
    highlights: dict[str, list[tuple[int, int]]] = {"V1": [], "V2": [], "V3": []}
    evidence: list[dict] = []
    v_leads = [6, 7, 8]
    lead_names = ["V1", "V2", "V3"]

    for l_idx, lead_name in zip(v_leads, lead_names):
        sig = sig_2d[:, l_idx]
        try:
            _, rpeaks = nk.ecg_peaks(sig, sampling_rate=fs)
            j_offsets = []
            source_method = "none"
            for method in ("dwt", "peak"):
                try:
                    _, waves = nk.ecg_delineate(sig, rpeaks, sampling_rate=fs, method=method)
                    offsets = waves.get("ECG_R_Offsets", [])
                    j_offsets = sorted(
                        {
                            int(x)
                            for x in offsets
                            if x is not None and not np.isnan(x) and 0 <= int(x) < len(sig) - 8
                        }
                    )
                    if j_offsets:
                        source_method = method
                        break
                except Exception:
                    continue

            if j_offsets:
                j_heights = [sig[j] for j in j_offsets]
                avg_j_height = float(np.mean(j_heights))

                v20, v40, v60, v80 = [], [], [], []
                lead_segments: list[tuple[int, int]] = []
                for j in j_offsets:
                    if j + 8 < len(sig):
                        v20.append(sig[j + 2])
                        v40.append(sig[j + 4])
                        v60.append(sig[j + 6])
                        v80.append(sig[j + 8])
                        lead_segments.append((max(0, j - 2), min(len(sig) - 1, j + 10)))

                avg_slope = float((np.mean(v80) - avg_j_height) / 8) if v80 else 0.0
                curvature = float((np.mean(v40) - np.mean(v20)) - (np.mean(v80) - np.mean(v60))) if v80 else 0.0

                expert_feat.extend([avg_j_height, avg_slope, curvature])
                highlights[lead_name] = _merge_segments(lead_segments)[:12]

                # Heuristic evidence score for clinical explanation only (not used by classifier).
                score = max(0.0, avg_j_height) + max(0.0, -avg_slope) + max(0.0, curvature)
                evidence.append(
                    {
                        "lead": lead_name,
                        "j_height": avg_j_height,
                        "st_slope": avg_slope,
                        "curvature": curvature,
                        "segments": len(highlights[lead_name]),
                        "source": source_method,
                        "score": float(score),
                        "tier": _evidence_tier(float(score), len(highlights[lead_name]), source_method),
                        "reliability": _evidence_reliability(len(highlights[lead_name]), source_method),
                    }
                )
            else:
                expert_feat.extend([0, 0, 0])
                evidence.append(
                    {
                        "lead": lead_name,
                        "j_height": 0.0,
                        "st_slope": 0.0,
                        "curvature": 0.0,
                        "segments": 0,
                        "source": "none",
                        "score": 0.0,
                        "tier": "weak",
                        "reliability": "poor",
                    }
                )
        except Exception:
            expert_feat.extend([0, 0, 0])
            evidence.append(
                {
                    "lead": lead_name,
                    "j_height": 0.0,
                    "st_slope": 0.0,
                    "curvature": 0.0,
                    "segments": 0,
                    "source": "none",
                    "score": 0.0,
                    "tier": "weak",
                    "reliability": "poor",
                }
            )

    return expert_feat, highlights, evidence


def extract_clinical_package(sig_2d: np.ndarray, fs: float) -> tuple[np.ndarray, dict[str, list[tuple[int, int]]], list[dict]]:
    stat_feat = _extract_stat_features(sig_2d)
    expert_feat, highlights, evidence = _extract_expert_features_and_highlights(sig_2d, fs)
    return np.array(stat_feat + expert_feat), highlights, evidence


def _build_explanation(probability: float, is_detected: bool, in_gray_zone: bool, evidence: list[dict]) -> str:
    sorted_evidence = sorted(evidence, key=lambda x: x["score"], reverse=True)
    informative_leads = [
        e
        for e in sorted_evidence
        if e.get("segments", 0) > 0 and e.get("tier", "weak") in {"strong", "moderate"}
    ][:2]
    weak_leads = [e for e in sorted_evidence if e.get("segments", 0) > 0 and e.get("tier", "weak") == "weak"]

    if informative_leads:
        lead_summary = ", ".join(
            [
                f"{e['lead']} (J={e['j_height']:.3f}, slope={e['st_slope']:.4f}, curvature={e['curvature']:.4f})"
                for e in informative_leads
            ]
        )
        if is_detected or in_gray_zone:
            morphology_note = f"Highlighted V1-V3 segments indicate morphology around J-point/ST regions, most notable in {lead_summary}."
        else:
            morphology_note = (
                f"V1-V3 morphology windows were analyzed (most notable in {lead_summary}); "
                "these features alone are not sufficient for a positive Brugada flag."
            )
    elif weak_leads:
        morphology_note = (
            "V1-V3 morphology windows were detected, but current handcrafted morphology strength is weak; "
            "decision should rely on full model output with manual cardiology review."
        )
    else:
        morphology_note = "No robust V1-V3 morphological segment was detected for explainable highlight in this record."

    if in_gray_zone:
        triage_note = "Prediction is in the gray zone; cardiologist review is strongly recommended."
    elif is_detected:
        triage_note = "Model probability crosses the clinical decision threshold and is flagged as high-risk triage."
    else:
        triage_note = "Model probability is below the clinical decision threshold and is flagged as low-risk triage."

    return f"Brugada risk probability={probability:.4f}. {triage_note} {morphology_note} This AI output supports triage and does not replace physician diagnosis."

def load_all_models():
    """Load all .keras and .pkl files centrally"""
    if MODELS: return 
    
    print("Initializing Core Models...")
    custom_objs = {'LeadSpatialAttention': LeadSpatialAttention}
    
    MODELS['resnet']  = keras.models.load_model('extractor_resnet.keras', custom_objects=custom_objs)
    MODELS['blstm']   = keras.models.load_model('extractor_bilstm.keras', custom_objects=custom_objs)
    MODELS['eegnet']  = keras.models.load_model('extractor_eegnet.keras', custom_objects=custom_objs)
    MODELS['cwt_cnn'] = keras.models.load_model('extractor_cwt_cnn.keras', custom_objects=custom_objs)

    # Build intermediate feature models (32-d embeddings) instead of final 1-d classifiers.
    MODELS['resnet_feat'] = keras.Model(
        inputs=MODELS['resnet'].input,
        outputs=MODELS['resnet'].get_layer(FEATURE_LAYER_BY_MODEL['resnet']).output,
    )
    MODELS['blstm_feat'] = keras.Model(
        inputs=MODELS['blstm'].input,
        outputs=MODELS['blstm'].get_layer(FEATURE_LAYER_BY_MODEL['blstm']).output,
    )
    MODELS['eegnet_feat'] = keras.Model(
        inputs=MODELS['eegnet'].input,
        outputs=MODELS['eegnet'].get_layer(FEATURE_LAYER_BY_MODEL['eegnet']).output,
    )
    MODELS['cwt_feat'] = keras.Model(
        inputs=MODELS['cwt_cnn'].input,
        outputs=MODELS['cwt_cnn'].get_layer(FEATURE_LAYER_BY_MODEL['cwt_cnn']).output,
    )
    
    MODELS['scaler']   = joblib.load('brugada_scaler.pkl')
    MODELS['selector'] = joblib.load('brugada_selector.pkl')
    MODELS['meta']     = joblib.load('brugada_meta_learner.pkl')
    print("All models loaded successfully!")

# =============================================================================
# Feature Engineering Functions (Exactly matched with Notebook)
# =============================================================================
def extract_clinical_features(sig_2d: np.ndarray, fs: float) -> np.ndarray:
    """Extract exactly 84D statistical + 9D expert features (Total: 93D)"""
    clinical_feat, _, _ = extract_clinical_package(sig_2d, fs)
    return clinical_feat

def generate_cwt_scalograms(sig_2d: np.ndarray) -> np.ndarray:
    """Extract V1, V2, V3 wavelet features and resize to 128x128"""
    leads = [6, 7, 8]
    target_size = (128, 128)
    X_cwt = np.zeros((target_size[0], target_size[1], len(leads)))
    scales = np.arange(1, 129)
    
    for c, lead_idx in enumerate(leads):
        coeffs, _ = pywt.cwt(sig_2d[:, lead_idx], scales, 'cmor1.5-1.0')
        resized_mag = cv2.resize(np.abs(coeffs), target_size, interpolation=cv2.INTER_AREA)
        X_cwt[:, :, c] = resized_mag
        
    return np.expand_dims(X_cwt, axis=0) # shape (1, 128, 128, 3)

# =============================================================================
# Core Inference Logic
# =============================================================================
def preprocess_signal(record_path: str) -> tuple[np.ndarray, float]:
    """Read WFDB, apply 3rd-order bandpass, and pad/truncate"""
    record = wfdb.rdrecord(str(record_path))
    raw_signal = record.p_signal
    fs = record.fs 
    
    # 1. Bandpass filter (0.5 - 40 Hz, Order 3) exactly as in training phase
    nyquist = 0.5 * fs
    b, a = signal.butter(3, [0.5 / nyquist, 40.0 / nyquist], btype='band')
    clean_signal = signal.filtfilt(b, a, raw_signal, axis=0)
    
    # 2. Truncate or Zero-Pad to target length (1200)
    TARGET_LENGTH = 1200
    curr_len = clean_signal.shape[0]
    
    if curr_len >= TARGET_LENGTH:
        standardized_signal = clean_signal[:TARGET_LENGTH, :]
    else:
        pad_len = TARGET_LENGTH - curr_len
        standardized_signal = np.pad(clean_signal, ((0, pad_len), (0, 0)), mode='constant')
        
    return standardized_signal, fs

def predict_from_record(record_path: str) -> dict:
    """
    The Core Multi-View Stacking Pipeline
    """
    load_all_models()
    
    # 1. Preprocessing
    base_signal, fs = preprocess_signal(record_path)
    
    # Format for Keras models
    signal_1d = np.expand_dims(base_signal, axis=0).astype(np.float32)
    signal_2d = generate_cwt_scalograms(base_signal).astype(np.float32)
    
    # 2. Multi-view Feature Extraction
    feat_resnet = MODELS['resnet_feat'].predict(signal_1d, verbose=0)
    feat_eegnet = MODELS['eegnet_feat'].predict(signal_1d, verbose=0)
    feat_blstm  = MODELS['blstm_feat'].predict(signal_1d, verbose=0)
    feat_cwt    = MODELS['cwt_feat'].predict(signal_2d, verbose=0)

    # Embedding-norm share offers a lightweight view of which deep view dominates this sample.
    emb_norms = {
        "resnet": float(np.linalg.norm(feat_resnet)),
        "eegnet": float(np.linalg.norm(feat_eegnet)),
        "blstm": float(np.linalg.norm(feat_blstm)),
        "cwt_cnn": float(np.linalg.norm(feat_cwt)),
    }
    total_norm = sum(emb_norms.values()) + 1e-8
    model_contributions = {k: float(v / total_norm * 100.0) for k, v in emb_norms.items()}
    
    # 3. Clinical Handcrafted Features (Statistical + Expert)
    clinical_feat, highlighted_segments, evidence = extract_clinical_package(base_signal, fs=fs)
    feat_clinical = np.expand_dims(clinical_feat, axis=0)
    
    # 4. Strict Concatenation Order:
    # Stat(84) + Expert(9) -> ResNet(32) -> EEGNet(32) -> BiLSTM(32) -> CWT(32)
    final_features = np.concatenate([
        feat_clinical, 
        feat_resnet, 
        feat_eegnet, 
        feat_blstm, 
        feat_cwt
    ], axis=1) # Expected shape: (1, 221)

    expected_n = getattr(MODELS['scaler'], 'n_features_in_', None)
    if expected_n is not None and final_features.shape[1] != int(expected_n):
        raise ValueError(
            f"Feature dimension mismatch: got {final_features.shape[1]}, expected {int(expected_n)}"
        )
    
    # 5. ML Dimensionality Reduction & Prediction
    scaled_features   = MODELS['scaler'].transform(final_features)
    selected_features = MODELS['selector'].transform(scaled_features)
    
    probabilities = MODELS['meta'].predict_proba(selected_features)[0]
    brugada_proba = probabilities[1] 
    
    # 6. Logical Decision (strictly aligned to notebook thresholding style)
    is_detected = brugada_proba >= DECISION_THRESHOLD
    # Asymmetric gray-zone for low-threshold deployment:
    # only flag borderline positives as gray-zone to avoid overwhelming low-risk normals.
    in_gray_zone = DECISION_THRESHOLD <= brugada_proba <= UPPER_BOUND

    # Two confidence views for clearer interpretation.
    # 1) Class confidence: confidence of the predicted class.
    class_confidence = float(brugada_proba if is_detected else (1.0 - brugada_proba))

    # 2) Threshold stability: how far the sample is from the decision boundary.
    # 50% near threshold, approaching 100% farther away.
    max_margin = max(DECISION_THRESHOLD, 1.0 - DECISION_THRESHOLD)
    decision_margin = abs(brugada_proba - DECISION_THRESHOLD)
    decision_stability = float(0.5 + 0.5 * np.clip(decision_margin / max_margin, 0.0, 1.0))
    confidence_percent = class_confidence * 100.0
    stability_percent = decision_stability * 100.0
    explanation = _build_explanation(brugada_proba, is_detected, in_gray_zone, evidence)
    clinician_explain = _build_clinician_explain(brugada_proba, is_detected, in_gray_zone, evidence)
    
    return {
        "status": "success",
        "label": "Brugada Syndrome Detected" if is_detected else "Normal ECG Pattern",
        "risk": "High" if is_detected else "Low",
        "probability": float(brugada_proba),
        "confidence": float(confidence_percent),
        "decision_stability": float(stability_percent),
        "decision_threshold": float(DECISION_THRESHOLD),
        "gray_zone": bool(in_gray_zone),
        "highlighted_segments": highlighted_segments,
        "lead_names": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "explanation": explanation,
        "clinical_evidence": evidence,
        "model_contributions": model_contributions,
        "clinician_explain": clinician_explain,
        "signal_for_plot": base_signal,
        "fs": fs
    }
