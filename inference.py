from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import keras
import numpy as np
import tensorflow as tf
import wfdb


MODEL_FILES = {
    "extractor_bilstm": "extractor_bilstm.keras",
    "extractor_cwt_cnn": "extractor_cwt_cnn.keras",
    "extractor_eegnet": "extractor_eegnet.keras",
    "extractor_resnet": "extractor_resnet.keras",
}

FEATURE_LAYER_BY_MODEL = {
    "extractor_resnet": "nature_resnet_feature",
    "extractor_bilstm": "bilstm_feature",
    "extractor_eegnet": "eegnet_feature",
    "extractor_cwt_cnn": "cwt_feature",
}

DEEP_FEATURE_ORDER = [
    "extractor_resnet",
    "extractor_bilstm",
    "extractor_eegnet",
    "extractor_cwt_cnn",
]

ARTIFACT_FILES = {
    "scaler": "brugada_scaler.pkl",
    "selector": "brugada_selector.pkl",
    "meta": "brugada_meta_learner.pkl",
}

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


@keras.utils.register_keras_serializable(package="Custom")
class LeadSpatialAttention(keras.layers.Layer):
    """Lightweight lead-wise attention used by saved extractor models."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # inputs: (batch, time, leads) -> soft attention over lead axis
        scores = tf.reduce_mean(inputs, axis=1, keepdims=True)
        weights = tf.nn.softmax(scores, axis=-1)
        return inputs * weights


CUSTOM_OBJECTS = {
    "LeadSpatialAttention": LeadSpatialAttention,
}


@dataclass
class PredictionResult:
    label: str
    probability: float
    confidence_percent: float
    explanation: str
    lead_names: list[str]
    signal: np.ndarray
    fs: float
    highlighted_segments: dict[str, list[tuple[int, int]]]
    feature_importance: dict[str, float]


def _project_root() -> Path:
    return Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def load_artifacts() -> dict[str, Any]:
    root = _project_root()

    models: dict[str, keras.Model] = {}
    for model_key, filename in MODEL_FILES.items():
        model_path = root / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")
        models[model_key] = keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=CUSTOM_OBJECTS,
        )

    feature_models: dict[str, keras.Model] = {}
    for model_name, model in models.items():
        layer_name = FEATURE_LAYER_BY_MODEL[model_name]
        feature_models[model_name] = keras.Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output,
        )

    artifacts = {}
    for artifact_key, filename in ARTIFACT_FILES.items():
        file_path = root / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing artifact file: {file_path}")
        artifacts[artifact_key] = joblib.load(file_path)

    return {
        "models": models,
        "feature_models": feature_models,
        "scaler": artifacts["scaler"],
        "selector": artifacts["selector"],
        "meta": artifacts["meta"],
    }


def read_wfdb_record(record_base_path: str | Path) -> tuple[np.ndarray, float, list[str]]:
    record = wfdb.rdrecord(str(record_base_path))
    if record.p_signal is None:
        raise ValueError("WFDB record has no p_signal data")

    signal = np.asarray(record.p_signal, dtype=np.float32)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    if signal.ndim != 2:
        raise ValueError(f"Expected 2D signal array, got shape={signal.shape}")

    lead_names = list(record.sig_name) if record.sig_name else []
    fs = float(record.fs) if record.fs else 500.0
    return signal, fs, lead_names


def _ensure_12_leads(signal: np.ndarray, target_leads: int = 12) -> np.ndarray:
    n_samples, n_leads = signal.shape
    if n_leads == target_leads:
        return signal
    if n_leads > target_leads:
        return signal[:, :target_leads]

    pad = np.zeros((n_samples, target_leads - n_leads), dtype=np.float32)
    return np.hstack([signal, pad])


def _fix_length(signal: np.ndarray, target_len: int = 5000) -> np.ndarray:
    n_samples = signal.shape[0]
    if n_samples == target_len:
        return signal
    if n_samples > target_len:
        return signal[:target_len, :]

    pad = np.zeros((target_len - n_samples, signal.shape[1]), dtype=np.float32)
    return np.vstack([signal, pad])


def preprocess_signal(signal: np.ndarray) -> np.ndarray:
    sig = np.asarray(signal, dtype=np.float32)
    sig = _ensure_12_leads(sig, target_leads=12)

    mean = np.mean(sig, axis=0, keepdims=True)
    std = np.std(sig, axis=0, keepdims=True) + 1e-6
    sig = (sig - mean) / std

    return sig


def _resize_2d(signal: np.ndarray, target_time: int, target_leads: int) -> np.ndarray:
    """Resize a (time, leads) matrix to target shape using linear interpolation."""
    src_t, src_l = signal.shape

    if src_t != target_time:
        old_t = np.linspace(0.0, 1.0, src_t, dtype=np.float32)
        new_t = np.linspace(0.0, 1.0, target_time, dtype=np.float32)
        time_resized = np.empty((target_time, src_l), dtype=np.float32)
        for j in range(src_l):
            time_resized[:, j] = np.interp(new_t, old_t, signal[:, j]).astype(np.float32)
    else:
        time_resized = signal.astype(np.float32, copy=False)

    if src_l != target_leads:
        old_l = np.linspace(0.0, 1.0, src_l, dtype=np.float32)
        new_l = np.linspace(0.0, 1.0, target_leads, dtype=np.float32)
        lead_resized = np.empty((target_time, target_leads), dtype=np.float32)
        for i in range(target_time):
            lead_resized[i, :] = np.interp(new_l, old_l, time_resized[i, :]).astype(np.float32)
        return lead_resized

    return time_resized


def _build_model_input(processed_signal: np.ndarray, model: keras.Model) -> np.ndarray:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) == 3:
        target_time = int(input_shape[1]) if input_shape[1] is not None else processed_signal.shape[0]
        target_leads = int(input_shape[2]) if input_shape[2] is not None else processed_signal.shape[1]
        resized = _resize_2d(processed_signal, target_time, target_leads)
        return resized[np.newaxis, :, :].astype(np.float32)

    if len(input_shape) == 4:
        target_h = int(input_shape[1]) if input_shape[1] is not None else 128
        target_w = int(input_shape[2]) if input_shape[2] is not None else 128
        target_c = int(input_shape[3]) if input_shape[3] is not None else 1

        base_img = _resize_2d(processed_signal, target_h, target_w)
        base_img = (base_img - np.mean(base_img)) / (np.std(base_img) + 1e-6)

        if target_c == 1:
            img = base_img[:, :, np.newaxis]
        else:
            img = np.repeat(base_img[:, :, np.newaxis], target_c, axis=2)
        return img[np.newaxis, :, :, :].astype(np.float32)

    raise RuntimeError(f"Unsupported model input shape: {input_shape}")


def _infer_feature_vector(model: keras.Model, processed_signal: np.ndarray) -> np.ndarray:
    try:
        model_input = _build_model_input(processed_signal, model)
        output = model.predict(model_input, verbose=0)
        flat = np.ravel(np.asarray(output, dtype=np.float32))
        if flat.size == 0:
            raise RuntimeError("Extractor returned empty output")
        return flat
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Unable to run extractor. Error: {exc}") from exc


def _compute_handcrafted_features(signal_1200x12: np.ndarray, lead_names: list[str]) -> np.ndarray:
    """Compute 93 deterministic handcrafted features to match training contract.

    Layout:
    - 72 lead-wise stats (12 leads x 6 stats)
    - 12 global stats
    - 9 right-precordial stats (V1-V3 x 3 stats)
    """
    sig = _resize_2d(signal_1200x12, target_time=1200, target_leads=12)
    feats: list[float] = []

    for i in range(12):
        x = sig[:, i]
        rms = float(np.sqrt(np.mean(x * x)))
        feats.extend(
            [
                float(np.mean(x)),
                float(np.std(x)),
                float(np.min(x)),
                float(np.max(x)),
                float(np.ptp(x)),
                rms,
            ]
        )

    flat = sig.reshape(-1)
    q05 = float(np.quantile(flat, 0.05))
    q95 = float(np.quantile(flat, 0.95))
    iqr = float(np.quantile(flat, 0.75) - np.quantile(flat, 0.25))
    zc = float(np.mean(np.diff(np.signbit(flat)).astype(np.float32)))
    feats.extend(
        [
            float(np.mean(flat)),
            float(np.std(flat)),
            float(np.min(flat)),
            float(np.max(flat)),
            float(np.median(flat)),
            iqr,
            float(np.mean(np.abs(flat))),
            float(np.std(np.abs(flat))),
            q05,
            q95,
            float(np.mean(flat * flat)),
            zc,
        ]
    )

    lead_idx = {name: idx for idx, name in enumerate(lead_names)}
    for lead in ["V1", "V2", "V3"]:
        idx = lead_idx.get(lead)
        if idx is None or idx >= sig.shape[1]:
            x = np.zeros(sig.shape[0], dtype=np.float32)
        else:
            x = sig[:, idx]
        feats.extend([float(np.mean(x)), float(np.std(x)), float(np.max(x))])

    out = np.asarray(feats, dtype=np.float32)
    if out.size != 93:
        raise RuntimeError(f"Handcrafted feature size mismatch: got {out.size}, expected 93")
    return out


def extract_stacked_features(processed_signal: np.ndarray, lead_names: list[str]) -> tuple[np.ndarray, dict[str, float]]:
    artifacts = load_artifacts()
    models = artifacts["feature_models"]

    feature_chunks = []
    feature_sizes: dict[str, float] = {}

    for model_name in DEEP_FEATURE_ORDER:
        model = models[model_name]
        features = _infer_feature_vector(model, processed_signal)
        feature_chunks.append(features)
        feature_sizes[model_name] = float(features.size)

    handcrafted = _compute_handcrafted_features(processed_signal, lead_names)
    feature_chunks.append(handcrafted)
    feature_sizes["handcrafted_features"] = float(handcrafted.size)

    stacked = np.concatenate(feature_chunks, axis=0).astype(np.float32)

    scaler_n = getattr(artifacts["scaler"], "n_features_in_", None)
    if scaler_n is not None and stacked.size != int(scaler_n):
        raise RuntimeError(
            f"Stacked feature size mismatch: got {stacked.size}, expected {int(scaler_n)}"
        )

    return stacked, feature_sizes


def _continuous_segments(mask: np.ndarray, min_len: int = 20) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    if not np.any(mask):
        return segments

    start = None
    for idx, val in enumerate(mask):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            if idx - start >= min_len:
                segments.append((start, idx))
            start = None

    if start is not None and len(mask) - start >= min_len:
        segments.append((start, len(mask)))
    return segments


def compute_v1_v3_highlights(processed_signal: np.ndarray, lead_names: list[str]) -> dict[str, list[tuple[int, int]]]:
    lead_map = {name: i for i, name in enumerate(lead_names)}
    highlights: dict[str, list[tuple[int, int]]] = {}

    for lead in ["V1", "V2", "V3"]:
        idx = lead_map.get(lead)
        if idx is None:
            continue

        wave = processed_signal[:, idx]
        importance = np.abs(wave)
        thr = np.percentile(importance, 90)
        mask = importance >= thr
        highlights[lead] = _continuous_segments(mask)

    return highlights


def _compose_clinical_text(prob: float, feature_footprint: dict[str, float], highlights: dict[str, list[tuple[int, int]]]) -> str:
    label = "Brugada Pattern" if prob >= 0.5 else "Normal Pattern"
    confidence = prob * 100.0

    dominant_model = max(feature_footprint, key=feature_footprint.get)
    active_leads = [lead for lead in ["V1", "V2", "V3"] if highlights.get(lead)]

    if label == "Brugada Pattern":
        lead_note = ", ".join(active_leads) if active_leads else "right precordial leads"
        return (
            f"Model predicts {label} with {confidence:.1f}% confidence. "
            f"High attention regions detected in {lead_note}, with strongest embedding contribution from {dominant_model}. "
            "Observed morphology may be compatible with Type 1 Brugada ST-segment pattern. "
            "Clinical correlation is required and electrophysiology consultation should be considered."
        )

    return (
        f"Model predicts {label} with {confidence:.1f}% confidence. "
        f"Feature stack is most influenced by {dominant_model}, without sustained high-risk pattern in V1-V3 highlights. "
        "If symptoms, syncope, or family history are present, follow-up ECG review is still recommended."
    )


def predict_from_record(record_base_path: str | Path) -> PredictionResult:
    signal, fs, lead_names = read_wfdb_record(record_base_path)
    if not lead_names:
        lead_names = DEFAULT_LEAD_NAMES[: signal.shape[1]]

    processed = preprocess_signal(signal)
    display_signal = _resize_2d(processed, target_time=1200, target_leads=12)
    if len(lead_names) < display_signal.shape[1]:
        lead_names = lead_names + DEFAULT_LEAD_NAMES[len(lead_names) : display_signal.shape[1]]

    stacked_features, feature_footprint = extract_stacked_features(processed, lead_names)

    artifacts = load_artifacts()
    scaler = artifacts["scaler"]
    selector = artifacts["selector"]
    meta = artifacts["meta"]

    x = stacked_features.reshape(1, -1)
    x_scaled = scaler.transform(x)
    x_selected = selector.transform(x_scaled)

    proba = float(meta.predict_proba(x_selected)[0, 1])
    label = "Brugada Syndrome Detected" if proba >= 0.5 else "Normal Pattern"

    highlights = compute_v1_v3_highlights(display_signal, lead_names)
    explanation = _compose_clinical_text(proba, feature_footprint, highlights)

    return PredictionResult(
        label=label,
        probability=proba,
        confidence_percent=proba * 100.0,
        explanation=explanation,
        lead_names=lead_names[: display_signal.shape[1]],
        signal=display_signal,
        fs=fs,
        highlighted_segments=highlights,
        feature_importance=feature_footprint,
    )


def predict_batch_from_folder(folder_path: str | Path) -> list[dict[str, Any]]:
    folder = Path(folder_path)
    results = []

    for hea_file in sorted(folder.glob("*.hea")):
        base = hea_file.with_suffix("")
        dat_file = base.with_suffix(".dat")
        if not dat_file.exists():
            continue

        try:
            pred = predict_from_record(base)
            results.append(
                {
                    "record": base.name,
                    "label": pred.label,
                    "probability": pred.probability,
                    "risk": "High" if pred.probability >= 0.5 else "Low",
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
