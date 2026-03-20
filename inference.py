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

# Deployment thresholds calibrated on aligned pipeline evaluation.
DECISION_THRESHOLD = 0.40
LOWER_BOUND = 0.35
UPPER_BOUND = 0.45

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
    
    # 1. Statistical View (84D)
    stat_feat = []
    for lead in range(12):
        sig = sig_2d[:, lead]
        stat_feat.extend([
            np.max(sig), 
            np.min(sig), 
            np.std(sig), 
            np.var(sig),
            stats.skew(sig), 
            stats.kurtosis(sig), 
            np.sqrt(np.mean(sig**2)) # RMS
        ])
        
    # 2. Expert Morphological View V1-V3 (9D)
    expert_feat = []
    v_leads = [6, 7, 8] # V1, V2, V3
    for l_idx in v_leads:
        sig = sig_2d[:, l_idx]
        try:
            _, rpeaks = nk.ecg_peaks(sig, sampling_rate=fs)
            _, waves = nk.ecg_delineate(sig, rpeaks, sampling_rate=fs, method="peak")
            j_offsets = [int(x) for x in waves['ECG_R_Offsets'] if not np.isnan(x)]

            if j_offsets:
                j_heights = [sig[j] for j in j_offsets]
                avg_j_height = np.mean(j_heights)

                v20, v40, v60, v80 = [], [], [], []
                for j in j_offsets:
                    if j + 8 < len(sig):
                        v20.append(sig[j+2])
                        v40.append(sig[j+4])
                        v60.append(sig[j+6])
                        v80.append(sig[j+8])

                avg_slope = (np.mean(v80) - avg_j_height) / 8 if v80 else 0
                curvature = (np.mean(v40) - np.mean(v20)) - (np.mean(v80) - np.mean(v60)) if v80 else 0

                expert_feat.extend([avg_j_height, avg_slope, curvature])
            else:
                expert_feat.extend([0, 0, 0])
        except Exception:
            expert_feat.extend([0, 0, 0])
            
    return np.array(stat_feat + expert_feat)

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
    
    # 3. Clinical Handcrafted Features (Statistical + Expert)
    feat_clinical = extract_clinical_features(base_signal, fs=fs)
    feat_clinical = np.expand_dims(feat_clinical, axis=0)
    
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
    in_gray_zone = LOWER_BOUND <= brugada_proba <= UPPER_BOUND
    confidence_percent = brugada_proba * 100.0
    
    return {
        "status": "success",
        "label": "Brugada Syndrome Detected" if is_detected else "Normal ECG Pattern",
        "risk": "High" if is_detected else "Low",
        "probability": float(brugada_proba),
        "confidence": float(confidence_percent),
        "decision_threshold": float(DECISION_THRESHOLD),
        "gray_zone": bool(in_gray_zone),
        "signal_for_plot": base_signal,
        "fs": fs
    }
