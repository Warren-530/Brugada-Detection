import os
from pathlib import Path

import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Layer, Reshape

# Global model cache
MODELS = {}
APP_ROOT = Path(__file__).resolve().parents[2]

FEATURE_LAYER_BY_MODEL = {
    "resnet": "nature_resnet_feature",
    "blstm": "bilstm_feature",
    "eegnet": "eegnet_feature",
    "cwt_cnn": "cwt_feature",
}

# Notebook-aligned deployment thresholds.
DECISION_THRESHOLD = 0.050
UPPER_BOUND = 0.060
DISPLAY_THRESHOLD = 0.350

REQUIRED_MODEL_KEYS = {
    "resnet",
    "blstm",
    "eegnet",
    "cwt_cnn",
    "resnet_feat",
    "blstm_feat",
    "eegnet_feat",
    "cwt_feat",
    "scaler",
    "selector",
    "meta",
}
REQUIRED_SKLEARN_VERSION = "1.6.1"


def _missing_required_model_keys() -> list[str]:
    return sorted(REQUIRED_MODEL_KEYS.difference(MODELS.keys()))


def _ensure_sklearn_compatibility() -> None:
    try:
        import sklearn
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "scikit-learn is required for classifier artifacts. "
            f"Please install scikit-learn=={REQUIRED_SKLEARN_VERSION}."
        ) from exc

    sklearn_version = getattr(sklearn, "__version__", "unknown")
    if sklearn_version != REQUIRED_SKLEARN_VERSION:
        raise RuntimeError(
            "Incompatible scikit-learn version detected "
            f"({sklearn_version}). This deployment requires "
            f"scikit-learn=={REQUIRED_SKLEARN_VERSION} for model artifact compatibility."
        )


@keras.utils.register_keras_serializable()
class LeadSpatialAttention(Layer):
    def __init__(self, **kwargs):
        super(LeadSpatialAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_leads = input_shape[-1]
        self.dense1 = Dense(self.num_leads // 2, activation="relu")
        self.dense2 = Dense(self.num_leads, activation="sigmoid")
        super(LeadSpatialAttention, self).build(input_shape)

    def call(self, inputs):
        squeeze = tf.reduce_mean(inputs, axis=1)
        excitation = self.dense1(squeeze)
        attention_weights = self.dense2(excitation)
        attention_weights = Reshape((1, self.num_leads))(attention_weights)
        return inputs * attention_weights


def load_all_models():
    """Load all .keras and .pkl files centrally."""
    missing_keys = _missing_required_model_keys()
    if not missing_keys:
        return

    if MODELS:
        print(f"[WARN] Partial model cache detected; reloading missing keys: {', '.join(missing_keys)}")
        MODELS.clear()

    _ensure_sklearn_compatibility()

    print("Initializing Core Models...")
    custom_objs = {"LeadSpatialAttention": LeadSpatialAttention}

    model_files = {
        "resnet": str(APP_ROOT / "models" / "extractor_resnet.keras"),
        "blstm": str(APP_ROOT / "models" / "extractor_bilstm.keras"),
        "eegnet": str(APP_ROOT / "models" / "extractor_eegnet.keras"),
        "cwt_cnn": str(APP_ROOT / "models" / "extractor_cwt_cnn.keras"),
    }

    classifier_files = {
        "scaler": APP_ROOT / "models" / "brugada_scaler.pkl",
        "selector": APP_ROOT / "models" / "brugada_selector.pkl",
        "meta": APP_ROOT / "models" / "brugada_meta_learner.pkl",
    }

    try:
        for model_key, model_file in model_files.items():
            print(f"  Loading {model_file}...")
            MODELS[model_key] = keras.models.load_model(model_file, custom_objects=custom_objs)
            print(f"    [OK] {model_file} loaded successfully")

        print("  Building feature extraction models...")
        MODELS["resnet_feat"] = keras.Model(
            inputs=MODELS["resnet"].input,
            outputs=MODELS["resnet"].get_layer(FEATURE_LAYER_BY_MODEL["resnet"]).output,
        )
        MODELS["blstm_feat"] = keras.Model(
            inputs=MODELS["blstm"].input,
            outputs=MODELS["blstm"].get_layer(FEATURE_LAYER_BY_MODEL["blstm"]).output,
        )
        MODELS["eegnet_feat"] = keras.Model(
            inputs=MODELS["eegnet"].input,
            outputs=MODELS["eegnet"].get_layer(FEATURE_LAYER_BY_MODEL["eegnet"]).output,
        )
        MODELS["cwt_feat"] = keras.Model(
            inputs=MODELS["cwt_cnn"].input,
            outputs=MODELS["cwt_cnn"].get_layer(FEATURE_LAYER_BY_MODEL["cwt_cnn"]).output,
        )
        print("    [OK] Feature models built successfully")

        print("  Loading classifier models...")
        for key, artifact_path in classifier_files.items():
            if not artifact_path.exists():
                raise FileNotFoundError(f"Missing classifier artifact: {artifact_path}")
            MODELS[key] = joblib.load(artifact_path)
        print("    [OK] Scaler, selector, and meta-learner loaded successfully")

        missing_after_load = _missing_required_model_keys()
        if missing_after_load:
            raise RuntimeError(
                "Model initialization incomplete. Missing keys after load: "
                f"{', '.join(missing_after_load)}"
            )
    except Exception as e:
        MODELS.clear()
        print(f"    [ERROR] Model initialization failed: {str(e)}")
        if isinstance(e, FileNotFoundError):
            raise RuntimeError(
                "Classifier artifact missing. Ensure these files exist in models/: "
                "brugada_scaler.pkl, brugada_selector.pkl, brugada_meta_learner.pkl"
            ) from e
        if "sklearn" in str(e).lower() or "scikit-learn" in str(e).lower():
            raise RuntimeError(
                "Classifier artifact load failed due to scikit-learn compatibility. "
                f"Install scikit-learn=={REQUIRED_SKLEARN_VERSION} and restart the app."
            ) from e
        if any(token in str(e).lower() for token in ["keras", "tensorflow", "layer", "model"]):
            raise RuntimeError(
                "Feature extractor loading failed. Reinstall dependencies from requirements.txt and restart the app."
            ) from e
        raise

    print("[OK] All models initialized successfully!")
