import os
import time
import streamlit as st
from typing import Optional

try:
    import google.genai as genai
    _GENAI_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    genai = None
    _GENAI_IMPORT_ERROR = exc


class BrugadaChatbot:
    """Conversational AI assistant for Brugada syndrome clinical advice"""

    # Models prioritized by speed & quota efficiency (fastest first)
    MODEL_CANDIDATES = [
        "gemini-2.5-flash",         # Fastest, lowest quota
        "gemini-2.0-flash",         # Alternative
        "gemini-2.0-flash-001",     # Fallback
    ]

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with robust API key and model selection."""
        if genai is None:
            raise ValueError(
                "google-genai is not installed. Install it with: "
                ".\\.venv\\Scripts\\python.exe -m pip install google-genai"
            ) from _GENAI_IMPORT_ERROR

        # 1. API Key Retrieval
        if api_key is None:
            api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please add it to .streamlit/secrets.toml"
            )

        # Configure the SDK with new google.genai
        self.client = genai.Client(api_key=api_key)
        self.api_key = api_key

        # System prompt for clinical guidance
        self.system_prompt = (
            "You are a clinical AI assistant specializing in Brugada syndrome. "
            "Provide evidence-based decision support based on ECG analysis. "
            "DO NOT include general medical disclaimers in your responses, as they are already prominently displayed in the UI."
        )

        # Try to initialize with best available model
        self.model = self._select_available_model()
        if not self.model:
            # Fallback to first candidate
            self.model = self.MODEL_CANDIDATES[0]
            print(f"Using fallback model: {self.model}")

        self.chat_history = []
        # Cache for ML result interpretations to avoid repeated API calls
        self._advice_cache = {}

    def _select_available_model(self) -> Optional[str]:
        """Select the first available model without test API calls.
        
        NOTE: We do NOT make test API calls here to avoid consuming quota during
        initialization. The first actual user request will validate the model.
        """
        if self.MODEL_CANDIDATES:
            model = self.MODEL_CANDIDATES[0]
            print(f"✓ Using model: {model} (will validate on first use)")
            return model
        return None

    def reset_conversation(self) -> None:
        """Reset chat history for new patient."""
        self.chat_history = []

    @staticmethod
    def _is_quota_error(error_str: str) -> bool:
        return any(
            token in error_str
            for token in [
                "429",
                "quota",
                "rate limit",
                "resource_exhausted",
                "too many requests",
            ]
        )

    @staticmethod
    def _is_transient_error(error_str: str) -> bool:
        return any(
            token in error_str
            for token in [
                "503",
                "unavailable",
                "high demand",
                "temporarily",
                "deadline exceeded",
                "backend error",
                "overloaded",
            ]
        )

    def _switch_to_next_model(self) -> bool:
        """Rotate to the next fallback model when current model is temporarily unavailable."""
        if not self.MODEL_CANDIDATES:
            return False
        if self.model not in self.MODEL_CANDIDATES:
            self.model = self.MODEL_CANDIDATES[0]
            return True

        if len(self.MODEL_CANDIDATES) == 1:
            return False

        idx = self.MODEL_CANDIDATES.index(self.model)
        next_model = self.MODEL_CANDIDATES[(idx + 1) % len(self.MODEL_CANDIDATES)]
        if next_model == self.model:
            return False

        self.model = next_model
        return True

    @staticmethod
    def _get_val(ml_result, key, default=0.0):
        """Safely extract value from dict or object."""
        if isinstance(ml_result, dict):
            return ml_result.get(key, default)
        return getattr(ml_result, key, default)

    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        try:
            return int(float(value))
        except Exception:
            return default

    @staticmethod
    def _to_dict(value) -> dict:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _dominant_evidence_tier(clinical_evidence) -> str:
        if not isinstance(clinical_evidence, list):
            return "unknown"

        tier_rank = {"weak": 1, "moderate": 2, "strong": 3}
        dominant = "unknown"
        dominant_rank = 0
        for item in clinical_evidence:
            if not isinstance(item, dict):
                continue
            tier = str(item.get("tier", "")).strip().lower()
            rank = tier_rank.get(tier, 0)
            if rank > dominant_rank:
                dominant = tier
                dominant_rank = rank

        return dominant

    def get_advice(self, ml_result: dict) -> str:
        """Generate initial clinical interpretation with aggressive caching to save quota."""
        if ml_result is None:
            return "Please run a diagnosis first to get AI advice."

        prob = float(self._get_val(ml_result, "display_probability", self._get_val(ml_result, "probability", 0.0)))
        threshold = float(self._get_val(ml_result, "display_threshold", 0.35))
        label = str(self._get_val(ml_result, "label", "Unknown"))
        gray_zone = bool(self._get_val(ml_result, "gray_zone", False))

        clinician_explain = self._to_dict(self._get_val(ml_result, "clinician_explain", {}))
        recommendation_tier = str(clinician_explain.get("recommendation_tier", "unknown")).strip().lower()
        recommendation_tier = recommendation_tier.replace(" ", "_") if recommendation_tier else "unknown"

        evidence_counts = self._to_dict(clinician_explain.get("evidence_counts", {}))
        strong_count = self._safe_int(evidence_counts.get("strong", 0))
        moderate_count = self._safe_int(evidence_counts.get("moderate", 0))
        weak_count = self._safe_int(evidence_counts.get("weak", 0))

        clinical_evidence = self._get_val(ml_result, "clinical_evidence", [])
        dominant_evidence_tier = self._dominant_evidence_tier(clinical_evidence)
        evidence_signature = f"s{strong_count}_m{moderate_count}_w{weak_count}"

        prob_rounded = round(prob, 2)
        label_key = label.strip().lower().replace(" ", "_")
        cache_key = (
            f"{label_key}_{prob_rounded:.2f}"
            f"_gz{int(gray_zone)}"
            f"_{recommendation_tier}"
            f"_{dominant_evidence_tier}"
            f"_{evidence_signature}"
        )
        
        # Return cached result if available (save quotas!)
        if cache_key in self._advice_cache:
            cached = self._advice_cache[cache_key]
            print(f"[Cache Hit] Reusing cached advice for key={cache_key}")
            return cached

        prompt = (
            f"Clinical case: ML model classified this ECG as '{label}' "
            f"with probability {prob:.4f}.\n"
            f"Use decision threshold {threshold:.2f} when describing risk relative to boundary.\n"
            f"Gray-zone flag: {'yes' if gray_zone else 'no'}.\n"
            f"Recommendation tier: {recommendation_tier}.\n"
            f"Dominant morphology evidence tier: {dominant_evidence_tier}.\n"
            f"Evidence counts (strong/moderate/weak): {strong_count}/{moderate_count}/{weak_count}.\n"
            f"Provide a structured clinical interpretation broken perfectly into these three exact Markdown headings:\n"
            f"### Interpretation\n"
            f"### Key Considerations\n"
            f"### Recommended Next Steps\n\n"
            f"Keep paragraphs very concise, use bullet points, and avoid any additional introductory or concluding text."
        )

        advice = self._send_with_retry(prompt, ml_result)

        # Cache only successful model outputs, not transient errors/fallback notices.
        non_cache_prefixes = (
            "**advisor error",
            "**api quota exceeded",
            "**ai advisor temporarily unavailable",
            "### ai advisor -",
        )
        if advice and not advice.strip().lower().startswith(non_cache_prefixes):
            self._advice_cache[cache_key] = advice
            print(f"[Cached] Stored advice for key={cache_key}")
        return advice

    def continue_conversation(self, user_message: str) -> str:
        """Handle follow-up questions with concise responses."""
        if not user_message.strip():
            return "Please enter a question."
        # Prepend conciseness instruction
        concise_message = f"Answer this briefly in 2-3 sentences: {user_message}"
        return self._send_with_retry(concise_message, None)

    def _send_with_retry(
        self, message: str, ml_result: Optional[dict] = None, retries: int = 8
    ) -> str:
        """Send message with aggressive quota-aware retry logic."""
        for attempt in range(retries):
            try:
                # Prepend system prompt to the message for google-genai SDK
                full_message = f"{self.system_prompt}\n\n{message}"
                
                # Call the API with correct format
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=full_message,  # Just pass the text string
                )

                return response.text

            except Exception as e:
                error_str = str(e).lower()
                is_quota_error = self._is_quota_error(error_str)
                is_transient_error = self._is_transient_error(error_str)

                # Retry temporary failures with backoff and model fallback.
                if (is_quota_error or is_transient_error) and attempt < retries - 1:
                    switched = self._switch_to_next_model()
                    wait = min(2 ** (attempt + 1), 60 if is_quota_error else 20)
                    reason = "Quota/rate limit" if is_quota_error else "Model temporarily busy"
                    switch_note = f" Switched to {self.model}." if switched else ""
                    msg = f"{reason}. Retrying in {wait}s ({attempt + 1}/{retries}).{switch_note}"
                    print(msg)
                    st.toast(msg)
                    time.sleep(wait)
                    continue

                if is_quota_error and ml_result:
                    # After retries, use offline fallback for patient safety continuity.
                    return self._offline_fallback(ml_result, reason="quota")
                elif is_quota_error:
                    return (
                        "**API Quota Exceeded**\n\n"
                        "The system is rate-limited. Please try again in a few minutes. "
                        "Run a diagnosis for offline guidance."
                    )

                if is_transient_error and ml_result:
                    return self._offline_fallback(ml_result, reason="busy")
                elif is_transient_error:
                    return (
                        "**AI Advisor Temporarily Unavailable (503)**\n\n"
                        "The model endpoint is currently busy (high demand). "
                        "Please try again shortly."
                    )

                # For 404 model not found, that's a configuration issue
                if "404" in str(e) and "not found" in error_str:
                    return (
                        f"**Model Configuration Error:** {str(e)[:80]}\n\n"
                        f"The AI model is not available. "
                        f"Please check at console.cloud.google.com that your API key has access "
                        f"to the Generative AI models."
                    )

                # Other errors
                return (
                    f"**Advisor Error:** {str(e)[:100]}\n\n"
                    f"Try these steps:\n"
                    f"1. Verify GEMINI_API_KEY in `.streamlit/secrets.toml`\n"
                    f"2. Check API quota at console.cloud.google.com\n"
                    f"3. Run: `pip install -U google-genai`"
                )

        return "Failed after retries. Please try again in a few minutes."

    def _offline_fallback(self, ml_result: dict, reason: str = "quota") -> str:
        """Provide offline clinical guidance when API service is unavailable."""
        prob = float(self._get_val(ml_result, "display_probability", self._get_val(ml_result, "probability", 0.0)))
        threshold = float(self._get_val(ml_result, "display_threshold", 0.35))
        label = self._get_val(ml_result, "label", "Unknown")

        if reason == "busy":
            title = "### AI Advisor - Service Busy"
            intro = "The Gemini API is temporarily unavailable due to high demand."
        else:
            title = "### AI Advisor - Quota Exceeded"
            intro = "The Gemini API has hit its rate limit."

        return (
            f"{title}\n\n"
            f"{intro} The system will resume "
            f"providing AI guidance shortly.\n\n"
            f"**Your ECG Analysis:**\n"
            f"- **Result:** {label}\n"
            f"- **ML Probability:** {prob:.4f}\n\n"
            f"**Interim Clinical Protocol:**\n"
            f"1. Manually review ECG leads V1-V3 for Brugada-type ST elevation pattern\n"
            f"2. **If probability > {threshold:.2f}:** Escalate for urgent cardiology review\n"
            f"3. Correlate with patient history:\n"
            f"   - Syncope or palpitations?\n"
            f"   - Family history of sudden death?\n"
            f"   - Known arrhythmias?\n"
            f"4. Follow your institution's Brugada syndrome triage pathway\n\n"
            f"**Next Steps:**\n"
            f"- The AI advisor will resume service shortly (typically within 1-2 minutes)\n"
            f"- Click \"Refresh\" or reload the page to try the AI advisor again"
        )