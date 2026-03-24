import os
import time
import streamlit as st
from typing import Optional
import google.genai as genai


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
    def _get_val(ml_result, key, default=0.0):
        """Safely extract value from dict or object."""
        if isinstance(ml_result, dict):
            return ml_result.get(key, default)
        return getattr(ml_result, key, default)

    def get_advice(self, ml_result: dict) -> str:
        """Generate initial clinical interpretation with aggressive caching to save quota."""
        if ml_result is None:
            return "Please run a diagnosis first to get AI advice."

        prob = self._get_val(ml_result, "probability")
        label = self._get_val(ml_result, "label", "Unknown")

        # Create cache key with wider ranges to increase cache hit rate
        # Round probability to 0.01 to group similar results together
        prob_rounded = round(prob, 2)
        cache_key = f"{label}_{prob_rounded}"
        
        # Return cached result if available (save quotas!)
        if cache_key in self._advice_cache:
            cached = self._advice_cache[cache_key]
            print(f"[Cache Hit] Reusing cached advice for {label} (prob ~{prob_rounded})")
            return cached

        prompt = (
            f"Clinical case: ML model classified this ECG as '{label}' "
            f"with probability {prob:.4f}.\n"
            f"Provide a structured clinical interpretation broken perfectly into these three exact Markdown headings:\n"
            f"### Interpretation\n"
            f"### Key Considerations\n"
            f"### Recommended Next Steps\n\n"
            f"Keep paragraphs very concise, use bullet points, and avoid any additional introductory or concluding text."
        )

        advice = self._send_with_retry(prompt, ml_result)
        
        # Cache the result
        self._advice_cache[cache_key] = advice
        print(f"[Cached] Stored advice for {label} (prob ~{prob_rounded}) for future use")
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
                is_quota_error = any(
                    x in error_str
                    for x in ["429", "quota", "rate limit", "resource", "too many", "resource_exhausted"]
                )

                # For quota errors, retry with exponential backoff (longer wait times)
                if is_quota_error and attempt < retries - 1:
                    wait = min(2 ** (attempt + 2), 60)  # Increased wait: 4s, 8s, 16s, 32s, 60s...
                    msg = f"Quota limit hit. Waiting {wait}s before retry ({attempt + 1}/{retries})..."
                    print(msg)
                    st.toast(msg) # Streamlit toast to show retrying without blocking
                    time.sleep(wait)
                    continue
                elif is_quota_error and ml_result:
                    # After all retries, use offline fallback
                    return self._offline_fallback(ml_result)
                elif is_quota_error:
                    return (
                        "**API Quota Exceeded**\n\n"
                        "The system is rate-limited. Please try again in a few minutes. "
                        "Run a diagnosis for offline guidance."
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

    def _offline_fallback(self, ml_result: dict) -> str:
        """Provide offline clinical guidance when API quota is exceeded."""
        prob = self._get_val(ml_result, "probability")
        label = self._get_val(ml_result, "label", "Unknown")

        return (
            f"### AI Advisor - Quota Exceeded\n\n"
            f"The Gemini API has hit its rate limit. The system will resume "
            f"providing AI guidance shortly.\n\n"
            f"**Your ECG Analysis:**\n"
            f"- **Result:** {label}\n"
            f"- **ML Probability:** {prob:.4f}\n\n"
            f"**Interim Clinical Protocol:**\n"
            f"1. Manually review ECG leads V1-V3 for Brugada-type ST elevation pattern\n"
            f"2. **If probability > 0.05:** Escalate for urgent cardiology review\n"
            f"3. Correlate with patient history:\n"
            f"   - Syncope or palpitations?\n"
            f"   - Family history of sudden death?\n"
            f"   - Known arrhythmias?\n"
            f"4. Follow your institution's Brugada syndrome triage pathway\n\n"
            f"**Next Steps:**\n"
            f"- The AI advisor will resume service shortly (typically within 1-2 minutes)\n"
            f"- Click \"Refresh\" or reload the page to try the AI advisor again"
        )