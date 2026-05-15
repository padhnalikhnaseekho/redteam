"""D7: Perplexity-based injection detection.

Uses a small local language model (GPT-2) to detect prompt injections by
measuring perplexity discontinuities in the input text. Injected instructions
create distributional anomalies -- the perplexity of injected segments is
significantly different from the surrounding natural language context.

Mathematical basis:
    Perplexity = exp( -(1/N) * sum_{i=1}^{N} log P(token_i | context) )

    For a sliding window of size W across the input:
        PPL_w = perplexity of tokens in window w
        mu = mean(PPL_w for all w)
        sigma = std(PPL_w for all w)
        z_max = (max(PPL_w) - mu) / sigma

    A spike (high z-score) indicates text that is distributionally different
    from the surrounding context -- characteristic of injected instructions
    within otherwise natural commodity trading queries.

    Information-theoretic interpretation:
        High perplexity = high cross-entropy = low probability under the LM
        = text the model finds "surprising" = potentially adversarial content.

References:
    - Jain et al. (2023), "Baseline Defenses for Adversarial Attacks Against
      Aligned Language Models"
    - Alon & Kamfonas (2023), "Detecting Language Model Attacks with
      Perplexity"
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .base import Defense, DefenseResult

logger = logging.getLogger(__name__)


class PerplexityFilterDefense(Defense):
    """Detects prompt injections via perplexity spike analysis.

    Uses GPT-2 (small, local, fast) to compute windowed perplexity
    across input text. Injections cause measurable perplexity spikes
    compared to surrounding commodity market language.

    Attributes:
        name: Defense identifier.
        model_name: HuggingFace model for perplexity computation.
        window_size: Token window size for sliding perplexity.
        stride: Stride between windows.
        spike_threshold: Z-score threshold for spike detection.
        absolute_threshold: Absolute perplexity above which text is flagged.
    """

    name: str = "perplexity_filter"

    def __init__(
        self,
        model_name: str = "gpt2",
        window_size: int = 50,
        stride: int = 25,
        spike_threshold: float = 2.5,
        absolute_threshold: float = 150.0,
    ) -> None:
        self.model_name = model_name
        self.window_size = window_size
        self.stride = stride
        self.spike_threshold = spike_threshold
        self.absolute_threshold = absolute_threshold

        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy-load GPT-2 model and tokenizer."""
        if self._model is None:
            import torch
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast

            logger.info("Loading perplexity model: %s", self.model_name)
            self._tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
            self._model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self._model.eval()

    def _compute_token_log_probs(self, text: str) -> np.ndarray:
        """Compute per-token log probabilities under the language model.

        Returns:
            Array of shape (n_tokens,) with log P(token_i | tokens_<i).
        """
        import torch

        self._load_model()

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = self._model(input_ids, labels=input_ids)
            # Get per-token logits
            logits = outputs.logits  # (1, seq_len, vocab_size)

        # Shift: logits[t] predicts token[t+1]
        shift_logits = logits[:, :-1, :].squeeze(0)  # (seq_len-1, vocab_size)
        shift_labels = input_ids[:, 1:].squeeze(0)    # (seq_len-1,)

        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

        return token_log_probs.numpy()

    def _compute_windowed_perplexity(self, text: str) -> tuple[list[float], list[str]]:
        """Compute perplexity for sliding windows across the text.

        Perplexity of window w:
            PPL_w = exp( -(1/|w|) * sum_{i in w} log P(token_i | context) )

        Returns:
            Tuple of (perplexity_values, window_texts) for each window.
        """
        token_log_probs = self._compute_token_log_probs(text)
        n_tokens = len(token_log_probs)

        if n_tokens < self.window_size:
            # Text too short for windowed analysis -- compute single perplexity
            avg_neg_log_prob = -float(np.mean(token_log_probs))
            ppl = float(np.exp(avg_neg_log_prob))
            return [ppl], [text]

        perplexities = []
        window_texts = []

        # Decode tokens for window text extraction
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        tokens = inputs["input_ids"].squeeze(0).tolist()

        for start in range(0, n_tokens - self.window_size + 1, self.stride):
            end = start + self.window_size
            window_log_probs = token_log_probs[start:end]
            avg_neg_log_prob = -float(np.mean(window_log_probs))
            ppl = float(np.exp(avg_neg_log_prob))
            perplexities.append(ppl)

            # Decode window tokens (+1 offset because log_probs are shifted)
            window_token_ids = tokens[start + 1:end + 1]
            window_text = self._tokenizer.decode(window_token_ids, skip_special_tokens=True)
            window_texts.append(window_text[:100])

        return perplexities, window_texts

    def _detect_spike(self, perplexities: list[float]) -> tuple[bool, float, int]:
        """Detect statistically significant perplexity spikes.

        Uses z-score for spike detection:
            z = (PPL_max - mean(PPL)) / std(PPL)

        A high z-score indicates a window that is distributionally
        different from the rest of the text.

        Returns:
            Tuple of (spike_detected, z_score, spike_window_index).
        """
        if len(perplexities) < 3:
            # Not enough windows for statistical analysis
            max_ppl = max(perplexities) if perplexities else 0
            return max_ppl > self.absolute_threshold, 0.0, 0

        ppl_array = np.array(perplexities)
        mu = float(np.mean(ppl_array))
        sigma = float(np.std(ppl_array))

        if sigma < 1e-6:
            # Constant perplexity -- no spike
            return False, 0.0, 0

        max_idx = int(np.argmax(ppl_array))
        z_score = (ppl_array[max_idx] - mu) / sigma

        spike_detected = (
            z_score > self.spike_threshold
            or ppl_array[max_idx] > self.absolute_threshold
        )

        return spike_detected, float(z_score), max_idx

    def check_input(self, user_query: str, context: list[dict] | None = None) -> DefenseResult:
        """Screen query for injection via perplexity spike analysis.

        Algorithm:
            1. Compute sliding-window perplexity across the input text.
            2. Detect spikes using z-score of the perplexity distribution.
            3. Flag if z_score > threshold or absolute perplexity too high.
            4. Also analyze context messages for embedded injections.
        """
        self._load_model()

        flags: list[str] = []
        max_z = 0.0

        # Analyze main query
        perplexities, window_texts = self._compute_windowed_perplexity(user_query)
        spike_detected, z_score, spike_idx = self._detect_spike(perplexities)
        max_z = z_score

        if spike_detected:
            spike_text = window_texts[spike_idx] if spike_idx < len(window_texts) else ""
            flags.append(
                f"perplexity_spike:query:z={z_score:.2f}"
                f":ppl={perplexities[spike_idx]:.1f}"
                f":window='{spike_text[:50]}'"
            )

        # Analyze context messages
        if context:
            for i, msg in enumerate(context):
                content = msg.get("content", "")
                if len(content) < 20:
                    continue
                # Truncate very long context to avoid slow inference
                content = content[:2000]
                ctx_ppls, ctx_texts = self._compute_windowed_perplexity(content)
                ctx_spike, ctx_z, ctx_idx = self._detect_spike(ctx_ppls)
                max_z = max(max_z, ctx_z)

                if ctx_spike:
                    ctx_text = ctx_texts[ctx_idx] if ctx_idx < len(ctx_texts) else ""
                    flags.append(
                        f"perplexity_spike:context[{i}]:z={ctx_z:.2f}"
                        f":ppl={ctx_ppls[ctx_idx]:.1f}"
                        f":window='{ctx_text[:50]}'"
                    )

        blocked = len(flags) > 0
        confidence = min(max_z / self.spike_threshold, 1.0) if max_z > 0 else 0.0

        return DefenseResult(
            allowed=not blocked,
            flags=flags,
            confidence=confidence if blocked else 1.0 - confidence * 0.3,
        )

    def get_perplexity_profile(self, text: str) -> dict:
        """Return full perplexity analysis for explainability."""
        self._load_model()

        perplexities, window_texts = self._compute_windowed_perplexity(text)
        spike_detected, z_score, spike_idx = self._detect_spike(perplexities)

        return {
            "n_windows": len(perplexities),
            "mean_perplexity": float(np.mean(perplexities)),
            "std_perplexity": float(np.std(perplexities)),
            "max_perplexity": float(max(perplexities)),
            "min_perplexity": float(min(perplexities)),
            "z_score_of_max": z_score,
            "spike_detected": spike_detected,
            "spike_window_text": window_texts[spike_idx] if spike_detected else None,
            "perplexities": perplexities,
            "window_texts": window_texts,
        }
