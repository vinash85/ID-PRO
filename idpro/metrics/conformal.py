"""
Conformal prediction for IDPro.

Implements the two calibration strategies from ARCHITECTURE_CHOICES.md §11:
  - conf-multi-sample: generate N samples per input, calibrate a prediction set
    whose *size* reflects uncertainty (large set = dark/novel protein).
  - conf-span-agreement: use pre-LLM vs post-LLM evidence-head agreement as a
    secondary confidence signal; aggregate with the generation-level signal.

All predictors are *split conformal* (Vovk et al.; Angelopoulos & Bates 2023):
given exchangeable calibration scores {s_i}, for miscoverage level alpha the
threshold is

    tau = Quantile_{(n+1)(1-alpha)/n}({s_i})

and for a fresh point we include candidates whose nonconformity score is
<= tau. This yields the marginal coverage guarantee

    P( y_true in PredSet(x) ) >= 1 - alpha

under exchangeability, independent of the underlying model.

The module is deliberately free of heavy dependencies (only numpy + stdlib) so
it can run without loading the LLM.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Nonconformity scores
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def token_f1(pred: str, gt: str) -> float:
    """Token-level F1 between pred and gt. Returns 0.0 on empty input."""
    p, g = _tokenize(pred), _tokenize(gt)
    if not p or not g:
        return 0.0
    from collections import Counter
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def f1_nonconformity(pred: str, gt: str) -> float:
    """1 - F1. Lower = better. Range [0, 1]."""
    return 1.0 - token_f1(pred, gt)


# ---------------------------------------------------------------------------
# Split conformal predictor (core)
# ---------------------------------------------------------------------------


@dataclass
class SplitConformalPredictor:
    """
    Split conformal predictor.

    Call `calibrate(scores)` once on a held-out calibration set, then use
    `threshold(alpha)` or `in_set(score, alpha)` at test time.

    Parameters
    ----------
    cal_scores : optional np.ndarray
        If given, calibration is done at construction time.
    """

    cal_scores: Optional[np.ndarray] = None

    def calibrate(self, scores: Sequence[float]) -> "SplitConformalPredictor":
        s = np.asarray(list(scores), dtype=float)
        if s.size == 0:
            raise ValueError("calibration scores must be non-empty")
        if np.any(~np.isfinite(s)):
            raise ValueError("calibration scores contain NaN/Inf")
        self.cal_scores = np.sort(s)
        return self

    @property
    def n_cal(self) -> int:
        return 0 if self.cal_scores is None else int(self.cal_scores.size)

    def threshold(self, alpha: float) -> float:
        """
        Finite-sample-corrected (1-alpha) quantile of calibration scores:
            q = ceil((n+1)(1-alpha)) / n
        Returns +inf if q > 1 (insufficient calibration data at that alpha).
        """
        if self.cal_scores is None:
            raise RuntimeError("must calibrate() before calling threshold()")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        n = self.n_cal
        q_level = math.ceil((n + 1) * (1 - alpha)) / n
        if q_level > 1.0:
            return float("inf")
        # np.quantile with interpolation='higher' matches the conservative
        # convention used in Angelopoulos & Bates.
        return float(np.quantile(self.cal_scores, q_level, method="higher"))

    def in_set(self, score: float, alpha: float) -> bool:
        """True iff a test score would be accepted into the prediction set."""
        return float(score) <= self.threshold(alpha)

    def empirical_coverage(self, test_scores: Sequence[float], alpha: float) -> float:
        t = self.threshold(alpha)
        s = np.asarray(list(test_scores), dtype=float)
        if s.size == 0:
            return 0.0
        return float(np.mean(s <= t))


# ---------------------------------------------------------------------------
# Multi-sample conformal (conf-multi-sample)
# ---------------------------------------------------------------------------


@dataclass
class MultiSampleConformal:
    """
    Multi-sample conformal predictor.

    For each input the model produces N candidate generations (e.g. N=5 CoT
    samples at T=0.7). Each candidate has a nonconformity score with respect
    to the ground truth; at calibration time we store the *best* (minimum)
    candidate score for each example — this asks the question "is there a
    correct answer somewhere in my N samples?".

    At test time, the prediction *set* is the subset of candidates whose score
    would be accepted at level alpha. The set size is the uncertainty signal:
      - Small set (1-2)   : confident, well-characterized protein.
      - Large set (N-1, N): uncertain, likely dark / novel.

    The coverage guarantee is:
        P( at least one candidate has score <= tau | alpha ) >= 1 - alpha.
    """

    predictor: SplitConformalPredictor = field(default_factory=SplitConformalPredictor)

    def calibrate(self, best_scores_per_example: Sequence[float]) -> "MultiSampleConformal":
        self.predictor.calibrate(best_scores_per_example)
        return self

    def prediction_set(
        self,
        candidate_scores: Sequence[float],
        alpha: float,
    ) -> List[int]:
        """Indices of candidates whose score is below tau(alpha)."""
        tau = self.predictor.threshold(alpha)
        return [i for i, s in enumerate(candidate_scores) if float(s) <= tau]

    def set_size(self, candidate_scores: Sequence[float], alpha: float) -> int:
        return len(self.prediction_set(candidate_scores, alpha))


# ---------------------------------------------------------------------------
# Span-agreement conformal (conf-span-agreement)
# ---------------------------------------------------------------------------


def span_agreement(
    pre_llm_probs: np.ndarray,
    post_llm_probs: np.ndarray,
) -> float:
    """
    Agreement between pre-LLM and post-LLM evidence-head predictions.

    Each array is shape (L, K): per-residue class probabilities. We use
    mean per-residue TV distance converted to an agreement score in [0, 1]:
        agreement = 1 - mean_i( 0.5 * sum_k |p_i^pre - p_i^post| )
    """
    pre = np.asarray(pre_llm_probs, dtype=float)
    post = np.asarray(post_llm_probs, dtype=float)
    if pre.shape != post.shape or pre.ndim != 2:
        raise ValueError(f"shape mismatch: pre={pre.shape} post={post.shape}")
    tv = 0.5 * np.abs(pre - post).sum(axis=1)
    return float(1.0 - tv.mean())


@dataclass
class SpanAgreementConformal:
    """
    Use (1 - span_agreement) as the nonconformity score for split conformal.
    High agreement between heads => low nonconformity => accepted.
    """

    predictor: SplitConformalPredictor = field(default_factory=SplitConformalPredictor)

    def nonconformity(self, pre: np.ndarray, post: np.ndarray) -> float:
        return 1.0 - span_agreement(pre, post)

    def calibrate_from_heads(
        self,
        pairs: Iterable[Tuple[np.ndarray, np.ndarray]],
    ) -> "SpanAgreementConformal":
        scores = [self.nonconformity(pre, post) for pre, post in pairs]
        self.predictor.calibrate(scores)
        return self

    def in_set(self, pre: np.ndarray, post: np.ndarray, alpha: float) -> bool:
        return self.predictor.in_set(self.nonconformity(pre, post), alpha)


# ---------------------------------------------------------------------------
# Combined (multi-signal) conformal
# ---------------------------------------------------------------------------


@dataclass
class CombinedConformal:
    """
    Multi-signal conformal predictor.

    Each signal is scored independently and turned into a per-signal p-value
    via the rank in its own calibration set. We combine via the Bonferroni
    intersection rule: accept iff every signal individually accepts at level
    alpha / K. This preserves the marginal coverage guarantee (1-alpha) while
    requiring agreement across signals.
    """

    signals: List[SplitConformalPredictor] = field(default_factory=list)

    def add(self, signal: SplitConformalPredictor) -> "CombinedConformal":
        self.signals.append(signal)
        return self

    def in_set(self, scores: Sequence[float], alpha: float) -> bool:
        if len(scores) != len(self.signals):
            raise ValueError(
                f"expected {len(self.signals)} scores, got {len(scores)}"
            )
        if not self.signals:
            raise RuntimeError("add at least one signal")
        per_signal_alpha = alpha / len(self.signals)
        return all(
            sig.in_set(float(s), per_signal_alpha)
            for sig, s in zip(self.signals, scores)
        )


# ---------------------------------------------------------------------------
# Convenience: from a list of (pred, gt) pairs
# ---------------------------------------------------------------------------


def calibrate_text_predictor(
    pairs: Sequence[Tuple[str, str]],
    score_fn: Callable[[str, str], float] = f1_nonconformity,
) -> SplitConformalPredictor:
    """Helper: build a split-conformal predictor from (pred, gt) text pairs."""
    scores = [score_fn(p, g) for p, g in pairs]
    return SplitConformalPredictor().calibrate(scores)
