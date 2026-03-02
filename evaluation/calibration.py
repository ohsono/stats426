"""
Expected Calibration Error (ECE) and reliability diagrams (Phase 4.1).

Measures the gap between a model's confidence and its actual accuracy,
critical for safety in autonomous driving.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error.

    Parameters
    ----------
    y_true : (N,) int array of true labels.
    y_prob : (N, C) float array of softmax probabilities.
    n_bins : Number of confidence bins.

    Returns
    -------
    ece : float — the ECE value.
    bin_confidences : (n_bins,) mean confidence per bin.
    bin_accuracies  : (n_bins,) accuracy per bin.
    bin_counts      : (n_bins,) sample count per bin.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidences = np.zeros(n_bins)
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        count = mask.sum()
        bin_counts[i] = count
        if count > 0:
            bin_confidences[i] = confidences[mask].mean()
            bin_accuracies[i] = accuracies[mask].mean()

    # Weighted average of |accuracy - confidence| per bin
    total = len(y_true)
    ece = float(np.sum(bin_counts / total * np.abs(bin_accuracies - bin_confidences)))

    return ece, bin_confidences, bin_accuracies, bin_counts


class TemperatureScaling:
    """
    Post-hoc temperature scaling to calibrate softmax outputs.

    Usage:
        ts = TemperatureScaling()
        ts.fit(val_logits, val_labels)
        calibrated = ts.calibrate(test_logits)
    """

    def __init__(self, initial_temperature: float = 1.5):
        self.temperature = initial_temperature

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100,
    ):
        """
        Optimize temperature via simple gradient descent on NLL.
        """
        import warnings
        T = self.temperature
        for _ in range(max_iter):
            scaled = logits / T
            # Stable softmax
            shifted = scaled - scaled.max(axis=1, keepdims=True)
            exp_s = np.exp(shifted)
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)

            # NLL
            n = len(labels)
            correct_probs = probs[np.arange(n), labels]
            correct_probs = np.clip(correct_probs, 1e-12, 1.0)

            # Gradient of NLL w.r.t. T
            # d(NLL)/dT = (1/T^2) * mean( sum_j p_j * s_j - s_y )
            # where s = logits
            weighted = (probs * logits).sum(axis=1)
            grad = np.mean(weighted - logits[np.arange(n), labels]) / (T ** 2)

            T -= lr * grad
            T = max(T, 0.01)  # Clamp to prevent collapse

        self.temperature = T

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply learned temperature to logits and return calibrated probs."""
        scaled = logits / self.temperature
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        return exp_s / exp_s.sum(axis=1, keepdims=True)
