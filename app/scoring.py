"""Losses, the noise threshold, gain and fitness. Runs outside the sandbox."""

from collections.abc import Callable, Sequence

import numpy as np

EPS = 1e-15


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def pinball_90(y: np.ndarray, p: np.ndarray) -> float:
    diff = y - p
    return float(np.mean(np.maximum(0.9 * diff, -0.1 * diff)))


LOSSES: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "log_loss": log_loss,
    "rmse": rmse,
    "pinball_90": pinball_90,
}


def score(loss: str, y: np.ndarray, preds: np.ndarray) -> float:
    """Loss of `preds` against `y`. Raises ValueError for unusable predictions."""
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    if preds.shape != y.shape:
        raise ValueError(f"expected {y.shape[0]} predictions, got {preds.shape[0]}")
    if not np.all(np.isfinite(preds)):
        raise ValueError("predictions contain NaN or inf")
    if loss == "log_loss" and (preds.min() < 0 or preds.max() > 1):
        raise ValueError("log_loss needs probabilities between 0 and 1")
    return LOSSES[loss](y, preds)


def spread(losses: Sequence[float]) -> float:
    """Sample standard deviation; 0 for a single value."""
    return float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0


def noise_threshold(baseline_losses: Sequence[float]) -> float:
    """The larger of 2 x the spread across seeds and 0.5% of the mean (L0)."""
    return max(2 * spread(baseline_losses), 0.005 * float(np.mean(baseline_losses)))


def beats(new: float, best: float, threshold: float) -> bool:
    return new < best - threshold


def gain(l0: float, lbest: float) -> float:
    return (l0 - lbest) / l0


def fitness(l0_per_seed: Sequence[float], lbest_per_seed: Sequence[float]) -> float:
    """Average gain across seeds (1, 2, 3 on the test split)."""
    return float(np.mean([gain(a, b) for a, b in zip(l0_per_seed, lbest_per_seed, strict=True)]))
