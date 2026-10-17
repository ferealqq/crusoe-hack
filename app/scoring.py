"""Losses, noise threshold, gain, fitness and ranking (README "Scoring" and "Curation")."""

import statistics
from dataclasses import dataclass

import numpy as np

from app.schemas import Experiment

EPS = 1e-15


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def spread(losses: list[float]) -> float:
    """Population standard deviation; 0 for a single value."""
    return statistics.pstdev(losses) if len(losses) > 1 else 0.0


def noise_threshold(baseline_losses: list[float]) -> float:
    """The larger of 2 x the spread and 0.5% of L0. Gaps below it are ties."""
    return max(2 * spread(baseline_losses), 0.005 * statistics.mean(baseline_losses))


def gain(l0: float, loss: float) -> float:
    return (l0 - loss) / l0


def fitness(l0_by_seed: list[float], best_by_seed: list[float]) -> float:
    """Average gain across seeds, pairing each seed's starting and best loss."""
    gains = [gain(a, b) for a, b in zip(l0_by_seed, best_by_seed, strict=True)]
    return statistics.mean(gains)


def beats(loss: float, best: float, threshold: float) -> bool:
    return loss < best - threshold


@dataclass(frozen=True)
class Ranked:
    agent: str
    best_run_id: str | None
    best_loss: float | None
    gain: float | None  # over the generation's starting best
    tied: bool  # within the noise threshold of the researcher ranked above


def rank_researchers(
    run_logs: dict[str, list[Experiment]], start_best: float, threshold: float
) -> list[Ranked]:
    """Rank by gain over the generation's starting best, from the run logs (not the reports).
    Researchers with no successful run rank last."""
    rows: list[Ranked] = []
    for agent, runs in run_logs.items():
        ok = [r for r in runs if r.status == "ok" and r.loss is not None]
        top = min(ok, key=lambda r: r.loss) if ok else None
        rows.append(
            Ranked(
                agent=agent,
                best_run_id=top.run_id if top else None,
                best_loss=top.loss if top else None,
                gain=gain(start_best, top.loss) if top else None,
                tied=False,
            )
        )
    rows.sort(key=lambda r: (r.best_loss is None, r.best_loss or 0.0, r.agent))
    ranked: list[Ranked] = []
    for i, r in enumerate(rows):
        prev = rows[i - 1] if i else None
        tied = bool(
            prev
            and prev.best_loss is not None
            and r.best_loss is not None
            and abs(r.best_loss - prev.best_loss) < threshold
        )
        ranked.append(Ranked(r.agent, r.best_run_id, r.best_loss, r.gain, tied))
    return ranked
