"""CDF normalization and aggregation for the Cross-environment
Hyperparameter Setting (CHS) benchmark (Patterson et al., RLC 2024).

The CHS selects a single hyperparameter configuration that performs well
across a set of environments.  Scores from different environments are
made comparable via an empirical CDF normalization that maps each raw
score to the fraction of the score pool (across all HP configs and seeds)
that is strictly lower.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def cdf_normalize(score: float, pool: np.ndarray) -> float:
    """CDF-normalize a single score against a reference pool.

    Args:
        score: Raw performance value to normalize.
        pool: 1-D array of all raw scores observed for the same
            environment (across all HP configs and seeds).

    Returns:
        Fraction of pool values strictly less than *score* (in [0, 1]).
    """
    if len(pool) == 0:
        return 0.0
    return float(np.sum(pool < score) / len(pool))


def cdf_normalize_batch(scores: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """CDF-normalize an array of scores against a reference pool.

    Each element is independently normalized against *pool*.

    Args:
        scores: 1-D array of raw scores.
        pool: 1-D reference pool.

    Returns:
        1-D array of CDF-normalized values (same shape as *scores*).
    """
    if len(pool) == 0:
        return np.zeros_like(scores, dtype=float)
    return np.array([float(np.sum(pool < s) / len(pool)) for s in scores])


@dataclass
class CHSScoreStore:
    """Accumulate raw scores per environment, then CDF-normalize post-hoc.

    Trial identifiers are strings (Orion trial IDs) or integers.

    Usage::

        store = CHSScoreStore()
        store.record("building_A", "trial_abc", -5100.0)
        store.record("building_A", "trial_abc", -5200.0)
        store.record("building_B", "trial_abc", -3800.0)
        ...
        best_trial = store.best_trial(store.trial_ids())
    """

    _scores: dict[str, list[tuple[str, float]]] = field(default_factory=dict)

    def record(self, env_id: str, trial_id: str, score: float) -> None:
        """Record a raw score for a given environment and trial."""
        self._scores.setdefault(env_id, []).append((trial_id, score))

    def env_ids(self) -> list[str]:
        return list(self._scores.keys())

    def trial_ids(self) -> list[str]:
        """Return the unique set of trial IDs seen across all environments."""
        ids: set[str] = set()
        for entries in self._scores.values():
            for tid, _ in entries:
                ids.add(tid)
        return sorted(ids)

    def pool_for_env(self, env_id: str) -> np.ndarray:
        """Return all raw scores recorded for *env_id*."""
        entries = self._scores.get(env_id, [])
        return np.array([s for _, s in entries])

    def cdf_score_for_trial(self, trial_id: str) -> float:
        """Compute the mean CDF-normalized score for a given trial.

        For each environment, all raw scores from that trial are
        CDF-normalized against the full pool for that environment,
        then averaged across environments.
        """
        per_env_means: list[float] = []
        for env_id, entries in self._scores.items():
            pool = self.pool_for_env(env_id)
            trial_scores = np.array([s for tid, s in entries if tid == trial_id])
            if len(trial_scores) == 0:
                continue
            normalized = cdf_normalize_batch(trial_scores, pool)
            per_env_means.append(float(normalized.mean()))

        if not per_env_means:
            return 0.0
        return float(np.mean(per_env_means))

    def best_trial(self, trial_ids: list[str]) -> str:
        """Return the trial ID with the highest mean CDF-normalized score."""
        best_id = trial_ids[0]
        best_score = -1.0
        for tid in trial_ids:
            s = self.cdf_score_for_trial(tid)
            if s > best_score:
                best_score = s
                best_id = tid
        return best_id

    def trial_summary(self, trial_ids: list[str]) -> list[tuple[str, float]]:
        """Return ``(trial_id, mean_cdf_score)`` sorted descending."""
        results = [(tid, self.cdf_score_for_trial(tid)) for tid in trial_ids]
        results.sort(key=lambda x: x[1], reverse=True)
        return results


def load_trial_rewards_from_dir(results_dir: Path) -> CHSScoreStore:
    """Load per-building rewards from trial JSON files into a score store.

    Each JSON file is expected to have the structure::

        {
            "trial_id": "<orion-trial-id>",
            "trial_idx": <int>,
            "rewards": {
                "<building_id>": [<float>, ...],
                ...
            }
        }
    """
    store = CHSScoreStore()
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        trial_id: str = str(data["trial_id"])
        rewards: dict[str, list[float]] = data["rewards"]
        for env_id, reward_list in rewards.items():
            for r in reward_list:
                store.record(env_id, trial_id, r)
    return store
