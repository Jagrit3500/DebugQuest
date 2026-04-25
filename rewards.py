"""
rewards.py — Debug Quest Reward Functions

Four fully programmatic reward components, zero LLM judges.

Key Design Principles:
- Dense reward signals (agent learns even when failing)
- Anti-reward-hacking (precision + regression penalties)
- Fully deterministic (pytest-based, no LLM evaluation)
- Submission-aware (agent must verify solution explicitly)

Weights:
    R1  test_completion      0.45
    R2  efficiency           0.20
    R3  anti_regression      0.20
    R4  surgical_precision   0.15
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from models import EpisodeResult


# ---------------------------------------------------------------------------
# Weight registry — single source of truth
# ---------------------------------------------------------------------------

WEIGHTS: Dict[str, float] = {
    "test_completion": 0.45,
    "efficiency": 0.20,
    "anti_regression": 0.20,
    "surgical_precision": 0.15,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Reward weights must sum to 1.0"


# ---------------------------------------------------------------------------
# Reward container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RewardBreakdown:
    r1_test_completion: float
    r2_efficiency: float
    r3_anti_regression: float
    r4_surgical_precision: float
    total: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "r1_test_completion": self.r1_test_completion,
            "r2_efficiency": self.r2_efficiency,
            "r3_anti_regression": self.r3_anti_regression,
            "r4_surgical_precision": self.r4_surgical_precision,
            "total": self.total,
        }


# ---------------------------------------------------------------------------
# R1 — Test Completion
# ---------------------------------------------------------------------------

def r1_test_completion(result: EpisodeResult) -> float:
    """
    Fraction of tests passing at episode end.

    Uses final_passing so partial progress is rewarded
    even if submit_solution was never called.
    """
    if result.tests_total == 0:
        return 0.0

    return result.final_passing / result.tests_total


# ---------------------------------------------------------------------------
# R2 — Efficiency
# ---------------------------------------------------------------------------

def r2_efficiency(result: EpisodeResult) -> float:
    """
    Rewards solving in fewer steps.

    Range: [0.7, 1.0]
    Always positive → prevents reward collapse early in training.
    """
    if result.max_steps == 0:
        return 1.0

    step_fraction = result.steps_used / result.max_steps
    return max(0.0, 1.0 - 0.3 * step_fraction)


# ---------------------------------------------------------------------------
# R3 — Anti-Regression
# ---------------------------------------------------------------------------

def r3_anti_regression(result: EpisodeResult) -> float:
    """
    Penalizes breaking previously passing tests.

    Encourages safe fixes instead of reckless edits.
    """
    regressions = max(0, result.originally_passing - result.final_passing)

    if regressions == 0:
        return 1.0

    penalty = regressions * 0.25
    return max(0.0, 1.0 - penalty)


# ---------------------------------------------------------------------------
# R4 — Surgical Precision
# ---------------------------------------------------------------------------

def r4_surgical_precision(result: EpisodeResult) -> float:
    """
    Rewards minimal edits.

    Fixes should touch only necessary lines.

    Edge case handled:
    - If ground_truth_lines = 0, clamp to 1 to avoid degenerate behaviour.
    """
    gt_lines = max(1, result.ground_truth_lines)

    excess = max(0, result.lines_changed - gt_lines)

    if excess == 0:
        return 1.0

    penalty = excess * 0.1
    return max(0.0, 1.0 - penalty)


# ---------------------------------------------------------------------------
# Final Reward
# ---------------------------------------------------------------------------

def compute_reward(
    result: EpisodeResult,
    submitted: bool = False,
) -> RewardBreakdown:
    """
    Compute final reward.

    Submission logic:
    - Agent is expected to call submit_solution() to verify correctness.
    - If agent solves the problem (all tests pass) but never submits,
      we apply a small penalty.
    
    Why:
    - Prevents agent from "accidentally" reaching a solved state
    - Encourages proper verification behavior
    - Models real-world debugging workflow

    This penalty is intentionally small (5%) so it does not dominate learning.
    """

    r1 = r1_test_completion(result)
    r2 = r2_efficiency(result)
    r3 = r3_anti_regression(result)
    r4 = r4_surgical_precision(result)

    total = (
        WEIGHTS["test_completion"] * r1
        + WEIGHTS["efficiency"] * r2
        + WEIGHTS["anti_regression"] * r3
        + WEIGHTS["surgical_precision"] * r4
    )

    # ------------------------------------------------------------------
    # Submission awareness
    # ------------------------------------------------------------------
    if result.solved and not submitted:
        total *= 0.95  # slight penalty for not verifying solution

    # Clamp to [0, 1]
    total = max(0.0, min(1.0, total))

    return RewardBreakdown(
        r1_test_completion=r1,
        r2_efficiency=r2,
        r3_anti_regression=r3,
        r4_surgical_precision=r4,
        total=total,
    )