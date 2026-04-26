"""
rewards.py — Debug Quest Reward Functions

Deterministic rewards only. No LLM judge.
Includes:
1. Final episode reward
2. Optional step/action shaping reward for RL training
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from models import EpisodeResult


WEIGHTS: Dict[str, float] = {
    "test_completion": 0.45,
    "efficiency": 0.20,
    "anti_regression": 0.20,
    "surgical_precision": 0.15,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Reward weights must sum to 1.0"


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


@dataclass(frozen=True)
class ActionRewardBreakdown:
    format_reward: float
    sequence_reward: float
    anti_loop_reward: float
    progress_reward: float
    total: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "format_reward": self.format_reward,
            "sequence_reward": self.sequence_reward,
            "anti_loop_reward": self.anti_loop_reward,
            "progress_reward": self.progress_reward,
            "total": self.total,
        }


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def r1_test_completion(result: EpisodeResult) -> float:
    if result.tests_total == 0:
        return 0.0
    return clamp01(result.final_passing / result.tests_total)


def r2_efficiency(result: EpisodeResult) -> float:
    if result.max_steps == 0:
        return 1.0
    step_fraction = result.steps_used / result.max_steps
    return clamp01(1.0 - 0.3 * step_fraction)


def r3_anti_regression(result: EpisodeResult) -> float:
    regressions = max(0, result.originally_passing - result.final_passing)
    return clamp01(1.0 - regressions * 0.25)


def r4_surgical_precision(result: EpisodeResult) -> float:
    gt_lines = max(1, result.ground_truth_lines)
    excess = max(0, result.lines_changed - gt_lines)
    return clamp01(1.0 - excess * 0.1)


def compute_reward(
    result: EpisodeResult,
    submitted: bool = False,
) -> RewardBreakdown:
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

    if result.solved and not submitted:
        total *= 0.95

    return RewardBreakdown(
        r1_test_completion=r1,
        r2_efficiency=r2,
        r3_anti_regression=r3,
        r4_surgical_precision=r4,
        total=clamp01(total),
    )


def expected_next_tool(
    actions_so_far: Sequence[str],
    investigation_order: Sequence[str],
) -> str:
    """
    Returns the next expected tool from a configurable workflow.

    This is not bug-specific hardcoding.
    It represents the debugging workflow:
    investigate -> inspect -> fix -> submit.
    """
    index = min(len(actions_so_far), len(investigation_order) - 1)
    return investigation_order[index]


def compute_action_reward(
    tool_name: str,
    actions_so_far: Sequence[str],
    has_valid_format: bool,
    tests_before: int,
    tests_after: int,
    investigation_order: Sequence[str] = (
        "run_tests",
        "read_file",
        "apply_fix",
        "submit_solution",
    ),
) -> ActionRewardBreakdown:
    """
    Step-level shaping reward for RL training.

    Purpose:
    - prevents run_tests spam
    - rewards correct investigation order
    - rewards real test progress
    - stays generic, not tied to any specific bug case
    """

    expected = expected_next_tool(actions_so_far, investigation_order)

    format_reward = 0.20 if has_valid_format else 0.0
    sequence_reward = 0.45 if tool_name == expected else 0.0

    anti_loop_reward = 0.20
    if actions_so_far and tool_name == actions_so_far[-1]:
        anti_loop_reward = 0.0

    progress_reward = 0.15 if tests_after > tests_before else 0.0

    total = clamp01(
        format_reward
        + sequence_reward
        + anti_loop_reward
        + progress_reward
    )

    return ActionRewardBreakdown(
        format_reward=format_reward,
        sequence_reward=sequence_reward,
        anti_loop_reward=anti_loop_reward,
        progress_reward=progress_reward,
        total=total,
    )