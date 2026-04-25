# env.py — Debug Quest Environment

from __future__ import annotations

import random
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dataset import generate_case, generate_curriculum_case, write_case_to_workspace
from models import (
    BugCase,
    DebugQuestAction,
    DebugQuestObservation,
    EpisodeResult,
)
from rewards import RewardBreakdown, compute_reward
from tools import dispatch_tool, run_tests


MAX_STEPS: int = 10
TOOL_TIMEOUT: int = 30


class DebugQuestEnv:
    def __init__(
        self,
        max_steps: int = MAX_STEPS,
        tool_timeout: int = TOOL_TIMEOUT,
        workspace_root: Optional[Path] = None,
    ) -> None:
        self.max_steps = max_steps
        self.tool_timeout = tool_timeout
        self._workspace_root = workspace_root

        self._case: Optional[BugCase] = None
        self._workspace: Optional[Path] = None
        self._tmpdir: Optional[tempfile.TemporaryDirectory] = None

        self._episode_id: str = ""
        self._steps_used: int = 0
        self._done: bool = False
        self._submitted: bool = False

        self._originally_passing: int = 0
        self._lines_changed_total: int = 0
        self._files_examined: List[str] = []
        self._fixes_attempted: int = 0

        self._episode_count: int = 0
        self._reward_history: List[float] = []

    # ------------------------------------------------------------------

    def reset(
        self,
        level: Optional[int] = None,
        case: Optional[BugCase] = None,
        curriculum: bool = False,
        seed: Optional[int] = None,
    ) -> DebugQuestObservation:

        self._cleanup_workspace()

        rng = random.Random(seed) if seed is not None else None

        if case is not None:
            self._case = case
        elif curriculum:
            self._case = generate_curriculum_case(
                episode_num=self._episode_count,
                reward_history=self._reward_history,
                rng=rng,
            )
        else:
            self._case = generate_case(
                level=level if level is not None else 1,
                rng=rng,
            )

        # Workspace creation
        if self._workspace_root is not None:
            self._workspace = (
                self._workspace_root / f"episode_{uuid.uuid4().hex[:8]}"
            )
            self._workspace.mkdir(parents=True, exist_ok=True)
        else:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="debugquest_")
            self._workspace = Path(self._tmpdir.name)

        write_case_to_workspace(self._case, self._workspace)

        # Reset state
        self._episode_id = f"ep-{uuid.uuid4().hex[:12]}"
        self._steps_used = 0
        self._done = False
        self._submitted = False
        self._lines_changed_total = 0
        self._files_examined = []
        self._fixes_attempted = 0
        self._episode_count += 1

        # Baseline tests
        baseline = run_tests(
            workspace=self._workspace,
            test_filename=self._case.test_filename,
            timeout=self.tool_timeout,
        )

        self._originally_passing = baseline["tests_passed"]

        files_listed = "\n".join(
            f"  - {f}" for f in sorted(self._case.files.keys())
        )

        intro = (
            f"=== Debug Quest | {self._case.level_name} ===\n"
            f"Episode : {self._episode_id}\n"
            f"Case    : {self._case.case_id}\n\n"
            f"A bug has been injected. Find it and fix it.\n\n"
            f"Files in this codebase:\n{files_listed}\n\n"
            f"Test file : {self._case.test_filename}\n"
            f"Max steps : {self.max_steps}\n\n"
            f"Tip: start with run_tests() to see failures."
        )

        return DebugQuestObservation(
            tool_output=intro,
            tool_success=True,
            steps_used=0,
            steps_remaining=self.max_steps,
            tests_passed=self._originally_passing,
            tests_total=baseline["tests_total"],
            failing_tests=baseline["failing_tests"],
            files_examined=[],
            fixes_attempted=0,
            level=self._case.level,
        )

    # ------------------------------------------------------------------

    def step(
        self,
        action: DebugQuestAction,
    ) -> Tuple[DebugQuestObservation, float, bool, Dict[str, Any]]:

        self._assert_ready()

        self._steps_used += 1
        tool_name = action.tool_name.value

        result = dispatch_tool(
            tool_name=tool_name,
            tool_args=action.tool_args,
            workspace=self._workspace,
            test_filename=self._case.test_filename,
        )

        success = result["success"]
        output = result["output"]
        metadata = result.get("metadata", {})

        # Track usage
        if tool_name == "read_file" and success:
            fp = action.tool_args.get("file_path", "")
            if fp and fp not in self._files_examined:
                self._files_examined.append(fp)

        if tool_name == "apply_fix" and success:
            self._fixes_attempted += 1
            self._lines_changed_total += metadata.get("lines_changed", 0)

        if tool_name == "submit_solution":
            self._submitted = True

        # Terminal conditions
        submitted = tool_name == "submit_solution"
        steps_exhausted = self._steps_used >= self.max_steps
        self._done = submitted or steps_exhausted

        # Always re-run tests for observation
        current = run_tests(
            workspace=self._workspace,
            test_filename=self._case.test_filename,
            timeout=self.tool_timeout,
        )

        obs = DebugQuestObservation(
            tool_output=output,
            tool_success=success,
            steps_used=self._steps_used,
            steps_remaining=max(0, self.max_steps - self._steps_used),
            tests_passed=current["tests_passed"],
            tests_total=current["tests_total"],
            failing_tests=current["failing_tests"],
            files_examined=list(self._files_examined),
            fixes_attempted=self._fixes_attempted,
            level=self._case.level,
        )

        info = {
            "episode_id": self._episode_id,
            "level": self._case.level,
            "steps_used": self._steps_used,
            "submitted": self._submitted,
            "done": self._done,
        }

        reward = 0.0

        # Terminal reward
        if self._done:
            breakdown = self._compute_terminal_reward(
                tests_passed=current["tests_passed"],
                tests_total=current["tests_total"],
            )

            reward = breakdown.total
            self._reward_history.append(reward)

            info["reward_breakdown"] = breakdown.as_dict()
            info["solved"] = (
                current["tests_total"] > 0
                and current["tests_passed"] == current["tests_total"]
            )

            self._cleanup_workspace()

        return obs, reward, self._done, info

    # ------------------------------------------------------------------

    def get_metadata(self) -> Dict[str, Any]:
        """Lightweight metadata for logging (W&B, dashboards, judges)."""
        if self._case is None:
            return {"status": "not_initialised"}

        return {
            "episode_id": self._episode_id,
            "case_id": self._case.case_id,
            "level": self._case.level,
            "level_name": self._case.level_name,
        }

    # ------------------------------------------------------------------

    def state(self) -> Dict[str, Any]:
        """Minimal state snapshot."""
        if self._case is None:
            return {"status": "not_initialised"}

        return {
            "episode_id": self._episode_id,
            "level": self._case.level,
            "level_name": self._case.level_name,
            "case_id": self._case.case_id,
            "steps_used": self._steps_used,
            "done": self._done,
        }

    # ------------------------------------------------------------------

    def _assert_ready(self) -> None:
        if self._case is None or self._workspace is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode finished. Call reset().")

    # ------------------------------------------------------------------

    def _compute_terminal_reward(
        self,
        tests_passed: int,
        tests_total: int,
    ) -> RewardBreakdown:

        ground_truth_lines = sum(
            len(lines) for lines in self._case.changed_lines.values()
        )

        result = EpisodeResult(
            episode_id=self._episode_id,
            level=self._case.level,
            tests_passed=tests_passed,
            tests_total=tests_total,
            originally_passing=self._originally_passing,
            final_passing=tests_passed,
            steps_used=self._steps_used,
            max_steps=self.max_steps,
            lines_changed=self._lines_changed_total,
            ground_truth_lines=max(1, ground_truth_lines),
            solved=tests_total > 0 and tests_passed == tests_total,
        )

        return compute_reward(result, submitted=self._submitted)

    # ------------------------------------------------------------------

    def _cleanup_workspace(self) -> None:
        """
        Clean up temporary workspace.

        NOTE:
        If workspace_root is provided, we intentionally do NOT delete
        the workspace to allow debugging and inspection.
        """
        if self._tmpdir is not None:
            try:
                self._tmpdir.cleanup()
            except Exception:
                pass
            self._tmpdir = None
            self._workspace = None