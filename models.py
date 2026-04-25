from enum import Enum
from typing import Any, Dict, List

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field, model_validator


class DebugTool(str, Enum):
    RUN_TESTS = "run_tests"
    READ_FILE = "read_file"
    SEARCH_CODEBASE = "search_codebase"
    APPLY_FIX = "apply_fix"
    SUBMIT_SOLUTION = "submit_solution"


LEVEL_NAMES = {
    1: "Level 1: Syntax Bug",
    2: "Level 2: Logic Bug",
    3: "Level 3: Multi-file Bug",
    4: "Boss Level: Misleading Test + Hidden Bug",
}


class DebugQuestAction(Action):
    """One tool call made by the agent."""

    tool_name: DebugTool = Field(..., description="Tool to execute")
    tool_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the selected tool",
    )


class DebugQuestObservation(Observation):
    """Observation returned after each environment step."""

    tool_output: str = Field(default="")
    tool_success: bool = Field(default=True)

    steps_used: int = Field(default=0)
    steps_remaining: int = Field(default=10)

    tests_passed: int = Field(default=0)
    tests_total: int = Field(default=0)
    failing_tests: List[str] = Field(default_factory=list)

    files_examined: List[str] = Field(default_factory=list)
    fixes_attempted: int = Field(default=0)

    level: int = Field(default=1)

    @property
    def level_name(self) -> str:
        return LEVEL_NAMES.get(self.level, "Unknown Level")

    @model_validator(mode="after")
    def validate_counts(self):
        if self.tests_passed > self.tests_total:
            raise ValueError("tests_passed cannot exceed tests_total")
        if self.steps_remaining < 0:
            raise ValueError("steps_remaining cannot be negative")
        return self


class BugCase(BaseModel):
    """Single generated debugging episode."""

    case_id: str
    level: int
    files: Dict[str, str]
    test_file: str
    test_filename: str
    ground_truth_fix: Dict[str, str]
    changed_lines: Dict[str, List[int]]
    bug_description: str

    @property
    def level_name(self) -> str:
        return LEVEL_NAMES.get(self.level, "Unknown Level")


class EpisodeResult(BaseModel):
    """Final episode result used for reward computation and logging."""

    episode_id: str
    level: int

    tests_passed: int
    tests_total: int

    originally_passing: int
    final_passing: int

    steps_used: int
    max_steps: int

    lines_changed: int
    ground_truth_lines: int

    solved: bool

    @model_validator(mode="after")
    def validate_result(self):
        if self.tests_passed > self.tests_total:
            raise ValueError("tests_passed cannot exceed tests_total")
        if self.final_passing > self.tests_total:
            raise ValueError("final_passing cannot exceed tests_total")
        if self.steps_used > self.max_steps:
            raise ValueError("steps_used cannot exceed max_steps")
        return self