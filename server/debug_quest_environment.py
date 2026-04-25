# Copyright (c) Meta Platforms, Inc. and affiliates.
# Debug Quest - Investigation-First Autonomous Debugging Environment

"""
OpenEnv-compliant server wrapper around DebugQuestEnv.

This module bridges the OpenEnv HTTP server interface
(reset / step / state) with the core DebugQuestEnv logic.

The server layer is intentionally thin:
  - All game logic lives in env.py
  - All reward logic lives in rewards.py
  - All tool logic lives in tools.py
  - This file only translates HTTP payloads to/from env calls
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..env import DebugQuestEnv
    from ..models import DebugQuestAction, DebugQuestObservation
except ImportError:
    from env import DebugQuestEnv
    from models import DebugQuestAction, DebugQuestObservation


class DebugQuestEnvironment(Environment):
    """
    OpenEnv server wrapper for Debug Quest.

    Each WebSocket client gets its own instance of this class,
    which owns one DebugQuestEnv episode.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._env = DebugQuestEnv(max_steps=10)
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self._episode_count: int = 0
        self._current_level: int = 1
        self._last_reward: float = 0.0
        self._last_done: bool = False
        self._last_info: dict[str, Any] = {}

    def reset(self) -> DebugQuestObservation:
        """
        Start a fresh debugging episode.

        First episode starts at Level 1.
        Later episodes may use curriculum progression.
        """

        use_curriculum = self._episode_count > 0
        obs = self._env.reset(curriculum=use_curriculum)

        self._episode_count += 1

        meta = self._env.get_metadata()
        self._state = State(
            episode_id=meta.get("episode_id", str(uuid4())),
            step_count=0,
        )

        self._current_level = obs.level
        self._last_reward = 0.0
        self._last_done = False
        self._last_info = {}

        return obs

    def step(self, action: DebugQuestAction) -> DebugQuestObservation:
        """
        Execute one agent tool call.

        Reward and done are embedded into the observation for OpenEnv server use.
        """

        obs, reward, done, info = self._env.step(action)

        self._state.step_count += 1
        self._last_reward = reward
        self._last_done = done
        self._last_info = info

        obs.reward = reward
        obs.done = done

        return obs

    @property
    def state(self) -> State:
        """
        Return lightweight state snapshot for the /state endpoint.
        """

        return self._state

    def get_metadata(self) -> dict[str, Any]:
        """
        Rich metadata for logging, W&B, and judge inspection.
        """

        env_meta = self._env.get_metadata()

        return {
            **env_meta,
            "episode_count": self._episode_count,
            "current_level": self._current_level,
            "last_reward": self._last_reward,
            "last_done": self._last_done,
            "reward_breakdown": self._last_info.get("reward_breakdown", {}),
            "solved": self._last_info.get("solved", False),
        }