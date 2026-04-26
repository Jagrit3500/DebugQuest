---
title: Debug Quest Environment Server
emoji: 🐛
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - debugging
  - world-modeling
  - code-repair
---

# Debug Quest

**Investigation-First Autonomous Debugging Environment**

An OpenEnv-compliant RL environment that trains LLMs to debug Python codebases autonomously. The agent must investigate a broken codebase using tools, localize the root cause, and apply a surgical fix — without being told where the bug is.

Built for the Meta x PyTorch OpenEnv Hackathon 2026.

---

## The Problem

Tools like GitHub Copilot suggest fixes with human oversight. But autonomous debugging — no human in the loop, no hints — is something LLMs genuinely fail at today. They guess fixes without investigating. They break passing tests. They rewrite entire files when one line needed changing.

Debug Quest trains LLMs to move beyond code suggestion into autonomous action.

---

## Environment Design

### How an Episode Works

The agent receives a buggy Python codebase. It does not know where the bug is. It must investigate using tools, find the bug, fix it, and submit.

```
[Step 1: run_tests] -> [Step 2: read_file] -> [Step 3: apply_fix] -> [Step 4: submit_solution] -> SUCCESS (0.976)
```

Because the agent maintains persistent state across steps — tracking files examined and test outcomes — it effectively models the internal structure of the codebase. This transforms debugging from a guessing game into a systematic investigation.

Maximum 10 steps per episode. Reward is only computed at submission.

### Difficulty Levels

| Level | Name | Description |
|---|---|---|
| 1 | Syntax Bug | Single file, one obvious bug, 1-3 tests failing |
| 2 | Logic Bug | Single file, subtle logic error, 2-3 tests failing |
| 3 | Multi-File Bug | Bug hidden in a helper module, requires cross-file investigation |
| 4 | Boss Level | Hidden bug behind a misleading failing test — designed to fool agents that do not truly investigate |

### Agent Tools

| Tool | Arguments | Purpose |
|---|---|---|
| run_tests | none | Run pytest, see pass/fail summary with tracebacks |
| read_file | file_path | Read a file with line numbers |
| search_codebase | query | Search all Python files for a string |
| apply_fix | file_path, old_code, new_code | Apply a surgical patch (old_code must match exactly once) |
| submit_solution | none | Submit fix, end episode, trigger reward computation |

### Safety Constraints

- Agent cannot modify test files
- Agent cannot delete files
- Agent cannot access system globals or environment variables
- All code execution sandboxed with 30-second timeout per tool call
- Path traversal attempts are blocked at the filesystem level

---

## Reward Design

Four fully programmatic reward functions. No LLM judge anywhere in the pipeline. Every signal is computed by running actual code.

```
Total Reward = 0.45 * R1 + 0.20 * R2 + 0.20 * R3 + 0.15 * R4
```

| Signal | Weight | Formula | What it teaches |
|---|---|---|---|
| R1 Test Completion | 0.45 | tests_passed / tests_total | Fix the actual bug |
| R2 Efficiency | 0.20 | 1.0 - 0.3 * (steps_used / max_steps) | Investigate strategically, not randomly |
| R3 Anti-Regression | 0.20 | Penalty per previously-passing test now failing | Do not break working code |
| R4 Surgical Precision | 0.15 | Penalty per unnecessary line changed | Change only what is needed |

The weights are validated at import time: `assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9`.

### Why Four Rewards

A single pass/fail reward is easy to game. Four independent signals make it much harder:

- An agent that passes tests but uses all 10 steps loses on R2
- An agent that fixes one bug but breaks another loses on R3
- An agent that rewrites entire files loses on R4
- All four must be high simultaneously to score above 0.9

---

## Training

### Approach

- Framework: OpenEnv + TRL GRPO + Unsloth
- Base model: Qwen2.5-1.5B-Instruct (4-bit quantized via Unsloth)
- Method: GRPO reinforcement learning with curriculum learning
- Curriculum: Starts at Level 1, advances when mean reward exceeds 0.6 over the last 5 episodes

### Baseline vs Trained

The untrained model investigates endlessly but never submits a fix. It uses all 10 steps describing failures without taking action.

After GRPO training, the agent learns to run tests first, read the relevant file, apply a minimal fix, and submit.

| Metric | Untrained | Trained |
|---|---|---|
| Mean Reward | 0.6025 | Update after training |
| Episodes Solved | 0 / 10 | Update after training |
| Avg Steps to Solve | never completes | Update after training |

### Smoke Test Result (Scripted Optimal Agent)

This confirms the environment and reward pipeline are correctly wired before any LLM training begins.

```
r1_test_completion   : 1.0000
r2_efficiency        : 0.8800
r3_anti_regression   : 1.0000
r4_surgical_precision: 1.0000
total                : 0.9760

[PASS] done           == True
[PASS] solved         == True
[PASS] reward         == 0.9760  (>= 0.7)
[PASS] r1_completion  == 1.0

Pipeline is wired correctly. Ready for GRPO training.
```

---

## Theme

**Primary: World Modeling 3.1 — Professional Tasks**

The agent maintains a consistent model of a codebase across multiple investigation steps using real tools. It must track what it has examined, what it found, and what remains unknown before acting. This is exactly the kind of real interaction with tools and dynamic systems that Theme 3.1 targets.

**Secondary: Long-Horizon Planning and Instruction Following**

Reward is delayed until submission. The agent must decompose the debugging task across up to 10 sequential steps, track state across those steps, and recover from inefficient early decisions.

**Bonus: Self-Improvement via Adaptive Curriculum**

Difficulty advances automatically as the agent improves. Level 1 cases become trivial before the agent sees Level 3 or Boss Level cases, keeping the training signal informative throughout.

### Why RL and Not Just Prompting

Current LLMs can suggest fixes when told exactly where to look. They cannot autonomously navigate an unknown codebase, decide which files to examine, determine when they have enough information to act, and apply a fix without breaking other functionality.

RL with verifiable rewards closes this gap. Tests either pass or they do not. There is no ambiguity in the signal. Because the agent maintains a persistent state across steps — tracking files examined and test outcomes — it effectively models the internal structure of the codebase, transforming the debugging process from a guessing game into a systematic investigation.

---

## Repository Structure

```
DebugQuest/
├── data/
│   └── cases/                          <- JSON bug case files, one per episode type
├── DebugQuest/
│   ├── server/
│   │   ├── app.py                      <- FastAPI server
│   │   ├── debug_quest_environment.py  <- OpenEnv server wrapper
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── __init__.py
│   ├── client.py                       <- OpenEnv client, never imports server
│   ├── dataset.py                      <- BugRepository, case loading, verification
│   ├── dataset_templates.py            <- BugTemplate definitions for all levels
│   ├── env.py                          <- DebugQuestEnv core logic
│   ├── models.py                       <- Action, Observation, BugCase, EpisodeResult
│   ├── rewards.py                      <- 4 reward functions with frozen dataclass
│   ├── smoke_test.py                   <- end-to-end pipeline verification
│   ├── tools.py                        <- 5 sandboxed agent tools
│   └── openenv.yaml                    <- OpenEnv manifest
└── training/
    └── train_debug_quest.ipynb         <- GRPO training notebook (Kaggle)
```

---

## Quick Start

### Run the smoke test locally

```bash
git clone https://github.com/Jagrit3500/DebugQuest.git
cd DebugQuest/DebugQuest
pip install openenv-core pytest pydantic
python smoke_test.py
```

Expected output:

```
[PASS] done           == True
[PASS] solved         == True
[PASS] reward         == 0.9760  (>= 0.7)
[PASS] r1_completion  == 1.0
Pipeline is wired correctly. Ready for GRPO training.
```

### Start the server locally

```bash
pip install -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Connect via HTTP

```python
import requests

# Reset environment — starts a fresh debugging episode
r = requests.post("http://localhost:8000/reset", json={})
obs = r.json()["observation"]
print(obs["tool_output"])

# Take a step — run the tests to see what is failing
r = requests.post("http://localhost:8000/step", json={
    "action": {
        "tool_name": "run_tests",
        "tool_args": {}
    }
})
print(r.json())
```

---

## Links

- HuggingFace Space: https://huggingface.co/spaces/Jagritjd/debug_quest
- GitHub Repository: https://github.com/Jagrit3500/DebugQuest
- Training Notebook: add Kaggle link after saving public version
- Trained Model: add HuggingFace model link after pushing
- Mini Blog: add HuggingFace blog link after publishing

---

## OpenEnv Validation

```
openenv validate
[OK] debug_quest: Ready for multi-mode deployment
```

Validated with `openenv validate` against openenv-core 0.2.3.

---

## Minimum Requirements Checklist

- OpenEnv framework used (openenv-core 0.2.3)
- Training script using Unsloth and HF TRL (Kaggle notebook)
- Environment hosted on HuggingFace Spaces
- openenv.yaml manifest present and valid
- Clean client/server separation — client never imports server internals
- reset, step, and state implemented following OpenEnv spec
- No reserved tool names used as MCP tools