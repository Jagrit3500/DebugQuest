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

Investigation-First Autonomous Debugging Environment

An OpenEnv-compliant RL environment that trains LLMs to debug Python codebases autonomously. The agent must investigate a broken codebase using tools, localize the root cause, and apply a surgical fix without being told where the bug is.

Built for the Meta x PyTorch OpenEnv Hackathon 2026.

---

## The Problem

Current LLMs guess fixes without investigating. They break passing tests. They rewrite entire files when one line is needed.

Debug Quest trains LLMs to move beyond code suggestion into autonomous action.

---

## Environment Design

### How an Episode Works

The agent receives a buggy Python codebase. It does not know where the bug is. It must investigate using tools, find the bug, fix it, and submit.

```
run_tests -> read_file -> apply_fix -> submit_solution -> SUCCESS
```

Maximum 10 steps per episode. Reward is computed only at submission.

---

### Difficulty Levels

| Level | Name | Description |
|---|---|---|
| 1 | Syntax Bug | Single file, obvious bug |
| 2 | Logic Bug | Subtle logic issue |
| 3 | Multi-File Bug | Cross-file debugging |
| 4 | Boss Level | Misleading test traps |

---

### Agent Tools

| Tool | Purpose |
|---|---|
| run_tests | Run pytest and see failures |
| read_file | Read source files |
| search_codebase | Search across files |
| apply_fix | Apply minimal patch |
| submit_solution | End episode and score |

---

### Safety Constraints

- Cannot modify test files
- Cannot delete files
- No system access
- Sandboxed execution with 30 second timeout

---

## Reward Design

Fully programmatic rewards with no LLM judge anywhere in the pipeline.

```
Total Reward = 0.45 * R1 + 0.20 * R2 + 0.20 * R3 + 0.15 * R4
```

| Signal | Meaning |
|---|---|
| R1 | Fix correctness |
| R2 | Efficiency |
| R3 | No regression |
| R4 | Minimal change |

---

## Training Results

| Metric | Before | After |
|---|---|---|
| Mean Reward | 0.3723 | 0.8225 |
| Tool Accuracy | ~50% | 100% |
| Solve Rate | 0% | Learned sequence |
| Steps Used | 10 | 4 |

---

### Learned Behavior

```
run_tests -> read_file -> apply_fix -> submit_solution
```

The untrained model loops on search_codebase, outputs prose instead of JSON, and uses all 10 steps without ever calling submit_solution. After two rounds of GRPO training the agent learns the correct 4-step sequence consistently.

---

## Reward Curve

Training reward improved steadily from 0.26 to 0.82 over 200 total training steps across two focused stages.

---

## Smoke Test Result

```
r1_test_completion    : 1.0000
r2_efficiency         : 0.8800
r3_anti_regression    : 1.0000
r4_surgical_precision : 1.0000
total                 : 0.9760

[PASS] done           == True
[PASS] solved         == True
[PASS] reward         == 0.9760  (>= 0.7)
[PASS] r1_completion  == 1.0

Pipeline is wired correctly. Ready for GRPO training.
```

---

## Theme

**Primary: World Modeling (3.1)**

The agent maintains a dynamic understanding of the codebase across steps using tools. It tracks what it has examined, what it found, and what remains unknown before acting.

**Secondary: Long-Horizon Planning**

The agent must plan across multiple steps before receiving any reward. Reward is delayed until submission.

---

## Repository Structure

```
DebugQuest/
├── data/
├── server/
├── env.py
├── tools.py
├── rewards.py
├── models.py
├── dataset.py
├── smoke_test.py
└── training/
```

---

## Quick Start

### Run smoke test

```bash
git clone https://github.com/Jagrit3500/DebugQuest.git
cd DebugQuest/DebugQuest
pip install openenv-core pytest pydantic
python smoke_test.py
```

### Run server

```bash
pip install -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

---

## Links

- HF Space: https://huggingface.co/spaces/Jagritjd/debug_quest
- GitHub: https://github.com/Jagrit3500/DebugQuest
- Training Notebook: https://www.kaggle.com/code/jagrit3500/debugquest-grpo-training

---

## OpenEnv Validation

```
openenv validate
[OK] debug_quest: Ready for multi-mode deployment
```