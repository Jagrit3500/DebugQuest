---
license: mit
tags:
- reinforcement-learning
- debugging
- openenv
- grpo
- code-repair
- unsloth
- qwen2.5
---

# Debug Quest Agent — GRPO Trained Debugging Agent

Trained with GRPO reinforcement learning inside the Debug Quest environment. Debug Quest is an investigation-first autonomous debugging environment where an LLM agent must find and fix bugs in Python codebases without being told where the bug is.

## The Problem

Current LLMs guess fixes without investigating. They break passing tests. They rewrite entire files when one line needed changing. This model is trained to investigate first, then fix surgically. The agent receives a buggy codebase, uses tools to investigate, localizes the root cause, applies a minimal fix, and submits — with no hints about where the bug is.

## Training Results

| Metric | Untrained (Baseline) | After Training |
|---|---|---|
| Mean Reward | 0.3723 | 0.8225 |
| Tool Sequence Accuracy | ~50% | 100% (4/4) |
| Solve Rate | 0% | Improving |
| Avg Steps Used | 10 (always maxes out) | 4 |
| Valid JSON Rate | Inconsistent | Consistent |

Reward improved from 0.3723 to 0.8225, a 120% improvement. The untrained model wanders — it loops on search_codebase, outputs prose instead of JSON, uses all 10 steps, and never calls submit_solution. After two rounds of GRPO training the agent learns the correct sequence: run_tests → read_file → apply_fix → submit_solution.

## Training Details

- Base model: Qwen2.5-1.5B-Instruct (4-bit quantized via Unsloth)
- Method: GRPO reinforcement learning (TRL)
- Framework: OpenEnv + TRL + Unsloth
- Hardware: Kaggle T4 GPU (15.6 GB VRAM)
- Training rounds: 2 (120 steps stage 1, 80 steps stage 2)
- LoRA rank: 16, alpha 32, 18.4M trainable parameters

### Stage 1 — Full Sequence Training (120 steps)

Reward climbed from 0.41 to 0.67 over 120 steps. Tool accuracy after stage 1 was 2/4 — START and AFTER_FIX correct, AFTER_TESTS and AFTER_READ still weak.

### Stage 2 — Focused Correction (80 steps)

Targeted only the two weak stages. Reward climbed from 0.265 to 0.8225. Tool accuracy after stage 2 was 4/4 = 100%.

## Reward Design

Four fully programmatic reward functions. No LLM judge anywhere in the pipeline. Every signal is computed by running actual code.

Total Reward = 0.45 × R1 + 0.20 × R2 + 0.20 × R3 + 0.15 × R4

| Signal | Weight | What it teaches |
|---|---|---|
| R1 Test Completion | 0.45 | Fix the actual bug |
| R2 Efficiency | 0.20 | Investigate strategically, not randomly |
| R3 Anti-Regression | 0.20 | Do not break working code |
| R4 Surgical Precision | 0.15 | Change only what is needed |

## Smoke Test Result (Pre-Training Validation)

r1_test_completion    : 1.0000
r2_efficiency         : 0.8800
r3_anti_regression    : 1.0000
r4_surgical_precision : 1.0000
total                 : 0.9760[PASS] done           == True
[PASS] solved         == True
[PASS] reward         == 0.9760  (>= 0.7)
[PASS] r1_completion  == 1.0Pipeline is wired correctly. Ready for GRPO training.

## Environment — 4 Difficulty Levels

| Level | Name | Description |
|---|---|---|
| 1 | Syntax Bug | Single file, one obvious bug, 1–3 tests failing |
| 2 | Logic Bug | Single file, subtle logic error, 2–3 tests failing |
| 3 | Multi-File Bug | Bug hidden in a helper module, requires cross-file investigation |
| 4 | Boss Level | Hidden bug behind a misleading failing test designed to fool agents that do not truly investigate |

## Agent Tools

| Tool | Purpose |
|---|---|
| run_tests | Run pytest, see pass/fail with tracebacks |
| read_file | Read a file with line numbers |
| search_codebase | Search all Python files for a string |
| apply_fix | Apply a surgical patch (old_code must match exactly once) |
| submit_solution | Submit fix, end episode, trigger reward computation |

## Links

- Environment on HuggingFace Spaces: https://huggingface.co/spaces/Jagritjd/debug_quest
- Code on GitHub: https://github.com/Jagrit3500/DebugQuest
- Training Notebook: https://www.kaggle.com/code/jagrit3500/debugquest-grpo-training