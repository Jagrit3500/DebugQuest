---
title: Debug Quest
emoji: 🐛
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# Debug Quest Agent — GRPO Trained Debugging Agent

Trained with GRPO reinforcement learning inside the Debug Quest environment. Debug Quest is an investigation-first autonomous debugging environment where an LLM agent must find and fix bugs in Python codebases without being told where the bug is.

## The Idea

Most LLMs guess fixes without investigating. They break passing tests. They rewrite entire files when one line needed changing.

This agent is trained to follow a real debugging workflow:

run_tests → read_file → apply_fix → submit_solution

No guessing. No shortcuts. Proper investigation-first behavior.

## Training Results

| Metric | Untrained (Baseline) | After Training |
|---|---|---|
| Mean Reward | 0.3723 | 0.8225 |
| Tool Sequence Accuracy | ~50% | 100% (4/4) |
| Solve Rate | 0% | Improving |
| Avg Steps Used | 10 | 4 |
| Valid JSON Rate | Inconsistent | Consistent |

Reward improved from 0.3723 to 0.8225 (120% improvement). The untrained model loops, outputs prose, and never submits. After training, it executes structured debugging sequences correctly.

## Training Details

- Base model: Qwen2.5-1.5B-Instruct  
- Quantization: 4-bit (Unsloth)  
- Method: GRPO reinforcement learning (TRL)  
- LoRA: rank 16, alpha 32  
- Trainable params: 18.4M  
- Hardware: Kaggle T4 GPU (15.6 GB VRAM)  

## Training Strategy

Stage 1 (120 steps):  
Reward increased from 0.41 → 0.67. Model learned START and AFTER_FIX but failed AFTER_TESTS and AFTER_READ.

Stage 2 (80 steps — targeted correction):  
Focused only on weak stages. Reward increased from 0.265 → 0.8225. Tool accuracy reached 4/4 = 100%.

## Reward Design

Fully programmatic reward — no LLM judge.

Total Reward = 0.45 × R1 + 0.20 × R2 + 0.20 × R3 + 0.15 × R4

| Reward | Purpose |
|------|--------|
| R1 Test Completion | Fix the bug |
| R2 Efficiency | Solve faster |
| R3 Anti-regression | Do not break working tests |
| R4 Precision | Minimal code changes |

This prevents reward hacking and forces correct behavior.

## Environment

Four difficulty levels:

- L1: Syntax bug  
- L2: Logic bug  
- L3: Multi-file bug  
- L4: Hidden deceptive bug  

## Agent Tools

- run_tests → Run pytest and view failures  
- read_file → Read source files  
- search_codebase → Search across repo  
- apply_fix → Apply surgical patch  
- submit_solution → End episode and compute reward  

## Example Episode

```json
{"tool": "run_tests", "args": {}}
{"tool": "read_file", "args": {"file_path": "calculator.py"}}
{"tool": "apply_fix", "args": {"file_path": "calculator.py", "old_code": "range(len(items)+1)", "new_code": "range(len(items))"}}
{"tool": "submit_solution", "args": {}}

Reward: 0.976

Generalization (Honest Result)
Metric	Value
Avg Reward	0.6385
Solve Rate	0%

The model learned structured debugging behavior but has not fully generalized yet.

Reward Curve

See: training/reward_curve.png

Links
HuggingFace Environment: https://huggingface.co/spaces/Jagritjd/debug_quest
GitHub Repository: https://github.com/Jagrit3500/DebugQuest
Training Notebook: https://www.kaggle.com/code/jagrit3500/debugquest-grpo-training
Blog: https://github.com/Jagrit3500/DebugQuest/blob/main/BLOG.md
Built For

Meta x PyTorch OpenEnv Hackathon 2026

Key Insight

A small 1.5B model can learn structured debugging workflows using reinforcement learning when the reward function enforces correctness, efficiency, and precision together.


---

# ✅ Now just do:

```powershell
git add README.md
git commit -m "final readme"
git push origin main