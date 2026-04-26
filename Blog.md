# Teaching a 1.5B Model to Debug Code Like a Senior Engineer

## The Idea

Most LLMs, when given a broken codebase, do one of two things: guess a fix immediately without looking at the code, or explain the error philosophically without doing anything. Neither is useful.

I built Debug Quest — an RL environment where a language model must actually investigate a broken Python codebase, find the bug, and fix it surgically. No hints. No human in the loop. Just tools, tests, and a reward signal.

The model I trained is Qwen2.5-1.5B-Instruct at 4-bit precision, fine-tuned with GRPO on a T4 GPU for free on Kaggle. Reward went from 0.3723 to 0.8225. Tool sequence accuracy went from ~50% to 100%. Total training time was under 40 minutes.

## What the Agent Has to Do

Each episode starts with a buggy Python codebase. The agent does not know where the bug is. It has 10 steps and 5 tools.

run_tests shows which tests are failing and the full traceback. read_file reads any source file with line numbers. search_codebase greps across the whole project. apply_fix patches exactly one code block — old_code must match exactly once or the patch fails. submit_solution ends the episode and triggers reward computation.

The reward is only computed at submission. The agent gets nothing for trying. A perfect episode looks like this:
Step 1: {"tool": "run_tests", "args": {}}
Step 2: {"tool": "read_file", "args": {"file_path": "calculator.py"}}
Step 3: {"tool": "apply_fix", "args": {"file_path": "calculator.py", "old_code": "range(len(items) + 1)", "new_code": "range(len(items))"}}
Step 4: {"tool": "submit_solution", "args": {}}
→ Reward: 0.9760

## What the Untrained Model Does

Before training, the baseline Qwen2.5-1.5B scores a mean reward of 0.3723 across 5 episodes and solves 0 out of 5. It loops on search_codebase with the same query multiple times. It outputs prose instead of JSON. It describes the error instead of fixing it. It uses all 10 steps every single time without ever calling submit_solution. This is exactly the behaviour we want to eliminate.

## The Reward Signal

Four programmatic reward components, no LLM judge anywhere.

Total = 0.45 × R1 + 0.20 × R2 + 0.20 × R3 + 0.15 × R4

R1 measures whether the tests pass after the fix. R2 rewards solving in fewer steps — 1.0 minus 0.3 times steps used divided by max steps. R3 penalises breaking any test that was passing before the agent touched the codebase. R4 penalises changing more lines than necessary.

All four must be high simultaneously to score above 0.9. You cannot game one without the others pulling you down. An agent that passes all tests but uses all 10 steps loses on R2. An agent that fixes the target bug but breaks another test loses on R3. An agent that rewrites the whole file instead of changing one line loses on R4.

## Training: Two-Stage GRPO

I used GRPO via TRL and Unsloth on Kaggle with a free T4 GPU.

Stage 1 ran for 120 steps on a dataset covering all four stages of a debugging episode. Reward climbed from 0.41 to 0.67. After stage 1 the model correctly handled START and AFTER_FIX but still chose search_codebase when it should choose read_file or apply_fix.

Instead of running more general training I identified the two weak stages and built a focused dataset targeting only those. Stage 2 ran for 80 steps. Reward climbed from 0.265 to 0.8225. After stage 2, tool sequence accuracy was 4 out of 4.

Stage: START        Expected: run_tests        Predicted: run_tests   

Stage: AFTER_TESTS  Expected: read_file        Predicted: read_file 

Stage: AFTER_READ   Expected: apply_fix        Predicted: apply_fix        

Stage: AFTER_FIX    Expected: submit_solution  Predicted: submit_solution  

## Results Summary

| Metric | Before Training | After Training |
|---|---|---|
| Mean Reward | 0.3723 | 0.8225 |
| Tool Accuracy | ~50% | 100% |
| Solve Rate | 0% | Improving |
| Steps per Episode | Always 10 | 4 |

120% reward improvement from a model that had never been trained on debugging tasks, running on a free GPU, in under 40 minutes of total training time.

## Key Lessons

Two-stage training works better than one long run. Identifying which stages fail and targeting them specifically got the model from 50% to 100% accuracy faster than continuing the general training would have. When you know exactly where the model is wrong, a focused 80-step correction beats a diffuse 200-step continuation.

The reward signal matters more than the dataset size. Four independent reward components made the training signal much richer than a single pass/fail. The model could not exploit one signal without the others exposing the gap. R2 punishes laziness. R3 punishes recklessness. R4 punishes imprecision. R1 alone would have been gameable.

Small models can learn tool use. A 1.5B model with 18.4M trainable LoRA parameters learned a consistent 4-step debugging workflow from scratch on a free GPU. The key was the reward structure, not the model size.

## Links

- Environment: https://huggingface.co/spaces/Jagritjd/debug_quest
- Code: https://github.com/Jagrit3500/DebugQuest
- Training Notebook: https://www.kaggle.com/code/jagrit3500/debugquest-grpo-training

Built for the Meta x PyTorch OpenEnv Hackathon 2026.