"""
smoke_test.py — Debug Quest end-to-end smoke test
Validates the full pipeline:
    dataset → env.reset() → tools → env.step() → rewards

Run from inside debug_quest/:
    python smoke_test.py

Expected result:
    DONE: True
    solved: True
    reward: ~0.9+
"""

from env import DebugQuestEnv
from models import DebugQuestAction, DebugTool

# ---------------------------------------------------------------------------
# 1. Boot environment
# ---------------------------------------------------------------------------

env = DebugQuestEnv()
obs = env.reset(level=1)

print("=" * 60)
print("INITIAL OBSERVATION")
print("=" * 60)
print(obs.tool_output)
print()
print(f"tests_passed    : {obs.tests_passed}")
print(f"tests_total     : {obs.tests_total}")
print(f"failing_tests   : {obs.failing_tests}")
print(f"steps_remaining : {obs.steps_remaining}")

# ---------------------------------------------------------------------------
# 2. Hardcoded optimal trajectory for level 1
#    (off-by-one in range(len(items) + 1))
# ---------------------------------------------------------------------------

actions = [
    # Step 1 — see what's failing
    DebugQuestAction(
        tool_name=DebugTool.RUN_TESTS,
        tool_args={},
    ),

    # Step 2 — read the source file
    DebugQuestAction(
        tool_name=DebugTool.READ_FILE,
        tool_args={"file_path": "calculator.py"},
    ),

    # Step 3 — apply the fix
    DebugQuestAction(
        tool_name=DebugTool.APPLY_FIX,
        tool_args={
            "file_path": "calculator.py",
            "old_code": "for i in range(len(items) + 1):",
            "new_code": "for i in range(len(items)):",
        },
    ),

    # Step 4 — submit
    DebugQuestAction(
        tool_name=DebugTool.SUBMIT_SOLUTION,
        tool_args={},
    ),
]

# ---------------------------------------------------------------------------
# 3. Step through the environment
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("EPISODE TRAJECTORY")
print("=" * 60)

for i, action in enumerate(actions, start=1):
    obs, reward, done, info = env.step(action)

    print(f"\n─── Step {i} | tool: {action.tool_name.value} ───")
    print(f"  output (first 400 chars):\n{obs.tool_output[:400]}")
    print(f"  tool_success    : {obs.tool_success}")
    print(f"  tests_passed    : {obs.tests_passed} / {obs.tests_total}")
    print(f"  steps_remaining : {obs.steps_remaining}")
    print(f"  reward          : {reward}")
    print(f"  done            : {done}")

    if done:
        print()
        print("=" * 60)
        print("EPISODE RESULT")
        print("=" * 60)
        print(f"  solved          : {info.get('solved')}")
        print(f"  reward          : {reward:.4f}")
        print()
        breakdown = info.get("reward_breakdown", {})
        for key, val in breakdown.items():
            print(f"  {key:<26}: {val:.4f}")
        break

# ---------------------------------------------------------------------------
# 4. Assertions — if these pass, the pipeline is wired correctly
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("ASSERTIONS")
print("=" * 60)

assert done,                          "Episode should be done after submit_solution"
assert info.get("solved") is True,    "Level 1 with correct fix should be solved"
assert reward >= 0.7,                 f"Expected reward >= 0.7, got {reward:.4f}"
assert info["reward_breakdown"]["r1_test_completion"] == 1.0, \
    "All tests should pass after correct fix"

print("  [PASS] done           == True")
print("  [PASS] solved         == True")
print(f"  [PASS] reward         == {reward:.4f}  (>= 0.7)")
print("  [PASS] r1_completion  == 1.0")
print()
print("Pipeline is wired correctly. Ready for GRPO training.")