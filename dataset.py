from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from models import BugCase


DEFAULT_CASES_DIR = Path(__file__).resolve().parent / "data" / "cases"


class BugRepository:
    """
    Loads Debug Quest bug cases from JSON files.

    Real bug cases live in data/cases/.
    Add new cases by adding JSON files, not by changing environment logic.
    """

    def __init__(self, cases_dir: Path | str = DEFAULT_CASES_DIR):
        self.cases_dir = Path(cases_dir)
        self._cases: List[BugCase] = []
        self.reload()

    def reload(self) -> None:
        self._cases.clear()

        if not self.cases_dir.exists():
            raise RuntimeError(f"Cases directory not found: {self.cases_dir}")

        for path in sorted(self.cases_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)

            self._cases.append(BugCase(**raw))

        if not self._cases:
            raise RuntimeError(f"No bug case JSON files found in {self.cases_dir}")

    def all_cases(self) -> List[BugCase]:
        return list(self._cases)

    def available_levels(self) -> List[int]:
        return sorted({case.level for case in self._cases})

    def get_by_level(self, level: int) -> List[BugCase]:
        return [case for case in self._cases if case.level == level]

    def sample_case(
        self,
        level: int,
        rng: Optional[random.Random] = None,
        verify: bool = True,
    ) -> BugCase:
        pool = self.get_by_level(level)

        if not pool:
            raise ValueError(f"No bug cases available for level {level}")

        picker = rng or random
        base_case = picker.choice(pool)

        if verify:
            ok, msg = verify_case(base_case)
            if not ok:
                raise RuntimeError(
                    f"Bug case verification failed for {base_case.case_id}: {msg}"
                )

        data = base_case.model_dump()
        data["case_id"] = f"{base_case.case_id}-{uuid.uuid4().hex[:8]}"
        return BugCase(**data)


def write_case_to_workspace(case: BugCase, workspace: Path | str) -> Path:
    """
    Writes a BugCase into an isolated workspace.

    The environment must mutate only this workspace,
    never the original JSON files.
    """

    workspace = Path(workspace)

    if workspace.exists():
        shutil.rmtree(workspace)

    workspace.mkdir(parents=True, exist_ok=True)

    for filename, content in case.files.items():
        path = workspace / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    test_path = workspace / case.test_filename
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text(case.test_file, encoding="utf-8")

    return workspace


def run_pytest(workspace: Path | str, test_filename: str) -> Tuple[int, str]:
    """
    Runs pytest inside a workspace.

    Returns:
        return_code, combined_output
    """

    workspace = Path(workspace)

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", test_filename, "-q", "--tb=short"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=30,
    )

    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode, output.strip()


def verify_case(case: BugCase) -> Tuple[bool, str]:
    """
    Verifies that:
    1. buggy code fails tests
    2. ground-truth fixed code passes tests
    """

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)

        write_case_to_workspace(case, workspace)
        buggy_code, _ = run_pytest(workspace, case.test_filename)

        if buggy_code == 0:
            return False, "Buggy code unexpectedly passes all tests"

        fixed_files = dict(case.files)
        fixed_files.update(case.ground_truth_fix)

        fixed_case = BugCase(
            case_id=case.case_id,
            level=case.level,
            files=fixed_files,
            test_file=case.test_file,
            test_filename=case.test_filename,
            ground_truth_fix=case.ground_truth_fix,
            changed_lines=case.changed_lines,
            bug_description=case.bug_description,
        )

        write_case_to_workspace(fixed_case, workspace)
        fixed_code, fixed_output = run_pytest(workspace, case.test_filename)

        if fixed_code != 0:
            return False, f"Ground-truth fixed code still fails:\n{fixed_output}"

        return True, "OK"


def generate_case(
    level: int,
    verify: bool = True,
    rng: Optional[random.Random] = None,
    cases_dir: Path | str = DEFAULT_CASES_DIR,
) -> BugCase:
    repo = BugRepository(cases_dir)
    return repo.sample_case(level=level, rng=rng, verify=verify)


def generate_curriculum_case(
    episode_num: int,
    reward_history: List[float],
    verify: bool = False,
    rng: Optional[random.Random] = None,
    cases_dir: Path | str = DEFAULT_CASES_DIR,
) -> BugCase:
    repo = BugRepository(cases_dir)
    levels = repo.available_levels()

    current_level = levels[0]

    if reward_history:
        recent = reward_history[-5:]
        avg_reward = sum(recent) / len(recent)

        if avg_reward >= 0.6:
            level_index = min(len(levels) - 1, episode_num // 20)
            current_level = levels[level_index]

    return repo.sample_case(level=current_level, verify=verify, rng=rng)


def bootstrap_cases_from_templates(
    output_dir: Path | str = DEFAULT_CASES_DIR,
    overwrite: bool = False,
) -> None:
    """
    Generates JSON cases from dataset_templates.py.

    This is used only once to populate data/cases/.
    After that, the environment loads from JSON files.
    """

    try:
        from debug_quest.dataset_templates import TEMPLATES
    except ModuleNotFoundError:
        from dataset_templates import TEMPLATES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, tmpl in enumerate(TEMPLATES, start=1):
        case_id = f"l{tmpl.level}-{index:03d}"
        out_path = output_dir / f"{case_id}.json"

        if out_path.exists() and not overwrite:
            print(f"SKIP existing: {out_path.name}")
            continue

        case_dict = {
            "case_id": case_id,
            "level": tmpl.level,
            "files": tmpl.files,
            "test_file": tmpl.test_file,
            "test_filename": tmpl.test_filename,
            "ground_truth_fix": tmpl.ground_truth_fix,
            "changed_lines": tmpl.changed_lines,
            "bug_description": tmpl.bug_description,
        }

        out_path.write_text(
            json.dumps(case_dict, indent=2),
            encoding="utf-8",
        )

        print(f"Generated: {out_path.name}")


def verify_all_cases(cases_dir: Path | str = DEFAULT_CASES_DIR) -> None:
    repo = BugRepository(cases_dir)

    print("=" * 60)
    print("Debug Quest — Case Verification")
    print("=" * 60)

    all_ok = True

    for case in repo.all_cases():
        ok, msg = verify_case(case)
        tag = "OK" if ok else "FAIL"
        print(f"[{tag}] L{case.level} | {case.case_id} | {case.bug_description}")

        if not ok:
            print(msg)
            all_ok = False

    if not all_ok:
        raise SystemExit(1)

    print("All bug cases verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug Quest dataset tools")

    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Generate JSON cases from dataset_templates.py",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON case files during bootstrap",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify all JSON cases",
    )

    args = parser.parse_args()

    if args.bootstrap:
        bootstrap_cases_from_templates(overwrite=args.overwrite)

    if args.verify:
        verify_all_cases()

    if not args.bootstrap and not args.verify:
        print("Use --bootstrap, --verify, or both.")