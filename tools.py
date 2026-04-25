from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


MAX_SEARCH_RESULTS = 20


def _safe_workspace_path(workspace: Path | str, relative: str) -> Path:
    workspace = Path(workspace).resolve()
    resolved = (workspace / relative).resolve()

    if workspace != resolved and workspace not in resolved.parents:
        raise ValueError(f"Path traversal detected: {relative}")

    return resolved


def _is_test_file(path: Path) -> bool:
    name = path.name
    return name.startswith("test_") or name.endswith("_test.py")


def _count_changed_lines(original: str, patched: str) -> int:
    original_lines = original.splitlines()
    patched_lines = patched.splitlines()

    max_len = max(len(original_lines), len(patched_lines))
    changed = 0

    for i in range(max_len):
        before = original_lines[i] if i < len(original_lines) else None
        after = patched_lines[i] if i < len(patched_lines) else None

        if before != after:
            changed += 1

    return changed


def _normalize(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "success": bool(result.get("success", False)),
        "output": str(result.get("output", "")),
        "metadata": {
            key: value
            for key, value in result.items()
            if key not in {"success", "output"}
        },
    }


def run_tests(
    workspace: Path | str,
    test_filename: str,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Run pytest inside the current workspace.
    Use this first to see which tests fail.
    """

    workspace = Path(workspace)
    test_path = _safe_workspace_path(workspace, test_filename)

    if not test_path.exists():
        return {
            "success": False,
            "output": f"Test file not found: {test_filename}",
            "tests_passed": 0,
            "tests_total": 0,
            "failing_tests": [],
        }

    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(test_path),
                "-v",
                "--tb=short",
                "--no-header",
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": f"Test run timed out after {timeout} seconds.",
            "tests_passed": 0,
            "tests_total": 0,
            "failing_tests": [],
        }

    combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()

    passed_tests: List[str] = []
    failing_tests: List[str] = []

    for line in combined.splitlines():
        stripped = line.strip()

        if " PASSED" in stripped:
            passed_tests.append(stripped.split(" PASSED")[0].strip())
        elif " FAILED" in stripped:
            failing_tests.append(stripped.split(" FAILED")[0].strip())
        elif " ERROR" in stripped:
            failing_tests.append(stripped.split(" ERROR")[0].strip())

    tests_total = len(passed_tests) + len(failing_tests)
    tests_passed = len(passed_tests)

    return {
        "success": proc.returncode == 0,
        "output": combined,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "failing_tests": failing_tests,
    }


def read_file(workspace: Path | str, file_path: str) -> Dict[str, Any]:
    """
    Read a source file from the workspace.
    Use this before applying a fix so you can copy exact old_code.
    """

    try:
        resolved = _safe_workspace_path(workspace, file_path)
    except ValueError as exc:
        return {"success": False, "output": str(exc), "content": ""}

    if not resolved.exists():
        return {"success": False, "output": f"File not found: {file_path}", "content": ""}

    if not resolved.is_file():
        return {"success": False, "output": f"Path is not a file: {file_path}", "content": ""}

    try:
        raw = resolved.read_text(encoding="utf-8")
    except Exception as exc:
        return {"success": False, "output": f"Could not read file: {exc}", "content": ""}

    numbered = "\n".join(
        f"{index + 1:4d} | {line}" for index, line in enumerate(raw.splitlines())
    )

    return {
        "success": True,
        "output": numbered,
        "content": raw,
        "file_path": file_path,
        "line_count": len(raw.splitlines()),
    }


def search_codebase(
    workspace: Path | str,
    query: str,
    max_results: int = MAX_SEARCH_RESULTS,
) -> Dict[str, Any]:
    """
    Search Python files for a query string.
    Use this when you do not know which file contains the bug.
    """

    workspace = Path(workspace)

    if not query or not query.strip():
        return {"success": False, "output": "Search query cannot be empty.", "matches": []}

    query_lower = query.strip().lower()
    matches: List[Dict[str, Any]] = []

    for file in sorted(workspace.rglob("*.py")):
        if "__pycache__" in file.parts:
            continue

        try:
            _safe_workspace_path(workspace, str(file.relative_to(workspace)))
            lines = file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue

        rel = str(file.relative_to(workspace))

        for lineno, line in enumerate(lines, start=1):
            if query_lower in line.lower():
                matches.append(
                    {
                        "file": rel,
                        "line": lineno,
                        "text": line.rstrip(),
                    }
                )

                if len(matches) >= max_results:
                    break

        if len(matches) >= max_results:
            break

    if not matches:
        output = f"No results found for query: {query}"
    else:
        rows = [f"{m['file']}:{m['line']} -> {m['text']}" for m in matches]
        output = f"Found {len(matches)} match(es) for '{query}':\n" + "\n".join(rows)

    return {
        "success": True,
        "output": output,
        "matches": matches,
        "match_count": len(matches),
    }


def apply_fix(
    workspace: Path | str,
    file_path: str,
    old_code: str,
    new_code: str,
) -> Dict[str, Any]:
    """
    Apply a surgical patch by replacing exact old_code with new_code.
    Cannot modify tests. old_code must appear exactly once.
    """

    try:
        resolved = _safe_workspace_path(workspace, file_path)
    except ValueError as exc:
        return {"success": False, "output": str(exc), "lines_changed": 0}

    if _is_test_file(resolved):
        return {
            "success": False,
            "output": f"Forbidden: cannot modify test file '{file_path}'.",
            "lines_changed": 0,
        }

    if not new_code or not new_code.strip():
        return {
            "success": False,
            "output": "Forbidden: new_code is empty.",
            "lines_changed": 0,
        }

    if not resolved.exists():
        return {
            "success": False,
            "output": f"File not found: {file_path}",
            "lines_changed": 0,
        }

    try:
        original = resolved.read_text(encoding="utf-8")
    except Exception as exc:
        return {"success": False, "output": f"Could not read file: {exc}", "lines_changed": 0}

    occurrences = original.count(old_code)

    if occurrences == 0:
        return {
            "success": False,
            "output": "old_code not found. Use read_file first and copy exact text.",
            "lines_changed": 0,
        }

    if occurrences > 1:
        return {
            "success": False,
            "output": f"old_code appears {occurrences} times. Make patch more specific.",
            "lines_changed": 0,
        }

    patched = original.replace(old_code, new_code, 1)

    if resolved.suffix == ".py":
        try:
            ast.parse(patched)
        except SyntaxError as exc:
            return {
                "success": False,
                "output": f"Syntax error in patched code: {exc}",
                "lines_changed": 0,
            }

    lines_changed = _count_changed_lines(original, patched)
    resolved.write_text(patched, encoding="utf-8")

    return {
        "success": True,
        "output": f"Patch applied to {file_path}. {lines_changed} line(s) changed.",
        "lines_changed": lines_changed,
        "file_path": file_path,
    }


def submit_solution(
    workspace: Path | str,
    test_filename: str,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Submit the final answer.
    This runs pytest one final time and marks whether the episode is solved.
    """

    final = run_tests(workspace, test_filename, timeout=timeout)

    solved = (
        final["success"]
        and final["tests_total"] > 0
        and final["tests_passed"] == final["tests_total"]
    )

    return {
        "success": True,
        "solved": solved,
        "output": final["output"],
        "tests_passed": final["tests_passed"],
        "tests_total": final["tests_total"],
        "failing_tests": final["failing_tests"],
    }


def get_tool_definitions() -> str:
    """
    Return tool descriptions for prompt injection.
    """

    tools = [
        ("run_tests", run_tests),
        ("read_file", read_file),
        ("search_codebase", search_codebase),
        ("apply_fix", apply_fix),
        ("submit_solution", submit_solution),
    ]

    return "\n\n".join(
        f"{name}: {(func.__doc__ or '').strip()}" for name, func in tools
    )


def dispatch_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    workspace: Path | str,
    test_filename: str,
) -> Dict[str, Any]:
    """
    Route one agent tool call and always return:
    {
      "success": bool,
      "output": str,
      "metadata": dict
    }
    """

    name = tool_name.strip().lower()
    workspace = Path(workspace)

    try:
        if name == "run_tests":
            result = run_tests(workspace, test_filename)

        elif name == "read_file":
            file_path = tool_args.get("file_path", "")
            if not file_path:
                result = {
                    "success": False,
                    "output": "read_file requires argument: file_path",
                }
            else:
                result = read_file(workspace, file_path)

        elif name == "search_codebase":
            query = tool_args.get("query", "")
            if not query:
                result = {
                    "success": False,
                    "output": "search_codebase requires argument: query",
                }
            else:
                result = search_codebase(workspace, query)

        elif name == "apply_fix":
            file_path = tool_args.get("file_path", "")
            old_code = tool_args.get("old_code", "")
            new_code = tool_args.get("new_code", "")

            missing = []
            if not file_path:
                missing.append("file_path")
            if not old_code:
                missing.append("old_code")
            if not new_code:
                missing.append("new_code")

            if missing:
                result = {
                    "success": False,
                    "output": f"apply_fix missing required argument(s): {', '.join(missing)}",
                }
            else:
                result = apply_fix(
                    workspace=workspace,
                    file_path=file_path,
                    old_code=old_code,
                    new_code=new_code,
                )

        elif name == "submit_solution":
            result = submit_solution(workspace, test_filename)

        else:
            result = {
                "success": False,
                "output": (
                    f"Unknown tool: {tool_name}. "
                    "Available: run_tests, read_file, search_codebase, apply_fix, submit_solution."
                ),
            }

        return _normalize(result)

    except Exception as exc:
        return _normalize(
            {
                "success": False,
                "output": f"Tool execution crashed: {exc}",
            }
        )