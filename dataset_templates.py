from dataclasses import dataclass
from typing import Dict, List


@dataclass
class BugTemplate:
    level: int
    bug_description: str
    files: Dict[str, str]
    test_file: str
    test_filename: str
    ground_truth_fix: Dict[str, str]
    changed_lines: Dict[str, List[int]]


TEMPLATES = [
    BugTemplate(
        level=1,
        bug_description="Off-by-one bug in sum_list: loop iterates one index past the end of the list.",
        files={
            "calculator.py": (
                "def sum_list(items):\n"
                "    total = 0\n"
                "    for i in range(len(items) + 1):\n"
                "        total += items[i]\n"
                "    return total\n"
                "\n"
                "def average(items):\n"
                "    if not items:\n"
                "        return 0\n"
                "    return sum_list(items) / len(items)\n"
            )
        },
        test_file=(
            "from calculator import sum_list, average\n"
            "\n"
            "def test_sum_list_basic():\n"
            "    assert sum_list([1, 2, 3]) == 6\n"
            "\n"
            "def test_sum_list_single():\n"
            "    assert sum_list([5]) == 5\n"
            "\n"
            "def test_average_basic():\n"
            "    assert average([2, 4, 6]) == 4.0\n"
            "\n"
            "def test_average_empty():\n"
            "    assert average([]) == 0\n"
        ),
        test_filename="test_calculator.py",
        ground_truth_fix={
            "calculator.py": (
                "def sum_list(items):\n"
                "    total = 0\n"
                "    for i in range(len(items)):\n"
                "        total += items[i]\n"
                "    return total\n"
                "\n"
                "def average(items):\n"
                "    if not items:\n"
                "        return 0\n"
                "    return sum_list(items) / len(items)\n"
            )
        },
        changed_lines={"calculator.py": [3]},
    )
]