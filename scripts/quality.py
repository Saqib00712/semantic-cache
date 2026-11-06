"""One command for the full quality gate: lint, format check, tests, evaluation. Exit code != 0 on any failure."""

import subprocess
import sys

STEPS = [
    ["ruff", "check", "."],
    ["ruff", "format", "--check", "."],
    [sys.executable, "-m", "pytest"],
    [sys.executable, "-m", "evaluation.run_eval", "--out", "evaluation/report.md"],
]

for step in STEPS:
    print("$", " ".join(step), flush=True)
    if subprocess.call(step) != 0:
        sys.exit(1)
print("quality gate passed")
