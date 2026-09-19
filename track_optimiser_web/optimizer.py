import os
import subprocess
import sys
from pathlib import Path

REQUIRED_FILES = [
    "01_LINES.csv", "02_STATIONS.csv", "03_SECTORS.csv",
    "04_LOCATION_SUPPLY.csv", "05_BUFFER_LOCATION.csv",
    "06_PARAMETERS.csv", "07_PROJECT_DETAILS.csv", "08_ACTIVITY_DETAILS.csv",
]


def run_optimizer(input_dir: Path, scenario: str, timeout_seconds: int = 240):
    scenario = scenario.upper().strip()
    if scenario not in {"A", "B", "C"}:
        raise ValueError("Scenario must be A, B, or C.")

    missing = [name for name in REQUIRED_FILES if not (input_dir / name).exists()]
    if missing:
        raise ValueError("Missing required files: " + ", ".join(missing))

    solver_path = Path(__file__).with_name("solver.py")
    env = os.environ.copy()
    env["SCENARIO"] = scenario

    result = subprocess.run(
        [sys.executable, str(solver_path)],
        cwd=input_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )

    log = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    output_dir = input_dir / f"output_{scenario}"
    expected = [
        output_dir / "SCHEDULE_ACCESS.csv",
        output_dir / "SCHEDULE_OCCUPANCY.csv",
        output_dir / "RESULTS.csv",
    ]

    if result.returncode != 0:
        raise RuntimeError("Solver failed.\n\n" + log[-12000:])
    if not all(p.exists() for p in expected):
        raise RuntimeError("Solver did not produce all 3 CSV files.\n\n" + log[-12000:])

    return output_dir, log
