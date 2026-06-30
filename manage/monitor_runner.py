"""
Background entry point launched by experiment_manager after each sbatch submission.

Invoked as:
    python monitor_runner.py EXP_PATH DB_PATH USERNAME
"""

import logging
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 4:
        print("usage: monitor_runner.py EXP_PATH DB_PATH USERNAME", file=sys.stderr)
        sys.exit(1)

    exp_path, db_path, username = sys.argv[1], sys.argv[2], sys.argv[3]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filename=str(Path(exp_path) / "ear_monitor.log"),
        filemode="a",
    )

    # ear_monitor lives one level above experiments/
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    try:
        from ear_monitor import ExperimentMonitor  # noqa: E402

        ExperimentMonitor(
            experiment_path=exp_path,
            username=username,
            db_path=db_path,
        ).start()
    except Exception:
        logging.exception("monitor crashed")


if __name__ == "__main__":
    main()
