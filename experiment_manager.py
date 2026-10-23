#!/usr/bin/env python3
"""
experiment_manager.py — CLI for HPC energy characterization experiments.

Usage:
    python experiment_manager.py grid                          # Print full experiment list
    python experiment_manager.py grid --config default         # Filter by config
    python experiment_manager.py generate                      # Create all dirs + configs
    python experiment_manager.py generate --config default     # Selective generate
    python experiment_manager.py submit EXP_ID                 # Submit one experiment
    python experiment_manager.py submit --config default --dry-run
    python experiment_manager.py generate --grid probe_sn_size_sweep   # One grid from grids.yaml
    python experiment_manager.py submit --config default --no-monitor  # Skip EAR DB monitor
    python experiment_manager.py status                        # Show experiment table

Run from ~/experiments/.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

USERNAME = os.environ.get("USER", "dkravchuk")

from manage.config_generator import generate_experiment
from manage.grid import expand_all_grids, make_id, make_exp_path

HOME            = "/gpfs/home5/dkravchuk"
EXPERIMENTS_DIR = Path(f"{HOME}/experiments")
META_CSV        = EXPERIMENTS_DIR / "metadata.csv"

META_FIELDS = [
    "experiment_id", "stage", "dataset", "config", "n_tasks", "tasks_per_node", "tasks_per_job", "cpus_per_task",
    "replicate", "partition", "slurm_jobid", "nodes",
    "total_energy_kj", "total_runtime_sec", "mean_power_w", "mean_pkg_power_w",
    "status",
]


# === METADATA HELPERS ===

def _read_meta() -> tuple[list[str], list[dict]]:
    if not META_CSV.exists():
        return META_FIELDS, []
    with open(META_CSV, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or META_FIELDS), list(reader)


def _write_meta(rows: list[dict]):
    with open(META_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _append_meta(row: dict):
    exists = META_CSV.exists()
    with open(META_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _meta_index(rows: list[dict]) -> dict[str, dict]:
    return {r["experiment_id"]: r for r in rows}


# === FILTER HELPERS ===

def _filter_exps(exps: list[dict], args: list[str]) -> list[dict]:
    """Filter experiment list by --grid, --stage, --dataset, --config, --n_tasks flags."""
    filters = {}
    i = 0
    while i < len(args):
        if args[i] in ("--grid", "--stage", "--dataset", "--config", "--n_tasks"):
            filters[args[i].lstrip("-")] = args[i + 1]
            i += 2
        else:
            i += 1

    result = exps
    for key, value in filters.items():
        if key == "n_tasks":
            result = [e for e in result if str(e[key]) == value]
        else:
            result = [e for e in result if e[key] == value]
    return result


def _is_dry_run(args: list[str]) -> bool:
    return "--dry-run" in args


# === COMMANDS ===

def cmd_grid(args: list[str]):
    exps = _filter_exps(expand_all_grids(), args)
    print(f"\n{'#':<4} {'Experiment ID':<70} {'Tasks':>5} {'TPN':>4} {'CPUs':>4} {'Partition':<8}")
    print("=" * 101)
    for i, exp in enumerate(exps, 1):
        tpn = exp.get("tasks_per_node", "")
        print(f"{i:<4} {make_id(exp):<70} {exp['n_tasks']:>5} {str(tpn):>4} {exp['cpus_per_task']:>4} {exp['partition']:<8}")
    print(f"\nTotal: {len(exps)} experiments")


def cmd_generate(args: list[str]):
    exps = _filter_exps(expand_all_grids(), args)
    _, meta_rows = _read_meta()
    existing_ids = {r["experiment_id"] for r in meta_rows}

    created = 0
    skipped = 0
    for exp in exps:
        exp_id   = make_id(exp)
        exp_path = generate_experiment(exp, EXPERIMENTS_DIR)

        if exp_path is None:
            skipped += 1
            continue

        print(f"  CREATED  {exp_id}")

        if exp_id not in existing_ids:
            _append_meta({
                "experiment_id":  exp_id,
                "stage":          exp["stage"],
                "dataset":        exp["dataset"],
                "config":         exp["config"],
                "n_tasks":        exp["n_tasks"],
                "tasks_per_node": exp.get("tasks_per_node", ""),
                "tasks_per_job":  exp.get("tasks_per_job", ""),
                "cpus_per_task":  exp["cpus_per_task"],
                "replicate":      exp["replicate"],
                "partition":      exp["partition"],
                "status":         "generated",
            })
            existing_ids.add(exp_id)

        created += 1

    print(f"\nCreated: {created}  |  Already existed: {skipped}")


def _parse_batch(args: list[str]) -> int | None:
    """Return --batch N value, or None if not set."""
    for i, a in enumerate(args):
        if a == "--batch" and i + 1 < len(args):
            return int(args[i + 1])
    return None


def cmd_submit(args: list[str]):
    dry     = _is_dry_run(args)
    monitor = "--no-monitor" not in args
    batch   = _parse_batch(args)

    # Submit by explicit ID (args that are not flags and not values of flags)
    flag_value_positions = set()
    for i, a in enumerate(args):
        if a.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            flag_value_positions.add(i + 1)
    positional = [a for i, a in enumerate(args) if not a.startswith("--") and i not in flag_value_positions]
    if positional:
        exp_id = positional[0]
        exps   = expand_all_grids()
        exp    = next((e for e in exps if make_id(e) == exp_id), None)
        if exp is None:
            print(f"Unknown experiment: {exp_id}")
            sys.exit(1)
        _submit_one(exp, dry, monitor)
        return

    # Submit by filter flags, optionally limited to --batch N pending experiments
    exps = _filter_exps(expand_all_grids(), args)
    if not exps:
        print("No experiments match the given filters.")
        return

    _, meta_rows = _read_meta()
    done_ids = {r["experiment_id"] for r in meta_rows if r.get("status") in ("submitted", "completed")}
    exps = [e for e in exps if make_id(e) not in done_ids]

    if batch is not None:
        # Only count experiments that are actually ready to run (run.sh exists).
        capped = []
        for e in exps:
            if len(capped) >= batch:
                break
            if (make_exp_path(e, EXPERIMENTS_DIR) / "run.sh").exists():
                capped.append(e)
        exps = capped
        print(f"Batch of {len(exps)}:")

    if not exps:
        print("No pending experiments to submit (run generate first?).")
        return

    for exp in exps:
        _submit_one(exp, dry, monitor)


def _start_monitor(exp_path: Path) -> None:
    """Spawn an independent background process running ExperimentMonitor for exp_path."""
    runner = Path(__file__).parent / "manage" / "monitor_runner.py"
    db_path = EXPERIMENTS_DIR / "ear_monitor.db"
    proc = subprocess.Popen(
        [sys.executable, str(runner), str(exp_path), str(db_path), USERNAME],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    print(f"  MONITOR  {exp_path.name}  (pid={proc.pid}  log={exp_path}/ear_monitor.log)")


def _submit_one(exp: dict, dry: bool, monitor: bool = True):
    exp_id   = make_id(exp)
    exp_path = make_exp_path(exp, EXPERIMENTS_DIR)
    run_sh   = exp_path / "run.sh"

    if not run_sh.exists():
        print(f"  MISSING  {exp_id}  (run generate first)")
        return

    if dry:
        print(f"  DRY-RUN  bash {run_sh}")
        return

    print(f"  SUBMIT   {exp_id}")
    subprocess.run(["bash", str(run_sh)], check=False)
    if monitor:
        _start_monitor(exp_path)

    # Mark as submitted in metadata
    _, rows = _read_meta()
    idx     = _meta_index(rows)
    if exp_id in idx:
        idx[exp_id]["status"] = "submitted"
        _write_meta(list(idx.values()))


def cmd_status(args: list[str]):
    _, rows = _read_meta()
    if not rows:
        print("No experiments in metadata.csv yet.")
        return

    # Apply same filters
    exps_all = expand_all_grids()
    filtered_ids = {make_id(e) for e in _filter_exps(exps_all, args)}
    rows = [r for r in rows if r["experiment_id"] in filtered_ids] if filtered_ids != {make_id(e) for e in exps_all} else rows

    print(f"\n{'Experiment ID':<65} {'Tasks':>5} {'Energy':>10} {'Power':>8} {'Runtime':>9} {'Status':<12}")
    print("=" * 115)
    for r in rows:
        e = r.get("total_energy_kj", "")
        p = r.get("mean_power_w", "")
        t = r.get("total_runtime_sec", "")
        s = r.get("status", "?")
        print(
            f"  {r['experiment_id']:<63}"
            f" {r.get('n_tasks','?'):>5}"
            f" {(e+' kJ') if e else '—':>10}"
            f" {(p+' W') if p else '—':>8}"
            f" {(t+' s') if t else '—':>9}"
            f" {s:<12}"
        )
    print(f"\nTotal: {len(rows)}")


# === ENTRY POINT ===

COMMANDS = {
    "grid":     cmd_grid,
    "generate": cmd_generate,
    "submit":   cmd_submit,
    "status":   cmd_status,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0)

    COMMANDS[sys.argv[1]](sys.argv[2:])
