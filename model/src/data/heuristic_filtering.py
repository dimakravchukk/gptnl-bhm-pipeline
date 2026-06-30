"""Heuristic filtering stage: load runtime and EAR energy data from raw
task logs and stats JSONs.

  load_hf_runs(log_base, corpus_registry, alpha_map, alpha_default=0.80)
      → DataFrame  (one row per (corpus, config, rep) job)

  load_hf_ear(log_base, corpus_registry, alpha_map, ...)
      → DataFrame  (one row per (corpus, config, rep) job, EAR power/energy)

  load_hf_dataset(...) → RuntimeDataset  (load_hf_runs + design arrays, bundled)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import parse_runtime, reader_block, strip_task_suffix, make_design_arrays, RuntimeDataset


def load_hf_runs(
    log_base: Path | str,
    corpus_registry: list[dict],
    alpha_map: dict[str, float],
    alpha_default: float = 0.80,
) -> pd.DataFrame:
    """Parse all heuristic-filtering run dirs and return one row per (corpus, config, rep) job.

    corpus_registry entries require keys: corpus, hw, n_tasks.
    Optional: is_anchor (default False), corpus_dir (overrides corpus as the directory name).

    Returned columns:
        corpus, corpus_dir, config, rep, max_non_alpha_ratio,
        runtime_s, total_docs, total_chars, n_tasks, is_anchor,
        alpha_cfg, log_rt_per_char
    """
    log_base = Path(log_base)
    dfs = []
    for entry in corpus_registry:
        corpus_dir   = entry.get("corpus_dir", entry["corpus"])
        corpus_label = strip_task_suffix(entry["corpus"])
        dfs.append(_parse_hf_corpus(
            log_base        = log_base,
            corpus_dir_name = corpus_dir,
            corpus_label    = corpus_label,
            hw              = entry["hw"],
            n_tasks         = entry["n_tasks"],
            is_anchor       = entry.get("is_anchor", False),
            alpha_map       = alpha_map,
        ))
    df = pd.concat(dfs, ignore_index=True)
    if not df.empty:
        df["alpha_cfg"]       = df["max_non_alpha_ratio"] - alpha_default
        df["log_rt_per_char"] = np.log(df["runtime_s"]) - np.log(df["total_chars"])
    return df


def _parse_hf_corpus(
    log_base: Path,
    corpus_dir_name: str,
    corpus_label: str,
    hw: str,
    n_tasks: int,
    is_anchor: bool,
    alpha_map: dict[str, float],
) -> pd.DataFrame:
    records = []
    skipped = {"regex": 0, "config": 0, "no_logs": 0, "parse": 0}
    base   = log_base / corpus_dir_name
    run_re = re.compile(rf"^(?P<config>.+)__{re.escape(hw)}__r(?P<rep>\d+)$")

    for run_dir in sorted(base.glob(f"*__{hw}__r[0-9]*")):
        m = run_re.match(run_dir.name)
        if not m:
            skipped["regex"] += 1
            continue
        config, rep = m.group("config"), int(m.group("rep"))
        if config not in alpha_map:
            skipped["config"] += 1
            continue

        log_paths  = sorted((run_dir / "output/logs/logs").glob("task_*.log"))
        stat_paths = sorted((run_dir / "output/logs/stats").glob("*.json"))

        if len(log_paths) != n_tasks or len(stat_paths) != n_tasks:
            skipped["no_logs"] += 1
            print(f"  [BAD RUN] {run_dir.name}: {len(log_paths)} logs, {len(stat_paths)} stats (expected {n_tasks})")
            continue

        total_runtime, total_chars, total_docs = 0.0, 0, 0
        ok = True
        for log_path, stat_path in zip(log_paths, stat_paths):
            rt = parse_runtime(log_path.read_text())
            if rt is None:
                print(f"  [NO RUNTIME] {log_path}")
                ok = False
                break
            blocks = json.loads(stat_path.read_text())
            rb = reader_block(blocks)
            if rb is None:
                print(f"  [NO READER] {stat_path}")
                ok = False
                break
            total_runtime += rt
            total_chars   += rb["stats"]["doc_len"]["total"]
            total_docs    += rb["stats"]["documents"]["total"]

        if not ok:
            skipped["parse"] += 1
            continue

        records.append({
            "corpus":              corpus_label,
            "corpus_dir":          corpus_dir_name,
            "config":              config,
            "rep":                 rep,
            "max_non_alpha_ratio": alpha_map[config],
            "runtime_s":           total_runtime,
            "total_docs":          total_docs,
            "total_chars":         total_chars,
            "n_tasks":             n_tasks,
            "is_anchor":           is_anchor,
        })

    print(f"{corpus_dir_name}: {len(records)} jobs, skipped {skipped}")
    return pd.DataFrame(records)


def load_hf_ear(
    log_base: Path | str,
    corpus_registry: list[dict],
    alpha_map: dict[str, float],
    alpha_default: float = 0.80,
    tasks_per_node: int = 8,
    min_valid_node_power: float = 50.0,
    ear_glob: str = "output/logs/emission/ear_db_*_apps.csv",
    ear_sep: str = ";",
    share_field: str = "DC_NODE_POWER_W",
    time_field: str = "TIME_SEC",
) -> pd.DataFrame:
    """Load EAR energy measurements from all heuristic-filtering run dirs.

    min_valid_node_power filters out physically implausible readings; the
    corresponding per-task threshold is min_valid_node_power / tasks_per_node,
    since EAR attributes node power proportionally across co-resident tasks
    under the full-node packing protocol.

    Returned columns: corpus, config, rep, alpha_cfg, E_measured_J, P_mean_W, n_ear_tasks
    """
    log_base = Path(log_base)
    dfs = []
    for entry in corpus_registry:
        corpus_dir   = entry.get("corpus_dir", entry["corpus"])
        corpus_label = strip_task_suffix(entry["corpus"])
        dfs.append(_load_hf_ear_corpus(
            log_base             = log_base,
            corpus_dir_name      = corpus_dir,
            corpus_label         = corpus_label,
            hw                   = entry["hw"],
            alpha_map            = alpha_map,
            alpha_default        = alpha_default,
            tasks_per_node       = tasks_per_node,
            min_valid_node_power = min_valid_node_power,
            ear_glob             = ear_glob,
            ear_sep              = ear_sep,
            share_field          = share_field,
            time_field           = time_field,
        ))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _load_hf_ear_corpus(
    log_base: Path,
    corpus_dir_name: str,
    corpus_label: str,
    hw: str,
    alpha_map: dict[str, float],
    alpha_default: float,
    tasks_per_node: int,
    min_valid_node_power: float,
    ear_glob: str,
    ear_sep: str,
    share_field: str,
    time_field: str,
) -> pd.DataFrame:
    records   = []
    run_re    = re.compile(rf"^(?P<config>.+)__{re.escape(hw)}__r(?P<rep>\d+)$")
    min_share = min_valid_node_power / tasks_per_node

    for run_dir in sorted((log_base / corpus_dir_name).glob(f"*__{hw}__r[0-9]*")):
        m = run_re.match(run_dir.name)
        if not m:
            continue
        config, rep = m.group("config"), int(m.group("rep"))
        if config not in alpha_map:
            continue

        ear_files = list(run_dir.glob(ear_glob))
        if not ear_files:
            print(f"  [NO EAR] {run_dir.name}")
            continue

        ear = pd.concat([pd.read_csv(f, sep=ear_sep) for f in ear_files], ignore_index=True)
        if share_field not in ear.columns or time_field not in ear.columns:
            print(f"  [MISSING COLS] {run_dir.name}")
            continue

        ear_valid = ear[ear[share_field] >= min_share].copy()
        n_dropped = len(ear) - len(ear_valid)
        if n_dropped:
            print(f"  [DROPPED {n_dropped} rows] {run_dir.name}")
        if len(ear_valid) == 0:
            print(f"  [ALL DROPPED] {run_dir.name}")
            continue

        ear_valid["E_task"] = ear_valid[share_field] * ear_valid[time_field]
        records.append({
            "corpus":       corpus_label,
            "config":       config,
            "rep":          rep,
            "alpha_cfg":    alpha_map[config] - alpha_default,
            "E_measured_J": float(ear_valid["E_task"].sum()),
            "P_mean_W":     float(ear_valid[share_field].mean()),
            "n_ear_tasks":  len(ear_valid),
        })

    return pd.DataFrame(records)


def load_hf_dataset(
    log_base: Path | str,
    corpus_registry: list[dict],
    alpha_map: dict[str, float],
    alpha_default: float = 0.80,
    corpora: list[str] | None = None,
) -> RuntimeDataset:
    """load_hf_runs + design arrays, bundled into a RuntimeDataset."""
    df = load_hf_runs(log_base, corpus_registry, alpha_map, alpha_default)
    corpora, corp2idx = make_design_arrays(df, corpora=corpora)
    return RuntimeDataset(df=df, corpora=corpora, corp2idx=corp2idx,
                          stage="heuristic_filtering")