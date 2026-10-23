"""
grid.py — Experiment grid logic.

Data lives in YAML files next to this module:
  datasets.yaml          — short-name → absolute path
  configs/<stage>.yaml   — filter-config groups, keyed by group name
  grids.yaml             — grid specs; each references a configs_group by name

Public API (unchanged):
  DATASETS, expand_grid(spec), expand_all_grids(), make_id(exp), make_exp_path(exp, base_dir)

Backward-compat aliases still exported:
  STAGE2_CONFIGS, TOXIC_CONFIGS, STRING_NORM_CONFIGS, RO_ALPHA_SWEEP_CONFIGS, ALL_GRIDS
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import yaml

_HERE = Path(__file__).parent

# ── Load data ──────────────────────────────────────────────────────────────────

DATASETS: dict = yaml.safe_load((_HERE / "datasets.yaml").read_text())

# _CONFIGS[stage][group][config_name] → filter-override dict
_CONFIGS: dict = {}
for _f in (_HERE / "configs").glob("*.yaml"):
    _raw = yaml.safe_load(_f.read_text())
    # Strip anchor-only keys (start with _); they exist only to define YAML anchors.
    _CONFIGS[_f.stem] = {k: v for k, v in _raw.items() if not k.startswith("_")}

_GRIDS: dict = yaml.safe_load((_HERE / "grids.yaml").read_text())

# ── Backward-compat aliases ────────────────────────────────────────────────────

STAGE2_CONFIGS        = _CONFIGS["heuristic_filtering"]["ablations"]
RO_ALPHA_SWEEP_CONFIGS = _CONFIGS["heuristic_filtering"]["ro_alpha_sweep"]
TOXIC_CONFIGS         = _CONFIGS["toxic_language_detection"]["main"]
STRING_NORM_CONFIGS   = _CONFIGS["string_normalization"]["main"]
ALL_GRIDS             = list(_GRIDS.values())

# ── Path helpers ───────────────────────────────────────────────────────────────

def _tasks_part(exp: dict) -> str:
    tpn = exp.get("tasks_per_node")
    tpj = exp.get("tasks_per_job", 1)
    if tpn:
        return f"{exp['n_tasks']}t_{tpn}tpn_{exp['cpus_per_task']}c"
    if tpj > 1:
        return f"{exp['n_tasks']}t_{tpj}tpj_{exp['cpus_per_task']}c"
    return f"{exp['n_tasks']}t_{exp['cpus_per_task']}c"


def make_id(exp: dict) -> str:
    """Return canonical unique experiment identifier (used in metadata.csv)."""
    return (
        f"{exp['stage']}"
        f"__{exp['dataset']}"
        f"__{exp['config']}"
        f"__{_tasks_part(exp)}"
        f"__r{exp['replicate']}"
    )


def make_exp_path(exp: dict, base_dir) -> Path:
    """Return experiment directory path: base/stage/dataset/config__Nt_Cc__rR."""
    dir_name = f"{exp['config']}__{_tasks_part(exp)}__r{exp['replicate']}"
    return Path(base_dir) / exp["stage"] / exp["dataset"] / dir_name

# ── Probe datasets (preprocess/make_probes.py) ─────────────────────────────────

def probe_name(corpus: str, frac: float) -> str:
    """Dataset name for a probe: size in per mille, e.g. am_stories_probe004 = 0.004."""
    return f"{corpus}_probe{round(frac * 1000):03d}"


def probe_folder(probe_root, stage: str, corpus: str, frac: float) -> str:
    return f"{probe_root}/{stage}/{corpus}/size_{frac:g}/data"


def _infeasible_probes(probe_root) -> set[tuple[str, str, str]] | None:
    """(stage, corpus, size) of probes make_probes.py did not write, or None
    if its summary.csv is not there (e.g. when not on Snellius)."""
    import csv
    summary = Path(probe_root) / "summary.csv"
    if not summary.exists():
        return None
    with summary.open() as f:
        return {(r["stage"], r["corpus"], f"{float(r['size_frac']):g}")
                for r in csv.DictReader(f) if r["feasible"] != "True"}


def _expand_probe_datasets(spec: dict) -> list[tuple[str, str]]:
    """[(dataset_name, input_folder)] for a spec with probe_sizes."""
    root, stage = spec["probe_root"], spec["stage"]
    infeasible = _infeasible_probes(root)
    out = []
    for corpus, frac in product(spec["datasets"], spec["probe_sizes"]):
        if infeasible is not None and (stage, corpus, f"{frac:g}") in infeasible:
            print(f"  [skip] {stage} {corpus} size {frac:g}: probe infeasible (see {root}/summary.csv)")
            continue
        out.append((probe_name(corpus, frac), probe_folder(root, stage, corpus, frac)))
    return out

# ── Grid expansion ─────────────────────────────────────────────────────────────

def expand_grid(spec: dict = None) -> list[dict]:
    """
    Expand a grid spec into a flat list of experiment dicts.
    Defaults to am_stories_docs_scaling if no spec given.
    """
    if spec is None:
        spec = _GRIDS["am_stories_docs_scaling"]

    stage                  = spec["stage"]
    configs_group          = spec["configs_group"]
    config_map             = _CONFIGS[stage][configs_group]

    configs_raw            = spec["configs"]
    configs = list(config_map.keys()) if configs_raw == "all" else configs_raw

    if "probe_sizes" in spec:
        datasets = _expand_probe_datasets(spec)
    else:
        datasets = [(d, DATASETS[d]) for d in spec["datasets"]]
    n_tasks_list           = spec["n_tasks"]
    cpus_list              = spec["cpus_per_task"]
    replicates             = spec["replicates"]
    partition              = spec["partition"]
    time                   = spec["time"]
    limit                  = spec.get("limit", -1)
    tasks_per_node         = spec.get("tasks_per_node")
    tasks_per_job          = spec.get("tasks_per_job", 1)
    parallel_tasks_per_job = spec.get("parallel_tasks_per_job", False)
    hpc_mem_per_cpu_gb     = spec.get("hpc_mem_per_cpu_gb")
    ear_verbose            = spec.get("ear_verbose", 1)

    experiments = []
    for config, (dataset, input_folder), n_tasks, cpus, rep in product(
        configs, datasets, n_tasks_list, cpus_list, range(1, replicates + 1)
    ):
        exp = {
            "stage":         stage,
            "config":        config,
            "filter_flags":  config_map[config],
            "dataset":       dataset,
            "input_folder":  input_folder,
            "n_tasks":       n_tasks,
            "cpus_per_task": cpus,
            "replicate":     rep,
            "partition":     partition,
            "time":          time,
            "limit":         limit,
            "ear_verbose":   ear_verbose,
        }
        if tasks_per_node is not None:
            exp["tasks_per_node"] = tasks_per_node
        if tasks_per_job > 1:
            exp["tasks_per_job"] = tasks_per_job
        if parallel_tasks_per_job:
            exp["parallel_tasks_per_job"] = parallel_tasks_per_job
        if hpc_mem_per_cpu_gb is not None:
            exp["hpc_mem_per_cpu_gb"] = hpc_mem_per_cpu_gb
        experiments.append(exp)

    return experiments


def expand_all_grids() -> list[dict]:
    """Return experiments from every grid in grids.yaml, each tagged with its
    grid name under "grid"."""
    result = []
    for name, spec in _GRIDS.items():
        for exp in expand_grid(spec):
            exp["grid"] = name
            result.append(exp)
    return result


if __name__ == "__main__":
    exps = expand_all_grids()
    print(f"Total experiments: {len(exps)}")
    for e in exps[:3]:
        print(f"  {make_id(e)}")
    print("  ...")
