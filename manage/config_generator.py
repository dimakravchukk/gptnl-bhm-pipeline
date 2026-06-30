"""
config_generator.py — Generate config.yaml and run.sh for one experiment dict.

Public API:
  generate_experiment(exp, experiments_dir) → exp_path | None
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .grid import make_id, make_exp_path

HOME         = "/gpfs/home5/dkravchuk"
PIPELINE_DIR = f"{HOME}/projects/data-curation-pipeline"


def _filter_yaml_lines(filter_flags: dict) -> list[str]:
    """
    Convert filter_flags dict to YAML lines (4-space indent, under stage block).

    Format: {filter_name: {param: value, ...}, ...}
    List values (e.g. top_n_grams) are rendered as JSON flow sequences.
    """
    lines = []
    for filter_name, params in filter_flags.items():
        lines.append(f"    {filter_name}:")
        for param, value in params.items():
            if isinstance(value, list):
                lines.append(f"      {param}: {json.dumps(value)}")
            else:
                lines.append(f'      {param}: "{value}"')
    return lines


def make_config_yaml(exp: dict, exp_path: Path) -> str:
    output_folder = exp_path / "output"
    filter_lines  = _filter_yaml_lines(exp.get("filter_flags", {}))

    header = [
        "processing_type: hpc",
        "input_from_previous_output: false",
        "output_folder_template: unused",
        'logs_folder: "{output_folder}/logs"',
        'slurm_logs_folder: "{output_folder}/slurm_logs"',
        'hpc_ear: "true"',
        "",
        "stages:",
        f"  - stage: {exp['stage']}",
        f"    input_folder: {exp['input_folder']}",
        f"    output_folder: {output_folder}",
        f'    hpc_n_tasks: "{exp["n_tasks"]}"',
        f'    hpc_cpus_per_task: "{exp["cpus_per_task"]}"',
        f'    hpc_time: "{exp["time"]}"',
        f'    hpc_partition: "{exp["partition"]}"',
        *(
            [f'    hpc_tasks_per_node: "{exp["tasks_per_node"]}"']
            if exp.get("tasks_per_node")
            else []
        ),
        *(
            [f'    hpc_tasks_per_job: "{exp["tasks_per_job"]}"']
            if exp.get("tasks_per_job", 1) > 1
            else []
        ),
        *(
            [f'    hpc_parallel_tasks_per_job: "true"']
            if exp.get("parallel_tasks_per_job")
            else []
        ),
        *(
            [f'    hpc_mem_per_cpu_gb: "{exp["hpc_mem_per_cpu_gb"]}"']
            if exp.get("hpc_mem_per_cpu_gb")
            else []
        ),
        f'    env_commands: "export HF_HOME={HOME}/.hf_cache"',
        f"    ParquetReader:",
        f'      limit: "{exp.get("limit", -1)}"',
    ]

    return "\n".join(header + filter_lines) + "\n"


def make_run_sh(exp: dict, exp_path: Path) -> str:
    exp_id      = make_id(exp)
    config_path = exp_path / "config.yaml"
    ear_verbose = exp.get("ear_verbose", 1)

    sbatch_wrapper = ""
    if ear_verbose != 1:
        sbatch_wrapper = f"""
# Wrap sbatch to patch --ear-verbose before the job is submitted
REAL_SBATCH=$(command -v sbatch)
_SBATCH_WRAP=$(mktemp -d)
trap 'rm -rf "$_SBATCH_WRAP"' EXIT
cat > "$_SBATCH_WRAP/sbatch" <<'WRAP'
#!/bin/bash
for f in "$@"; do
    [ -f "$f" ] && sed -i 's/^#SBATCH --ear-verbose=.*/#SBATCH --ear-verbose={ear_verbose}/' "$f"
done
exec $REAL_SBATCH "$@"
WRAP
chmod +x "$_SBATCH_WRAP/sbatch"
export PATH="$_SBATCH_WRAP:$PATH"
"""

    return f"""#!/bin/bash
# {exp_id}

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

export PYTHONPATH="{PIPELINE_DIR}/.venv/lib64/python3.11/site-packages:{PIPELINE_DIR}/.venv/lib/python3.11/site-packages:{PIPELINE_DIR}"
export PATH="{HOME}/.local/bin:$PATH"

cd {PIPELINE_DIR}
ln -sf $(which python3.11) .venv/bin/python3.11
{sbatch_wrapper}
poetry run pipeline -s {config_path}
"""


def generate_experiment(exp: dict, experiments_dir: Path) -> Path | None:
    """
    Create experiment directory with config.yaml and run.sh.
    Returns exp_path if created, None if it already exists.
    """
    exp_path = make_exp_path(exp, experiments_dir)

    if exp_path.exists():
        return None

    exp_path.mkdir(parents=True, exist_ok=True)
    # EAR writes ear_db_<node>_*.csv here during job prolog — must exist before sbatch
    (exp_path / "output" / "logs" / "emission").mkdir(parents=True, exist_ok=True)

    (exp_path / "config.yaml").write_text(make_config_yaml(exp, exp_path))

    run_sh = exp_path / "run.sh"
    run_sh.write_text(make_run_sh(exp, exp_path))
    os.chmod(run_sh, 0o755)

    return exp_path


if __name__ == "__main__":
    from grid import expand_grid
    EXPERIMENTS_DIR = Path(f"{HOME}/experiments")

    exps = expand_grid()
    exp  = exps[1]  # pick one non-default to test filter rendering
    path = make_exp_path(exp, EXPERIMENTS_DIR)
    print(f"=== config.yaml preview for: {make_id(exp)} ===")
    print(make_config_yaml(exp, path))
