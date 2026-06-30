# GPT-NL Data Curation Pipeline — Runtime & Energy Prediction

---

## Components

**`experiment_manager.py`** — CLI for generating and submitting SLURM jobs on Snellius. Defines a grid of experiments (corpus × config × n_tasks × replicate), creates run directories, submits to SLURM, and tracks status in `metadata.csv`.

```bash
python experiment_manager.py grid      # list experiments
python experiment_manager.py generate  # create run dirs
python experiment_manager.py submit    # submit to SLURM
python experiment_manager.py status    # check progress
```

> ⚠️ **Requires the GPT-NL data curation environment.** The experiment manager depends on a patched version of the `datatrove` pipeline used in the GPT-NL project. It must be run in the same venv in which it was originally tested on Snellius — the standard `requirements.txt` in this repo is **not** sufficient to run experiments, only to use the model.

**`model/`** — Bayesian hierarchical models for runtime and energy prediction. See `model/README.md` for full usage, anchoring workflow, and how to reproduce thesis results. Only `requirements.txt` is needed.

---

## Structure

```
model/          — BHM package, results, tests, usage notebook
manage/         — SLURM config templates and experiment grid definitions
experiment_manager.py
requirements.txt   — model only (numpy, jax, numpyro, arviz, ...)
```