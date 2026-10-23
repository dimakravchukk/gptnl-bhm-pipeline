"""Collect every probe run into one CSV (one row per run) for the anchored BHM.

Runs on Snellius with plain Python (standard library only):

    python preprocess/collect_probe_runs.py \
        --experiments ~/experiments \
        --probes /projects/0/prjs0986/energy_studies/shared/probes \
        -o /projects/0/prjs0986/energy_studies/shared/probes/probe_runs.csv

Definitions follow model/src/data:
  runtime_s (filtering)      sum over tasks of the task log's "Total Runtime"   (load_hf_runs)
  runtime_s (normalisation)  sum over tasks of every pipeline step's total time
                             (reader + formatters + writer, as load_str_norm_jobs)
  total_chars / total_docs   sum over tasks of the reader's doc_len / documents
  energy_kj_measured         sum over EAR app rows of DC_NODE_POWER_W * TIME_SEC,
                             rows below 50 W / 8 tasks dropped                 (load_hf_ear)
Both runtime definitions are written for every run (runtime_log_s, runtime_steps_s).
"""

import argparse
import csv
import glob
import json
import math
import os
import re

HW = "8t_8tpj_128c"
N_TASKS = 8
ALPHA = {"ro_alpha060": 0.60, "ro_alpha065": 0.65, "ro_alpha075": 0.75, "ro_alpha080": 0.80}
ALPHA_DEFAULT = 0.80
FORMS = ["NFC", "NFD", "NFKC"]
MIN_TASK_POWER_W = 50.0 / 8
RUN_RE = re.compile(rf"^(?P<corpus>.+)_probe(?P<pm>\d{{3}})/(?P<config>.+)__{HW}__r(?P<rep>\d+)$")
RT_RE = re.compile(r"Total Runtime:\s*(.+)")

FIELDS = [
    "stage", "corpus", "model_corpus", "probe_frac", "config", "alpha_cfg", "form_idx",
    "n_tasks", "rep", "run_id", "runtime_s", "runtime_log_s", "runtime_steps_s",
    "total_docs", "total_chars", "log_rt_per_char", "energy_kj_measured", "ear_p_mean_w",
    "n_ear_rows", "n_ear_dropped", "target_chars", "probe_chars", "full_corpus_chars",
    "cost_frac", "chars_match_manifest", "status",
]


def parse_runtime(text):
    m = RT_RE.search(text)
    if not m:
        return None
    total = sum(float(v) * {"hour": 3600, "minute": 60, "second": 1}[u]
                for v, u in re.findall(r"([\d.]+)\s*(hour|minute|second)", m.group(1)))
    return total or None


def step_total(block):
    t = block.get("time_stats") or {}
    return float(t.get("total", 0.0) or 0.0)


def read_stats(run):
    docs = chars = 0
    steps_s = 0.0
    files = sorted(glob.glob(f"{run}/output/logs/stats/*.json"))
    for p in files:
        blocks = json.load(open(p))
        for b in blocks:
            steps_s += step_total(b)
            name = b.get("name", "")
            if "READER" in name and "Parquet" in name:
                docs += b["stats"]["documents"]["total"]
                chars += b["stats"]["doc_len"]["total"]
    return len(files), docs, chars, steps_s


def read_log_runtime(run):
    logs = sorted(glob.glob(f"{run}/output/logs/logs/task_*.log"))
    rts = [parse_runtime(open(p, errors="ignore").read()) for p in logs]
    if len(logs) != N_TASKS or any(r is None for r in rts):
        return None
    return sum(rts)


def read_energy(run):
    rows = []
    for f in glob.glob(f"{run}/output/logs/emission/ear_db_*_apps.csv"):
        with open(f, newline="") as fh:
            rows += list(csv.DictReader(fh, delimiter=";"))
    if not rows:
        return None, None, 0, 0
    valid = [r for r in rows if float(r["DC_NODE_POWER_W"]) >= MIN_TASK_POWER_W]
    if not valid:
        return None, None, len(rows), len(rows)
    e_j = sum(float(r["DC_NODE_POWER_W"]) * float(r["TIME_SEC"]) for r in valid)
    p = sum(float(r["DC_NODE_POWER_W"]) for r in valid) / len(valid)
    return e_j / 1000.0, p, len(valid), len(rows) - len(valid)


def manifest(probes, stage, corpus, frac):
    p = os.path.join(probes, stage, corpus, f"size_{frac:g}", "manifest.json")
    if not os.path.exists(p):
        return {}
    m = json.load(open(p))
    return {"target_chars": m["target_chars"], "probe_chars": m["actual_chars"],
            "full_corpus_chars": m["full_corpus_chars"], "cost_frac": m["cost_frac"]}


def collect(experiments, probes):
    out = []
    for stage in ("heuristic_filtering", "string_normalization"):
        base = os.path.join(experiments, stage)
        for run in sorted(glob.glob(f"{base}/*_probe[0-9][0-9][0-9]/*__{HW}__r[0-9]*")):
            m = RUN_RE.match(os.path.relpath(run, base))
            if not m:
                continue
            corpus, config, rep = m["corpus"], m["config"], int(m["rep"])
            frac = int(m["pm"]) / 1000
            n_stats, docs, chars, steps_s = read_stats(run)
            log_s = read_log_runtime(run)
            e_kj, p_w, n_ear, n_drop = read_energy(run)
            hf = stage == "heuristic_filtering"
            rt = log_s if hf else (steps_s if n_stats == N_TASKS else None)
            man = manifest(probes, stage, corpus, frac)
            status = ("ok" if n_stats == N_TASKS and rt and e_kj is not None
                      else "no_energy" if n_stats == N_TASKS and rt else "incomplete")
            out.append({
                "stage": stage,
                "corpus": corpus,
                # names the full-scale training data uses
                "model_corpus": f"{corpus}_str_norm" if hf else corpus,
                "probe_frac": frac,
                "config": config,
                "alpha_cfg": round(ALPHA[config] - ALPHA_DEFAULT, 6) if hf and config in ALPHA else "",
                "form_idx": FORMS.index(config) if not hf and config in FORMS else "",
                "n_tasks": N_TASKS,
                # probe reps are offset so they never share a job key with full-scale runs
                "rep": 100 + int(m["pm"]),
                "run_id": f"probe_{stage}_{corpus}_{config}_p{m['pm']}_r{rep}",
                "runtime_s": rt if rt else "",
                "runtime_log_s": log_s if log_s else "",
                "runtime_steps_s": steps_s if n_stats else "",
                "total_docs": docs,
                "total_chars": chars,
                "log_rt_per_char": math.log(rt) - math.log(chars) if rt and chars else "",
                "energy_kj_measured": e_kj if e_kj is not None else "",
                "ear_p_mean_w": p_w if p_w is not None else "",
                "n_ear_rows": n_ear,
                "n_ear_dropped": n_drop,
                **man,
                "chars_match_manifest": (chars == man["probe_chars"]) if man else "",
                "status": status,
            })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", default=os.path.expanduser("~/experiments"))
    ap.add_argument("--probes", default="/projects/0/prjs0986/energy_studies/shared/probes")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    rows = collect(args.experiments, args.probes)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    by = {}
    for r in rows:
        by.setdefault((r["stage"], r["status"]), 0)
        by[(r["stage"], r["status"])] += 1
    for (stage, status), n in sorted(by.items()):
        print(f"{stage:22s} {status:11s} {n}")
    bad = [r["run_id"] for r in rows if r["chars_match_manifest"] is False]
    print(f"{len(rows)} runs -> {args.out}")
    if bad:
        print(f"WARNING: reader chars differ from the probe manifest for {len(bad)} runs: {bad[:5]}")


if __name__ == "__main__":
    main()
