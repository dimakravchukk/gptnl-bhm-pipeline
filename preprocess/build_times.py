"""Per corpus x stage build times from a make_probes.py log.

Each corpus x stage starts at its "N source files under ..." line. The scan
ends at its "scanned ..." line; the rest (packing, re-reading the selected
rows, writing shards and manifests) ends where the next corpus x stage
starts, or at the final "summary:" line.

    python build_times.py probes_build_<jobid>.out [-o build_times.csv]

A corpus x stage still running when the log ends gets an empty total_s.
"""

import argparse
import csv
import re
import sys
from datetime import datetime

LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) \w+ (.*)$")
TAG = re.compile(r"^\[([^/\]]+)/([^\]]+)\] (.*)$")
FIELDS = ["stage", "corpus", "start", "scan_s", "rescans", "build_s", "total_s"]


def parse(lines):
    rows, cur = [], None

    def close(at):
        if cur is not None:
            if at is not None:
                cur["total_s"] = round((at - cur["_t0"]).total_seconds(), 1)
                if cur["_tscan"] is not None:
                    cur["build_s"] = round((at - cur["_tscan"]).total_seconds(), 1)
            rows.append(cur)

    for line in lines:
        m = LINE.match(line.rstrip("\n"))
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")
        msg = m.group(2)
        if msg.startswith("summary:") or re.match(r"^\d+/\d+ probes feasible", msg):
            close(t)
            cur = None
            continue
        tag = TAG.match(msg)
        if not tag:
            continue
        stage, corpus, rest = tag.groups()
        if "source files under" in rest:
            close(t)
            cur = {"stage": stage, "corpus": corpus, "start": t.isoformat(sep=" "),
                   "scan_s": "", "rescans": 0, "build_s": "", "total_s": "",
                   "_t0": t, "_tscan": None}
        elif cur is not None and rest.startswith("scanned "):
            cur["scan_s"] = round((t - cur["_t0"]).total_seconds(), 1)
            cur["_tscan"] = t
        elif cur is not None and "retrying with p=" in rest:
            cur["rescans"] += 1
    close(None)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("-o", "--out", help="CSV path (default: stdout)")
    args = ap.parse_args()

    with open(args.log) as f:
        rows = parse(f)
    out = open(args.out, "w", newline="") if args.out else sys.stdout
    w = csv.DictWriter(out, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    if args.out:
        out.close()
        done = [r for r in rows if r["total_s"] != ""]
        print(f"{len(rows)} corpus x stage rows ({len(done)} complete), "
              f"total {sum(r['total_s'] for r in done):.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
