"""Turn a per-seed result file into the aggregate rows the paper prints.

Means and sample standard deviations over the seeds present, and, for a
robustness file, each model's percentage change against its own clean score.

    python scripts/summarise.py \
        --results per_seed_results/table05_benchmark.json
    python scripts/summarise.py \
        --results robustness/table08_robustness.json --robustness
"""
import argparse
import json
import statistics as st

CONDITIONS = ["clean", "low_light", "overexposure", "shadow", "color_shift",
              "blur", "noise"]
METRICS = [("P", "Precision"), ("R", "Recall"), ("mAP50", "mAP50"),
           ("mAP50_95", "mAP50-95")]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True)
    p.add_argument("--robustness", action="store_true",
                   help="the file holds one record per condition per seed")
    return p.parse_args()


def agg(values):
    if not values:
        return "-"
    if len(values) == 1:
        return "%.3f" % values[0]
    return "%.3f +- %.3f" % (st.mean(values), st.stdev(values))


def accuracy_table(res):
    print("%-18s %-6s %-16s %-16s %-16s %s"
          % ("model", "seeds", "Precision", "Recall", "mAP50", "mAP50-95"))
    for label, runs in res.items():
        cols = [agg([runs[s][f] for s in runs if f in runs[s]]) for f, _ in METRICS]
        print("%-18s %-6d %-16s %-16s %-16s %s" % (label, len(runs), *cols))


def robustness_table(res):
    header = "%-14s" % "condition"
    for label in res:
        header += "%22s" % label
    print(header)
    clean = {}
    for label, runs in res.items():
        v = [runs[s]["clean"]["mAP50"] for s in runs if "clean" in runs[s]]
        clean[label] = st.mean(v) if v else float("nan")
    means = {}
    for cond in CONDITIONS:
        row = "%-14s" % cond
        for label, runs in res.items():
            v = [runs[s][cond]["mAP50"] for s in runs if cond in runs[s]]
            if not v:
                row += "%22s" % "-"
                continue
            m = st.mean(v)
            sd = st.stdev(v) if len(v) > 1 else 0.0
            change = "" if cond == "clean" else " %+.1f" % (100 * (m - clean[label]) / clean[label])
            if cond != "clean":
                means.setdefault(label, []).append(100 * (m - clean[label]) / clean[label])
            row += "%22s" % ("%.3f+-%.3f%s" % (m, sd, change))
        print(row)
    row = "%-14s" % "mean change"
    for label in res:
        row += "%22s" % ("%+.1f" % st.mean(means[label]) if label in means else "-")
    print(row)


def main():
    a = parse_args()
    data = json.load(open(a.results, encoding="utf8"))
    meta = data.get("_meta")
    res = data.get("results", data)
    if meta and meta.get("table"):
        print(meta["table"])
        print()
    if a.robustness:
        robustness_table(res)
    else:
        accuracy_table(res)


if __name__ == "__main__":
    main()
