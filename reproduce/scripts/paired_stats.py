"""Paired statistics between one architecture and a set of references.

Pairs are formed by seed, so each contrast uses only the runs the two arms share.
Reported per metric: the mean paired difference, a t-interval on n-1 degrees of
freedom, the standardized effect size, the two-sided t p-value, and the exact
sign-flip permutation p-value over all sign assignments. This produces Table 6.

    python reproduce/scripts/paired_stats.py \
        --results reproduce/results/per_seed/table05_benchmark.json \
        --model "DaYa-LAB" --against "Base (YOLO11)" "Base + EMA" "DaYa-RGB"

Note that with five paired runs there are 32 sign assignments, so the smallest
two-sided permutation p-value obtainable is 0.063. A design this size cannot
detect a small effect, which is a limit on what a null result here can mean.
"""
import argparse
import itertools
import json
import math
import statistics as st

METRICS = [("mAP50", "mAP50"), ("mAP50_95", "mAP50-95"), ("R", "Recall"),
           ("P", "Precision")]
# two-sided 95 percent critical values of the t distribution, by degrees of freedom
T_CRIT = {1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582,
          6: 2.446912, 7: 2.364624, 8: 2.306004, 9: 2.262157}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True,
                   help="per-seed JSON written by eval_benchmark.py")
    p.add_argument("--model", required=True)
    p.add_argument("--against", nargs="+", required=True)
    return p.parse_args()


def t_pvalue(t, df):
    """Two-sided p-value, by Simpson integration of the density's upper tail."""
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    lo, hi, steps = abs(t), abs(t) + 60.0, 20000
    h = (hi - lo) / steps
    acc = 0.0
    for i in range(steps + 1):
        x = lo + i * h
        w = 1 if i in (0, steps) else (4 if i % 2 else 2)
        acc += w * c * (1 + x * x / df) ** (-(df + 1) / 2)
    return min(1.0, 2 * acc * h / 3)


def paired(diffs):
    n = len(diffs)
    mean, sd = st.mean(diffs), st.stdev(diffs)
    se = sd / math.sqrt(n)
    crit = T_CRIT.get(n - 1, 1.96)
    hits = sum(1 for signs in itertools.product((1, -1), repeat=n)
               if abs(st.mean([s * d for s, d in zip(signs, diffs)])) >= abs(mean) - 1e-12)
    return (mean, mean - crit * se, mean + crit * se,
            mean / sd if sd else float("nan"),
            t_pvalue(mean / se, n - 1) if se else float("nan"),
            hits / 2 ** n)


def main():
    a = parse_args()
    data = json.load(open(a.results, encoding="utf8"))
    res = data.get("results", data)
    for ref in a.against:
        seeds = sorted(set(res[a.model]) & set(res[ref]), key=int)
        print("\n=== %s against %s, %d shared seeds ===" % (a.model, ref, len(seeds)))
        for field, label in METRICS:
            diffs = [res[a.model][s][field] - res[ref][s][field] for s in seeds]
            mean, lo, hi, dz, p_t, p_perm = paired(diffs)
            print("  %-10s %+0.4f [%+0.4f, %+0.4f]  d_z %+0.2f  p_t %.3f  p_perm %.3f"
                  % (label, mean, lo, hi, dz, p_t, p_perm))


if __name__ == "__main__":
    main()
