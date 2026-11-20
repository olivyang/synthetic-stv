from pathlib import Path

import numpy as np
import pandas as pd

from stv import allbutone_analysis
from basic_experiments import mallows_election, _make_seed

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _summarize(elections):
    return (elections.groupby(["m", "k", "n", "phi"])
            .agg(n_roots=("root", "count"),
                 mean_p_wrong=("p_wrong", "mean"),
                 se_p_wrong=("p_wrong",
                             lambda x: x.std(ddof=1) / np.sqrt(len(x))),
                 mean_e_replaced=("e_replaced", "mean"),
                 frac_adv_flip=("adv_flip", "mean"),
                 mean_max_replaced=("max_replaced", "mean"),
                 frac_roots_p_wrong_pos=("p_wrong",
                                         lambda x: (x > 0).mean()),
                 mean_n_types=("n_types", "mean"))
            .reset_index())


def _save(elections, summary, results_dir):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    elections.to_csv(results_dir / "allbutone_elections.csv", index=False)
    summary.to_csv(results_dir / "allbutone_summary.csv", index=False)
    return results_dir


def run_allbutn_experiment(m, k, ns, phis, n_roots=500, seed=0, paired=True,
                           save=True, results_dir=RESULTS_DIR, verbose=True):
    """
    All-but-one experiment on Mallows root elections for one (m, k).

    paired : if True, the rng seed ignores phi, so root r at size n uses the
             same uniforms at every phi (phi comparisons are paired).
    save   : write the two CSVs (allbutone_elections / allbutone_summary).
    """
    ns = [ns] if np.isscalar(ns) else list(ns)
    phis = [phis] if np.isscalar(phis) else list(phis)
    rows = []
    for n in ns:
        for r in range(n_roots):
            for phi in phis:
                s = _make_seed(seed, m, k, 0.0 if paired else phi, n, r)
                profile = mallows_election(n, m, phi, np.random.default_rng(s))
                res = allbutone_analysis(profile, m, k)
                rows.append({"m": m, "k": k, "n": n, "phi": phi, "root": r,
                             "winners": "-".join(map(str, res["winners"])),
                             "n_types": res["n_types"],
                             "p_wrong": res["p_wrong"],
                             "e_replaced": res["e_replaced"],
                             "adv_flip": res["adv_flip"],
                             "max_replaced": res["max_replaced"],
                             "n_flip_types": res["n_flip_types"]})
        if verbose:
            print(f"  m={m} k={k} n={n} done")
    elections = pd.DataFrame(rows)
    summary = _summarize(elections)
    if save:
        d = _save(elections, summary, results_dir)
        if verbose:
            print(f"saved to {d}")
    return elections, summary


def run_allbutn_sweep(ms=range(3, 9), ns=(50, 100),
                      phis=(0.2, 0.4, 0.6, 0.8, 1.0), n_roots=200, seed=0,
                      paired=True, results_dir=RESULTS_DIR):
    """
    Runs run_allbutn_experiment for every m in ms and every k in 2..m-1,
    then saves ONE combined allbutone_elections.csv / allbutone_summary.csv.
    """
    parts = []
    for m in ms:
        for k in range(2, m):
            print(f"m={m}, k={k}")
            el, _ = run_allbutn_experiment(m, k, ns, phis, n_roots=n_roots,
                                           seed=seed, paired=paired,
                                           save=False, verbose=False)
            parts.append(el)
    elections = pd.concat(parts, ignore_index=True)
    summary = _summarize(elections)
    d = _save(elections, summary, results_dir)
    print(f"saved to {d}")
    return elections, summary

if __name__ == "__main__":
    elections, summary = run_allbutn_sweep(n_roots=200)
    print(summary.to_string(index=False))