from stv import all_rankings, missing_ballot_analysis

import csv
import multiprocessing as mp
from collections import Counter
from functools import lru_cache

import numpy as np

@lru_cache(maxsize=None)
def _inversions(m) -> np.ndarray[np.int64]:
    """Distance of each ranking from the reference 0, 1,..., m-1"""
    R = all_rankings(m)
    inv = np.zeros(len(R), dtype=np.int64)
    for i in range(m):
        for j in range(i + 1, m):
            inv += R[:, i] > R[:, j]
    return inv

def mallows_ballot(m, phi, rng):
    """returns the ranking of one mallows(phi) ballot"""
    ranking = []
    for i in range(m):
        w = phi ** (i - np.arange(i + 1))
        ranking.insert(int(rng.choice(i + 1, p=w / w.sum())), i)
    return tuple(ranking)

def mallows_election(n, m, phi, rng):
    return Counter(mallows_ballot(m, phi, rng) for _ in range(n))

def mallows_dist(m, phi) -> np.ndarray[np.float64]:
    """Gives array of probability weights for each ranking"""
    w = phi ** _inversions(m).astype(float)
    return w / w.sum()

def _bits(mask, m):
    return "-".join(str(c) for c in range(m) if mask >> c & 1)

def _make_seed(seed, k, phi, n, trial):
    ss = np.random.SeedSequence([seed, k, int(round(phi * 1e6)), n, trial])
    return int(ss.generate_state(1)[0])

def _one_trial(args):
    """Simulates one election with one missing ballot from an rng seed"""
    culture, m, k, par, n, seed, trial = args
    rng = np.random.default_rng(seed)
    obs = mallows_election(n - 1, m, par, rng)
    dist = mallows_dist(m, par)
    r = missing_ballot_analysis(obs, m, k, dist)

    # test one scenario by drawing one random ballot from the culture and 
    # running stv
    i = int(rng.choice(len(dist), p=dist))
    sampled = tuple(int(x) for x in all_rankings(m)[i])
    full_mask = int(r["win_masks"][i])
    pred_mask = sum(1 << c for c in r["predicted"])
    top = r["flipping"][0] if r["flipping"] else None
    return {
        "culture": culture, "m": m, "k": k, "param": par, "n": n,
        "trial": trial, "seed": seed,
        "n_unique_observed": len(obs),
        "predicted_winners": _bits(pred_mask, m),
        "sampled_missing_ballot": "-".join(map(str, sampled)),
        "winners_with_sampled_ballot": _bits(full_mask, m),
        "prediction_correct": full_mask == pred_mask,
        "p_wrong_exact": r["p_wrong"],
        "e_replaced": r["e_replaced"],
        "adv_flip_possible": r["adv_flip"],
        "max_replaced": r["max_replaced"],
        "n_flipping_rankings": len(r["flipping"]),
        "top_flipping_ranking": "-".join(map(str, top[0])) if top else "",
        "top_flipping_prob": top[1] if top else "",
    }

def run_experiment(culture="mallows", m=6, ks=(2, 3, 4),
                   params=(0.5, 0.8, 1.0), ns=(51, 201), trials=50,
                   seed=0, n_jobs=1, out_prefix="results"):
    """
    Runs multiple trials for each combination of parameters to be tested
    Returns results as csv!
    """
    # parameter tuples are arbitrary and should be edited to test different values
    elections, summary = [], []
    pool = mp.Pool(None if n_jobs == -1 else n_jobs) if n_jobs != 1 else None
    try:
        for k in ks:
            for par in params:
                for n in ns:
                    jobs = [(culture, m, k, par, n,
                             _make_seed(seed, k, par, n, t), t)
                            for t in range(trials)]
                    rows = (pool.map(_one_trial, jobs) if pool
                            else [_one_trial(j) for j in jobs])
                    elections.extend(rows)
                    cell = {
                        "culture": culture, "m": m, "k": k, "param": par,
                        "n": n, "trials": trials,
                        "mean_p_wrong": np.mean([x["p_wrong_exact"] for x in rows]),
                        "mean_e_replaced": np.mean([x["e_replaced"] for x in rows]),
                        "frac_adv_flip_possible": np.mean([x["adv_flip_possible"] for x in rows]),
                        "frac_sampled_wrong": np.mean([not x["prediction_correct"] for x in rows]),
                        "mean_max_replaced": np.mean([x["max_replaced"] for x in rows]),
                    }
                    summary.append(cell)
                    print(f"{k:>2} {par:>6} {n:>6} {cell['mean_p_wrong']:>9.4f} "
                          f"{cell['mean_e_replaced']:>9.4f} "
                          f"{cell['frac_adv_flip_possible']:>12.3f} "
                          f"{cell['frac_sampled_wrong']:>19.3f}")
    finally:
        if pool:
            pool.close()
    for name, rows in (("elections", elections), ("summary", summary)):
        path = f"{out_prefix}_{name}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path} ({len(rows)} rows)")
    return elections, summary

if __name__ == "__main__":
    run_experiment(n_jobs=-1)