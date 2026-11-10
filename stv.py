"""
Variables:
    m = number of candidates
    k = number of seats
    n = number of ballots (n-1 = number of sampled ballots)

Metrics:
    p_wrong: P(winner set changes) when drawing from culture
    e_replaced: expected # of winners replaced when drawing from culture
    adv_flip: True if ANY unknown ranking changes the winner set
    max_replaced: max winners the unknown ballot can replace
    flipping: rankings that change the winners, with probabilities

STV rules (I will try to update w/ different rules soon!):
    Droop quota
    Fractional transfers
    Tie breaker = lowest candidate ID
    All complete rankings
"""

import itertools
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def all_rankings(m):
    """Lists all permutations of full rankings with m candidates"""
    # TO DO: be able to generate partial rankings?
    return np.array(list(itertools.permutations(range(m))), dtype=np.int64)


def to_arrays(profile):
    """
    Converts election profile into two arrays:
    B: lists all distinct rankings as lists of candidate IDs
    w: lists counts of each distinct ranking
    """
    B = np.array(list(profile.keys()), dtype=np.int64)
    w = np.array(list(profile.values()), dtype=float)
    return B, w


def stv(B, w, m, k):
    w = w.astype(float).copy()
    quota = int(w.sum() // (k + 1)) + 1  # TO DO: take as argument...
    cont = np.ones(m + 1, dtype=bool)
    rows = np.arange(len(B))
    winners, nwin = 0, 0
    while nwin < k:
        left = cont[:m]
        if left.sum() <= k - nwin:
            for c in np.flatnonzero(left):
                winners |= 1 << int(c)
            break
        first = B[rows, np.argmax(cont[B], axis=1)]
        tally = np.bincount(first, weights=w, minlength=m + 1)[:m]
        top = int(np.argmax(np.where(left, tally, -np.inf)))
        if tally[top] >= quota:
            winners |= 1 << top
            nwin += 1
            w[first == top] *= (tally[top] - quota) / tally[top]
            cont[top] = False
        else:
            cont[int(np.argmin(np.where(left, tally, np.inf)))] = False
    return winners


_POPCOUNT = np.array([bin(i).count("1") for i in range(1 << 8)])


def missing_ballot_analysis(observed, m, k, dist):
    """
    Returns different metrics (explained at top of module) for the missing
    ballot election
    """
    # observed & predicted: election w/ n - 1 ballots & its winners
    B, w = to_arrays(observed)
    predicted = stv(B, w, m, k)
    T = len(B)
    Bx = np.vstack([B, np.zeros((1, m), dtype=np.int64)])
    wx = np.append(w, 1.0)
    R = all_rankings(m)
    replaced = np.zeros(len(R), dtype=np.int64)
    masks = np.zeros(len(R), dtype=np.int64)
    for i, r in enumerate(R):
        Bx[T] = r
        win = stv(Bx, wx, m, k)
        masks[i] = win
        replaced[i] = k - _POPCOUNT[win & predicted]
    flips = replaced > 0
    order = np.argsort(-dist * flips)
    flipping = [(tuple(int(x) for x in R[i]), float(dist[i]))
                for i in order if flips[i]]
    return {"predicted": [c for c in range(m) if predicted >> c & 1],
            "p_wrong": float(dist[flips].sum()),
            "e_replaced": float((dist * replaced).sum()),
            "adv_flip": bool(flips.any()),
            "max_replaced": int(replaced.max()),
            "flipping": flipping,
            "win_masks": masks}