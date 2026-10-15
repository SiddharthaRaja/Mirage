"""Statistics, with the choices stated.

* **Bootstrap intervals** for accuracies, resampling questions. Spider dev is
  ~1,000 questions; an accuracy on it has a 95% interval of roughly ±3 points,
  which is larger than many published "improvements". Every accuracy in the
  report carries its interval.
* **Paired bootstrap** for comparing two systems on the same questions:
  resample questions, recompute the difference. Pairing removes the variance
  from some questions being harder than others.
* **Cluster bootstrap** for rollout-level rates from training (false-positive
  share, match rate). Eight rollouts of the same question are not eight
  independent observations; resampling *questions* and keeping their rollouts
  together gives honest intervals. Resampling rollouts would make them far too
  narrow.
* **pass@k** uses the unbiased estimator from the Codex paper (Chen et al.,
  2021): with n samples of which c are correct, 1 - C(n-c, k) / C(n, k).

The rule carried over from Vantage: if the interval for a difference includes
zero, the report does not call it a difference.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np


def mean_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return (float("nan"),) * 3
    if a.size == 1:
        return float(a[0]), float(a[0]), float(a[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boots = a[idx].mean(axis=1)
    return float(a.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_diff_ci(
    a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> dict:
    """Difference b - a over the same items."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError("paired comparison needs the same items on both sides")
    d = y - x
    m, lo, hi = mean_ci(d, n_boot=n_boot, seed=seed, alpha=alpha)
    return {
        "diff": m,
        "ci95": (lo, hi),
        "significant": bool(lo > 0 or hi < 0),
        "n": int(d.size),
        "wins": int((d > 0).sum()),
        "losses": int((d < 0).sum()),
    }


def cluster_rate_ci(
    numer: Sequence[float], denom: Sequence[float], clusters: Sequence[str], n_boot: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """A ratio sum(numer)/sum(denom) with a bootstrap over clusters.

    Used for e.g. false-positive share = (#false positives) / (#single-db matches),
    clustered by question.
    """
    by: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for n, d, c in zip(numer, denom, clusters, strict=True):
        by[c][0] += n
        by[c][1] += d
    arr = np.asarray(list(by.values()), dtype=float)
    if arr.size == 0 or arr[:, 1].sum() == 0:
        return float("nan"), float("nan"), float("nan")
    point = arr[:, 0].sum() / arr[:, 1].sum()
    rng = np.random.default_rng(seed)
    k = arr.shape[0]
    idx = rng.integers(0, k, size=(n_boot, k))
    num = arr[idx, 0].sum(axis=1)
    den = arr[idx, 1].sum(axis=1)
    ok = den > 0
    ratios = num[ok] / den[ok]
    if ratios.size == 0:
        return float(point), float("nan"), float("nan")
    return float(point), float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts. Reported
    alongside the paired bootstrap as a sanity check."""
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2**n
    return float(min(1.0, 2 * p))
