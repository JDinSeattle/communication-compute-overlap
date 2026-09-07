"""Pointwise, distribution-free interval for the median paired trial speedup."""
import math
import statistics


def paired_comparison(ratios, margin=.05, confidence=.95):
    """Exact binomial/order-statistic interval; a process repeat is the sample unit.

    Assumes independent, representative repeat pairs. No normality assumption,
    no resampling of correlated CUDA iterations, and no simultaneous grid claim.
    With fewer than six pairs a finite 95% interval cannot be formed this way.
    """
    if not 0 < margin < 1 or not 0 < confidence < 1:
        raise ValueError("invalid comparison policy")
    if not ratios or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in ratios):
        raise ValueError("invalid paired speedup")
    ordered = sorted(ratios)
    n = len(ordered)
    interval = None
    for k in range(1, n // 2 + 1):
        coverage = 1 - 2 * sum(math.comb(n, j) for j in range(k)) / 2**n
        if coverage >= confidence:
            interval = {"lower": ordered[k-1], "upper": ordered[n-k],
                        "minimum_coverage": coverage, "order_index": k}
    decision = "insufficient_repeats"
    if interval:
        lo, hi = interval["lower"], interval["upper"]
        decision = ("beneficial" if lo > 1 + margin else "regressed" if hi < 1 - margin else
                    "within_margin" if lo >= 1 - margin and hi <= 1 + margin else "inconclusive")
    return {"median_paired_speedup": statistics.median(ordered), "paired_ratios": list(ratios),
            "repeat_pairs": n, "median_interval": interval, "requested_confidence": confidence,
            "classification_margin": margin, "region": decision,
            "method": "exact_binomial_order_statistic_median",
            "scope": "pointwise independent process-repeat pairs; not simultaneous grid inference"}
