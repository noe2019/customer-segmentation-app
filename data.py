"""Simulated retail transactions (one year)."""
import numpy as np
import pandas as pd


def simulate(n_customers=2500, seed=5, end="2026-10-31"):
    r = np.random.default_rng(seed)
    end = pd.Timestamp(end)
    # latent customer types: champions, loyal, occasional, at-risk, lost
    types = r.choice(5, n_customers, p=[.08, .2, .35, .2, .17])
    freq = np.array([30, 14, 5, 6, 2])[types] * r.uniform(.6, 1.4, n_customers)
    basket = np.array([120, 70, 45, 60, 35])[types] * r.uniform(.6, 1.5, n_customers)
    last_gap = np.array([7, 25, 60, 150, 280])[types] * r.uniform(.3, 1.3, n_customers)
    rows = []
    for cid in range(n_customers):
        k = max(1, r.poisson(freq[cid]))
        last = end - pd.Timedelta(days=int(last_gap[cid]))
        days = np.sort(r.integers(0, max(2, 365 - int(last_gap[cid])), k))
        dates = last - pd.to_timedelta(days, unit="D")
        amounts = r.gamma(4, basket[cid] / 4, k).round(2)
        rows.append(pd.DataFrame({"customer_id": f"CU{cid:05d}", "date": dates, "amount": amounts}))
    return pd.concat(rows, ignore_index=True)
