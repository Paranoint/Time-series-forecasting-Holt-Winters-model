import numpy as np
import pandas as pd
from pathlib import Path


def _sanitize(v):
    return str(v).replace('.', '').replace('-', 'm').replace(' ', '')


def generate_time_series(
    n=100,
    intercept=10.0,      # начальный уровень
    trend=0.1,           # коэффициент тренда
    season_amp=0.0,      # амплитуда сезонности
    season_period=None,  # период сезонности
    ar_phi=None,         # коэффициент AR(1)
    sigma=1.0,           # стандартное отклонение шума
    outlier_frac=0.0,
    seed=None,
):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    
    # база в виде intercept + trend*t
    y = intercept + trend * t
    
    # сезонность
    if season_period and season_amp != 0:
        y = y + season_amp * np.sin(2 * np.pi * t / season_period)
    
    # AR(1)
    if ar_phi is not None:
        y_ar = np.zeros_like(y, dtype=float)
        y_ar[0] = y[0]
        for i in range(1, n):
            eps_i = rng.normal(0, sigma)
            y_ar[i] = intercept + trend * i + ar_phi * (y_ar[i-1] - intercept - trend * i) + eps_i
        y = y_ar
    else:
        # шум
        eps = rng.normal(0, sigma, size=n)
        y = y + eps

    # outliers
    if outlier_frac and outlier_frac > 0.0:
        k = max(1, int(n * outlier_frac))
        idx = rng.choice(n, size=k, replace=False)
        y[idx] += rng.normal(0, 5 * sigma, size=k)

    df = pd.DataFrame({"t": t, "y": y})
    return df

params = {
    "n": 1000,
    "intercept": 10.0,
    "trend": 0.69,
    "season_amp": 20.0,
    "season_period": 12,
    "ar_phi": None,
    "sigma": 1.0,
    "outlier_frac": 0,
    "seed": 42,
}

df = generate_time_series(
    n=int(round(params["n"] * 1.3)),
    intercept=params["intercept"],
    trend=params["trend"],
    season_amp=params["season_amp"],
    season_period=params["season_period"],
    ar_phi=params["ar_phi"],
    sigma=params["sigma"],
    outlier_frac=params["outlier_frac"],
    seed=params["seed"],
)

print(df.head(10))
print(f"\nShape: {df.shape}")
print(f"Mean: {df['y'].mean():.4f}, Std: {df['y'].std():.4f}")

project_root = Path(__file__).resolve().parent.parent
out_dir = project_root / "data"
out_dir.mkdir(parents=True, exist_ok=True)

fname = (
    "season_"
    + "_".join(
        _sanitize(params[k])
        for k in ("n", "outlier_frac")
    )
    + ".csv"
)
out_path = out_dir / fname

df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
