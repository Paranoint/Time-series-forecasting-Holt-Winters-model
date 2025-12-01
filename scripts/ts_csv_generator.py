import numpy as np
import pandas as pd
import os
from pathlib import Path

def _sanitize(v):
    return str(v).replace('.', '').replace('-', 'm').replace(' ', '')

def generate_time_series(
    n=100,
    n_features=1,
    outlier_frac=0.0,
    seed=None,
    # yt = a + b * t + (season) + AR + et
    intercept=10.0, # a
    trend=0.1,      # b
    season_amp=0.0,
    season_period=None,
    ar_phi=None,    # зависимость от предыдущего значения
    sigma=1.0,      # для шума
):

    rng = np.random.default_rng(seed)
    t = np.arange(n)
    
    # базовая структура признаков (X_t)
    X = rng.normal(0, 1, size=(n, n_features))
    
    # базовый сигнал
    y = intercept + trend * t
    if season_period and season_amp != 0:
        y += season_amp * np.sin(2 * np.pi * t / season_period)
    
    eps = rng.normal(0, sigma, size=n)
    y = y + eps

    # AR(1)
    if ar_phi is not None:
        y_ar = np.zeros_like(y)
        y_ar[0] = y[0]
        for i in range(1, n):
            y_ar[i] = intercept + trend * i + ar_phi * (y_ar[i - 1] - intercept) + eps[i]
        y = y_ar

    # выбросы
    if outlier_frac > 0:
        k = int(n * outlier_frac)
        pos = rng.choice(n, size=k, replace=False)
        y[pos] += rng.normal(0, 5 * sigma, size=k)

    df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(n_features)])
    df["y"] = y
    df["t"] = t

    # добавляем истину
    cfg_dict = {
        "intercept": intercept,
        "trend": trend,
        "season_amp": season_amp,
        "season_period": season_period,
        "ar_phi": ar_phi,
        "sigma": sigma,
    }
    for k, v in cfg_dict.items():
        df[f"true_{k}"] = v

    return df

params = {
    "n": 1000,
    "n_features": 1,
    "outlier_frac": 0.2,
    "seed": 1,
    "intercept": 10.0,
    "trend": 0.1,
    "season_amp": 0.0,
    "season_period": None,
    "ar_phi": None,
    "sigma": 1.0,
}

df_small = generate_time_series(n=params["n"], 
                                n_features=params["n_features"], 
                                outlier_frac=params["outlier_frac"], 
                                seed=params["seed"])

# Пример вывода
print(df_small.head(), "\n")

# Сохранение в data с параметрами в имени файла
project_root = Path(__file__).resolve().parent.parent
out_dir = project_root / "data"
out_dir.mkdir(parents=True, exist_ok=True)

fname = (
    "ts_"
    + "_".join(_sanitize(params[k]) for k in ("n", "n_features", "outlier_frac"))
    + ".csv"
)
out_path = out_dir / fname

df_small.to_csv(out_path, index=False)
print(f"Saved CSV: {out_path}")
