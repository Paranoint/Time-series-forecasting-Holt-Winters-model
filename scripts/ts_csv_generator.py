import numpy as np
import pandas as pd
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pathlib import Path


def _sanitize(v):
    return str(v).replace('.', '').replace('-', 'm').replace(' ', '')


def generate_time_series(
    n=100,
    n_features=1,
    outlier_frac=0.0,
    seed=None,
    model_type="linear",
    poly_degree=2,
    intercept=10.0,
    trend=0.1,
    season_amp=0.0,
    season_period=None,
    ar_phi=None,
    sigma=1.0,
):

    rng = np.random.default_rng(seed)
    t = np.arange(n)

    # X features
    X = rng.normal(0, 1, size=(n, n_features))
    df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(n_features)])

    # base noise
    eps = rng.normal(0, sigma, size=n)

    # --- MODEL STRUCTURES ---

    # linear w.r.t features
    if model_type == "linear":
        coefs = rng.normal(0, 1, size=n_features)
        y = intercept + X @ coefs + eps
        true_params = {"intercept": intercept, "coefs": coefs}

    # polynomial w.r.t features
    elif model_type == "poly":
        coefs = rng.normal(0, 1, size=n_features * poly_degree)
        X_poly = np.hstack([X**d for d in range(1, poly_degree + 1)])
        y = intercept + X_poly @ coefs + eps
        true_params = {"intercept": intercept, "coefs": coefs, "poly_degree": poly_degree}

    # trend only
    elif model_type == "trend":
        y = intercept + trend * t + eps
        true_params = {"intercept": intercept, "trend": trend}

    # trend + polynomial features
    elif model_type == "trend+poly":
        coefs = rng.normal(0, 1, size=n_features * poly_degree)
        X_poly = np.hstack([X**d for d in range(1, poly_degree + 1)])
        y = intercept + trend * t + X_poly @ coefs + eps
        true_params = {
            "intercept": intercept,
            "trend": trend,
            "coefs": coefs,
            "poly_degree": poly_degree
        }

    else:
        raise ValueError(f"unknown model_type: {model_type}")

    # --- SEASONALITY ---
    if season_period and season_amp != 0:
        y += season_amp * np.sin(2 * np.pi * t / season_period)
        true_params.update({"season_amp": season_amp, "season_period": season_period})

    # --- AR(1) ---
    if ar_phi is not None:
        y_ar = np.zeros_like(y)
        y_ar[0] = y[0]
        for i in range(1, n):
            y_ar[i] = intercept + trend * i + ar_phi * (y_ar[i - 1] - intercept) + eps[i]
        y = y_ar
        true_params.update({"ar_phi": ar_phi})

    # --- OUTLIERS ---
    if outlier_frac > 0:
        k = int(n * outlier_frac)
        pos = rng.choice(n, size=k, replace=False)
        y[pos] += rng.normal(0, 5 * sigma, size=k)
        true_params.update({"outlier_frac": outlier_frac})

    # write result
    df["y"] = y
    df["t"] = t

    # save truth params as constant columns
    for k, v in true_params.items():
        df[f"true_{k}"] = str(v)

    return df


params = {
    "n": 13000,
    "n_features": 200,
    "outlier_frac": 0.0,
    "seed": 1,
    "intercept": 10.0,
    "trend": 0.1,
    "season_amp": 0.0,
    "season_period": None,
    "ar_phi": None,
    "model_type": "poly",
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
    "ts_poly_"
    + "_".join(_sanitize(params[k]) for k in ("n", "n_features", "outlier_frac"))
    + ".csv"
)
out_path = out_dir / fname

df_small.to_csv(out_path, index=False)
print(f"Saved CSV: {out_path}")
