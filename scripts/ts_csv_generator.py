import numpy as np
import pandas as pd
from pathlib import Path


def _sanitize(v):
    return str(v).replace('.', '').replace('-', 'm').replace(' ', '')


def generate_time_series(
    n=100,
    intercept=10.0,      # начальный уровень
    trend=0.1,           # коэффициент тренда
    seasonal_type=None,   # "add" | "mul" | None
    season_amp=0.0,      # амплитуда сезонности
    season_period=None,  # период сезонности
    sigma=1.0,           # стандартное отклонение шума
    outlier_frac=0.0,
    seed=None,
):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    
    # база в виде intercept + trend*t
    base = intercept + trend * t
    
    # сезонность
    if seasonal_type and season_period and season_amp != 0:
        phase = np.sin(2 * np.pi * t / season_period)

        if seasonal_type == "add":
            seasonal = season_amp * phase
            y = base + seasonal

        elif seasonal_type == "mul":
            seasonal = 1.0 + season_amp * phase
            y = base * seasonal

        else:
            raise ValueError("seasonal_type must be 'add', 'mul' or None")
    else:
        y = base.copy()
    
    # шум
    eps = rng.normal(0, sigma, size=n)
    if seasonal_type == "mul":
        y = y * (1.0 + eps / np.maximum(1.0, np.abs(y)))
    else:
        y = y + eps

    # outliers
    if outlier_frac and outlier_frac > 0.0:
        k = max(1, int(n * outlier_frac))
        idx = rng.choice(n, size=k, replace=False)
        if seasonal_type == "mul":
            y[idx] *= rng.normal(1.0, 5.0, size=k)
        else:
            y[idx] += rng.normal(0, 5 * sigma, size=k)

    min_y = y.min()
    if min_y <= 0 and seasonal_type == "mul":
        y = y + (1 - min_y)


    df = pd.DataFrame({"t": t, "y": y})
    return df

base_params = {
    "intercept": 10.0,
    "trend": 0.69,
    "seasonal_type": "mul", # "add" | "mul" | None
    "season_amp": 0.3,
    "season_period": 24,
    "sigma": 1.0,
    "seed": 42,
}


# Комбинации для генерации
n_list = [50, 100, 500, 1000, 10000]
outliers_for_big_n = [0.1, 0.2, 0.3]

project_root = Path(__file__).resolve().parent.parent
out_dir = project_root / "data"
out_dir.mkdir(parents=True, exist_ok=True)

def create_and_save(n, outlier_frac):
    """Генерация + сохранение в CSV."""
    params = base_params.copy()
    params["n"] = n
    params["outlier_frac"] = outlier_frac

    df = generate_time_series(
        n=int(round(n * 1.3)),
        intercept=params["intercept"],
        trend=params["trend"],
        seasonal_type=params["seasonal_type"],
        season_amp=params["season_amp"],
        season_period=params["season_period"],
        sigma=params["sigma"],
        outlier_frac=params["outlier_frac"],
        seed=params["seed"],
    )

    fname = (
        "season_mul_"
        + "_".join(
            _sanitize(params[k])
            for k in ("n", "outlier_frac")
        )
        + ".csv"
    )
    out_path = out_dir / fname
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


# 1) ДАННЫЕ БЕЗ ВЫБРОСОВ
for n in n_list:
    create_and_save(n, outlier_frac=0)

# 2) ДАННЫЕ С ВЫБРОСАМИ ТОЛЬКО ДЛЯ n=10000
for o in outliers_for_big_n:
    create_and_save(10000, outlier_frac=o)