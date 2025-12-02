import json
import pandas as pd
from pathlib import Path
import re


def parse_filename(name):
    """
    Parse filename:
    ts_X_Y_Z_hw_metrics.json
    ts_poly_X_Y_Z_hw_metrics.json
    """

    # remove extension
    name = name.replace(".json", "")

    # poly or not
    if name.startswith("ts_poly_"):
        model = "poly"
        name = name[len("ts_poly_"):]  # drop prefix
    elif name.startswith("ts_"):
        model = "linear"
        name = name[len("ts_"):]  # drop prefix
    else:
        model = "unknown"

    # now should be X_Y_Z_hw_metrics
    # split on "_"
    parts = name.split("_")

    # expected: X, Y, Z, hw, metrics (so take first 3)
    try:
        n = int(parts[0])
        n_features = int(parts[1])
        outlier_frac = float(parts[2])
    except:
        n, n_features, outlier_frac = None, None, None

    return model, n, n_features, outlier_frac


def collect_json_to_csv(input_dir, output_csv):

    rows = []
    input_dir = Path(input_dir)

    for fp in sorted(input_dir.glob("*.json")):
        with open(fp, "r") as f:
            data = json.load(f)

        row = data.copy()

        # add filename
        row["filename"] = fp.name

        # extract model params from filename
        model, n, n_features, outlier_frac = parse_filename(fp.name)

        row["model"] = model
        row["file_n"] = n
        row["file_n_features"] = n_features
        row["file_outlier_frac"] = outlier_frac

        # flatten param_deviation
        dev = row.pop("param_deviation", {})

        for k, v in dev.items():
            row[f"param_dev__{k}"] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print(f"Saved CSV with {len(df)} rows to: {output_csv}")
    
if __name__ == "__main__":
    collect_json_to_csv(
        input_dir="./output/python",      # папка с json
        output_csv="./output/results_py.csv"       # куда сохранять
    )
