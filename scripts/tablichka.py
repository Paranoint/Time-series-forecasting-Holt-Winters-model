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
    if name.startswith("linear_"):
        model = "linear"
        name = name[len("linear_"):]  # drop prefix
    elif name.startswith("season_"):
        model = "season"
        name = name[len("season_"):]  # drop prefix
    elif name.startswith("AR_"):
        model = "AR"
        name = name[len("AR_"):]  # drop prefix
    else:
        model = "unknown"

    # now should be X_Y_Z_hw_metrics
    # split on "_"
    parts = name.split("_")

    # expected: X, Y, Z, hw, metrics (so take first 3)
    try:
        n = int(parts[0])
        outlier_frac = float(parts[1])
    except:
        n, outlier_frac = None, None

    return model, n, outlier_frac


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
        model, n, outlier_frac = parse_filename(fp.name)

        row["model"] = model
        row["file_n"] = n
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
        input_dir="./output/r",      # папка с json
        output_csv="./output/results_r.csv"       # куда сохранять
    )
