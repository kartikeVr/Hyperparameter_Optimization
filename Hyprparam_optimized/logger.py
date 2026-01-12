import json
import pandas as pd

def log_to_csv(results, path):
    df = pd.DataFrame([
        {**r["params"], "mean_score": r["mean_score"], "std_score": r["std_score"]}
        for r in results
    ])
    df.to_csv(path, index=False)


def log_to_json(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
