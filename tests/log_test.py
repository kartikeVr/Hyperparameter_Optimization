from hyprparam_optimized.logger import log_to_csv

def test_csv_logging(tmp_path):
    results = [
        {
            "params": {"a": 1, "b": 2},
            "mean_score": 0.85,
            "std_score": 0.02
        }
    ]

    log_file = tmp_path / "results.csv"
    log_to_csv(results, log_file)

    assert log_file.exists()
