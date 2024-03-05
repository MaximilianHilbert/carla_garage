import os
import pandas as pd

weathers = {"train": [1, 3, 6, 8], "test": [10, 14]}


def weather_mapping(value):
    for key, value_lst in weathers.items():
        if value in value_lst:
            return key


def main(args):
    result_files = {}
    for root, dirs, files in os.walk(args.eval_root):
        for dir in dirs:
            eval_reps = os.path.join(root, dir)
            for rep_root, rep_dirs, rep_files in os.walk(eval_reps):
                for rep_dir in rep_dirs:
                    rep_path = os.path.join(eval_reps, rep_dir)
                    if rep_dir == "results":
                        for filename in os.listdir(rep_path):
                            result_files[dir] = os.path.join(rep_path, filename)
    df_lst = []
    for eval_rep, path in result_files.items():
        eval_results = pd.read_csv(path)
        eval_results["eval_rep"] = eval_rep
        df_lst.append(eval_results)
    df = pd.concat(df_lst, ignore_index=True)
    df["success"] = (df["timeout"] == 0) & (df["collision"] == 0)
    df["success"] = df["success"].astype("int")
    df["weather"] = df["weather"].map(lambda x: weather_mapping(x))
    groups = df.groupby(["town", "baseline", "experiment", "traffic", "weather", "eval_rep", "setting"]).agg(
        timeouts_percentage=("timeout", "mean"),
        collisions_percentage=("collision", "mean"),
        success_percentage=("success", "mean"),
    )
    groups = groups.groupby(["town", "baseline", "experiment", "traffic", "weather", "setting"]).agg(
        timeout_mean=("timeouts_percentage", "mean"),
        timeout_std=("timeouts_percentage", "std"),
        collisions_mean=("collisions_percentage", "mean"),
        collisions_std=("collisions_percentage", "std"),
        success_mean=("success_percentage", "mean"),
        success_std=("success_percentage", "std"),
    )
    groups.to_csv(os.path.join(args.eval_root, "combined.csv"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root")

    args = parser.parse_args()
    main(args)
