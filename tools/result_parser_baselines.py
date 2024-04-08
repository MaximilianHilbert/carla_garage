import os
import pandas as pd
import regex as re
from tqdm import tqdm
weathers = {"train": [1, 3, 6, 8], "test": [10, 14]}


def weather_mapping(value):
    for key, value_lst in weathers.items():
        if value in value_lst:
            return key


def main():
    print("Started scanning...")
    result_files = {}
    for root, dirs, files in os.walk(os.environ.get("RESULT_ROOT")):
        for dir in dirs:
            eval_reps = os.path.join(root, dir)
            if re.findall(".*_.*", dir):
                for rep_root, rep_dirs, rep_files in os.walk(eval_reps):
                    for rep_dir in rep_dirs:
                        file_lst=[]
                        rep_path = os.path.join(eval_reps, rep_dir)
                        if rep_dir == "results":
                            for filename in tqdm(os.listdir(rep_path)):
                                file_lst.append(os.path.join(rep_path, filename))
                            result_files[dir] = file_lst
    df_lst = []
    print("Started merging...")
    for eval_rep, path_lst in tqdm(result_files.items()):
        for path in path_lst:
            eval_results = pd.read_csv(path)
            eval_results["eval_rep"] = eval_rep
            df_lst.append(eval_results)
    df = pd.concat(df_lst, ignore_index=True)
    df["success"] = (df["timeout"] == 0) & (df["collision"] == 0)
    df["success"] = df["success"].astype("int")
    df["weather"] = df["weather"].map(lambda x: weather_mapping(x))
    groups = df.groupby(["baseline", "experiment", "town","traffic", "weather", "eval_rep", "setting"]).agg(
        timeouts_percentage=("timeout", "mean"),
        collisions_percentage=("collision", "mean"),
        success_percentage=("success", "mean"),
    )
    groups = groups.groupby(["baseline", "experiment", "town","traffic", "weather", "eval_rep", "setting"]).agg(
        timeout_mean=("timeouts_percentage", "mean"),
        timeout_std=("timeouts_percentage", "std"),
        collisions_mean=("collisions_percentage", "mean"),
        collisions_std=("collisions_percentage", "std"),
        success_mean=("success_percentage", "mean"),
        success_std=("success_percentage", "std"),
    )
    groups.to_csv(os.path.join(os.environ.get("RESULT_ROOT"), os.path.join(os.environ.get("RESULT_ROOT"),"combined.csv")))


if __name__ == "__main__":

    main()
