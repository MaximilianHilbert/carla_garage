import argparse
import numpy as np
import os
from action_correlation_model import train_ape_model
from tqdm import tqdm
from coil_utils.baseline_helpers import merge_config

def generate_experiment_name():
    return f"waypoint_weight_generation_training"
def main(args):
    experiment_name=generate_experiment_name()
    merged_config_object = merge_config(args,experiment_name)
    checkpoint_path = os.path.join(
        os.environ.get("WORK_DIR"),
        "_logs",
        "waypoint_weight_generation",
        f"repetition_{str(args.training_repetition)}",
        "checkpoints",
    )
    for repetition, seed in enumerate(tqdm(args.seeds)):
        checkpoint_name = f"checkpoint_waypoint_weight_generation{merged_config_object.number_previous_waypoints}_rep{repetition}_neurons{args.neurons[0]}.npy"
        checkpoint_full_path = os.path.join(checkpoint_path, checkpoint_name)
        train_ape_model(
            args,
            seed,
            repetition,
            merged_config_object,
            checkpoint_full_path,
            if_save=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--neurons",
        type=int,
        nargs="+",
        default=[300],
        help="the dimensions of the FC layers",
    )
    parser.add_argument(
        "--baseline-folder-name",
        dest="baseline_folder_name",
        type=str,
        default="keyframes",
        help="name of the folder that gets created for the baseline",
    )

    parser.add_argument(
        "--use-disk-cache",
        dest="use_disk_cache",
        type=int,
        default=0,
        help="use caching on /scratch",
    )
    parser.add_argument(
        "--number-of-workers",
        dest="number_of_workers",
        type=int,
        default=12,
        help="dataloader workers",
    )
    parser.add_argument(
        "--speed",
        type=int,
        choices=[0,1],
        default=0,
    )
    parser.add_argument(
        "--prevnum",
        type=int,
        choices=[0,1],
        default=0,
        help="n-1 is considered"
    )
    parser.add_argument(
        "--bev",
        type=int,
        choices=[0,1],
        default=0,
    )
    parser.add_argument(
        "--detectboxes",
        type=int,
        choices=[0,1],
        default=0,
    )

    parser.add_argument(
        "--datarep",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--seeds",
        dest="seeds",
        type=int,
        nargs="+",
        help="seeds for training, also determining the retrain amount",
    )
    parser.add_argument("--batch-size", dest="batch_size", type=int, help="batch")
    parser.add_argument("--training-repetition", dest="training_repetition", type=int, default=0)
    parser.add_argument("--setting", type=str, default="all")
    args = parser.parse_args()
    main(args)
