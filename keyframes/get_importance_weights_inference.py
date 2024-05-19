import argparse
import numpy as np
import torch
import os
from action_correlation_model import train_ape_model
from team_code.config import GlobalConfig
from team_code.data import CARLA_Data
from torch.utils.data import Dataset, DataLoader
from action_correlation_model import ActionModel
import re
from torch.utils.data import SequentialSampler
from tqdm import tqdm
from coil_utils.baseline_helpers import merge_config

def generate_experiment_name():
    return f"keyframes_inference"
def main(args):
    experiment_name=generate_experiment_name()
    merged_config_object = merge_config(args, experiment_name)
    checkpoint_path = os.path.join(
        os.environ.get("WORK_DIR"),
        "_logs",
        "keyframes",
        f"repetition_{str(args.repetition)}",
        "checkpoints",
    )
    checkpoint = os.listdir(checkpoint_path)[0]
    number_previous_actions, repetition, neurons = re.findall(r"\d+", checkpoint)
    action_prediction_model = ActionModel(
        input_dim=(int(number_previous_actions)-1)  * 2,
        output_dim=(merged_config_object.number_future_waypoints-1) * 2,
        neurons=[int(neurons)],
    )
    checkpoint = torch.load(os.path.join(checkpoint_path, checkpoint))
    action_prediction_model.load_state_dict(checkpoint["state_dict"])
    action_prediction_model = action_prediction_model.cuda()
    if args.use_case=="training":
        full_dataset = CARLA_Data(root=merged_config_object.train_data, config=merged_config_object)
    else:
        full_dataset = CARLA_Data(root=merged_config_object.val_data, config=merged_config_object)
    sampler = SequentialSampler(full_dataset)
    data_loader = DataLoader(
        full_dataset,
        sampler=sampler,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.number_of_workers,
    )
    action_prediction_model.eval()
    action_predict_losses = []
    for data in tqdm(data_loader):
        previous_wp = data["previous_ego_waypoints"].cuda().reshape(args.batch_size, -1)
        current_wp = data["ego_waypoints"].cuda().reshape(args.batch_size, -1)

        predict_curr_action = action_prediction_model(previous_wp)
        test_loss = ((predict_curr_action - current_wp).pow(2)).mean().cpu().item()  # TODO whatch out with sum not mean!
        action_predict_losses.append(test_loss)
    os.makedirs(os.path.join(os.environ.get("WORK_DIR"), "_logs", "keyframes"), exist_ok=True)
    np.save(
        os.path.join(
            os.environ.get("WORK_DIR"),
            "_logs",
            "keyframes",
            f"repetition_{str(args.repetition)}",
            f"bcoh_weights_{args.use_case}_prev{number_previous_actions}_rep{repetition}_neurons{neurons}.npy",
        ),
        action_predict_losses,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-folder-name",
        dest="baseline_folder_name",
        type=str,
        default="keyframes",
        help="name of the folder that gets created for the baseline",
    )
    parser.add_argument(
        "--use-case",
        type=str,
        default="training",
        choices=["training", "copycat"],
    )
    parser.add_argument(
        "--training-repetition",
        dest="repetition",
        type=int,
        default=0,
        help="repetition",
    )
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=1, help="batch")
    parser.add_argument(
        "--number-of-workers",
        dest="number_of_workers",
        type=int,
        default=1,
        help="workers to load with",
    )
    parser.add_argument("--setting", type=str, default="all")
    parser.add_argument("--experiment", type=str)
    args = parser.parse_args()
    main(args)
