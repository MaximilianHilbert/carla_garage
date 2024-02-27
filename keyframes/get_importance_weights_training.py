import argparse
import numpy as np
import torch
import os
from action_correlation_model import train_ape_model
from team_code.config import GlobalConfig
from team_code.data import CARLA_Data
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from coil_utils.baseline_helpers import merge_config_files


def get_prev_actions(index, img_path_list, prev_action_num, measurements_list):
    previous_actions_list = []
    previous_action_index = index
    previous_action_index_buffer = index
    while len(previous_actions_list) < prev_action_num * 3:
        previous_action_index -= 3
        if (previous_action_index < 0) or (
                img_path_list[previous_action_index].split('/')[0] != img_path_list[index].split('/')[0]):
            previous_actions_list.append(measurements_list[previous_action_index_buffer]['brake'])
            previous_actions_list.append(measurements_list[previous_action_index_buffer]['throttle'])
            previous_actions_list.append(measurements_list[previous_action_index_buffer]['steer'])
        else:
            previous_actions_list.append(measurements_list[previous_action_index]['brake'])
            previous_actions_list.append(measurements_list[previous_action_index]['throttle'])
            previous_actions_list.append(measurements_list[previous_action_index]['steer'])
            previous_action_index_buffer = previous_action_index
    previous_actions_list.reverse()
    previous_action_stack = np.array(previous_actions_list).astype(np.float)
    return previous_action_stack


def get_target_actions(index, img_path_list, target_action_num, measurements_list):
    target_actions_list = []
    target_actions_index = index
    target_actions_index_buff = index
    while len(target_actions_list) < target_action_num * 3:
        if (target_actions_index >= len(img_path_list)) or (img_path_list[target_actions_index].split('/')[0] != img_path_list[index].split('/')[0]):
            target_actions_list.append(measurements_list[target_actions_index_buff]['steer'])
            target_actions_list.append(measurements_list[target_actions_index_buff]['throttle'])
            target_actions_list.append(measurements_list[target_actions_index_buff]['brake'])
        else:
            target_actions_list.append(measurements_list[target_actions_index]['steer'])
            target_actions_list.append(measurements_list[target_actions_index]['throttle'])
            target_actions_list.append(measurements_list[target_actions_index]['brake'])
            target_actions_index_buff = target_actions_index
        target_actions_index += 3
    return np.array(target_actions_list).astype(np.float)


def main(args):
    merged_config_object=merge_config_files(args)
    checkpoint_path=os.path.join(os.environ.get("WORK_DIR"), "_logs","keyframes", f"repetition_{str(args.training_repetition)}","checkpoints")
    for repetition, seed in enumerate(tqdm(args.seeds)):
        checkpoint_name=f"correlation_weights_prev{merged_config_object.number_previous_waypoints}_rep{repetition}_neurons{args.neurons[0]}.npy"
        checkpoint_full_path=os.path.join(checkpoint_path, checkpoint_name)
        train_ape_model(args, seed, repetition,merged_config_object, checkpoint_full_path, if_save=True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neurons', type=int, nargs='+', default=[300], help='the dimensions of the FC layers')
    parser.add_argument('--baseline-folder-name', dest="baseline_folder_name", type=str, default="keyframes", help='name of the folder that gets created for the baseline')
    parser.add_argument('--experiment',type=str, default="keyframes_vanilla_weights", help='name of the experiment/subfoldername that gets created for the baseline')
    parser.add_argument('--use-disk-cache', dest="use_disk_cache", type=int, default=0, help='use caching on /scratch')
    parser.add_argument('--number-of-workers', dest="number_of_workers", type=int, default=12, help='dataloader workers')
    parser.add_argument('--seeds', dest="seeds", type=int, nargs="+", help='seeds for training, also determining the retrain amount')
    parser.add_argument('--batch-size', dest="batch_size", type=int, help='batch')
    parser.add_argument('--training-repetition', dest="training_repetition", type=int, default=0)
    parser.add_argument('--setting', type=str, default="all")
    args = parser.parse_args()
    main(args)

