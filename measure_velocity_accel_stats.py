import os
import time
import traceback
import torch
from tqdm import tqdm
import torch.optim as optim

import csv
import numpy as np
from torchinfo import summary

from copy import deepcopy
from diskcache import Cache
from coil_utils.baseline_helpers import generate_experiment_name, visualize_model,save_checkpoint_and_delete_prior,extract_and_normalize_data
from coil_utils.copycat_helper import get_action_predict_loss_threshold
from team_code.data import CARLA_Data
from team_code.timefuser_model import TimeFuser
from torch.distributed.optim import ZeroRedundancyOptimizer
from pytictoc import TicToc

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from coil_utils.baseline_logging import Logger
import pickle
from coil_utils.baseline_helpers import find_free_port, SequentialSampler
import datetime
from coil_utils.baseline_helpers import (
    set_seed,
    merge_config,
    is_ready_to_save,
    get_latest_saved_checkpoint,
    
)

def main(args):
    timer = TicToc()
    timing=[]
    torch.cuda.empty_cache()
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    print(f"World-size {world_size}, Rank {rank}")
    dist.init_process_group(
        backend="nccl",
        init_method=f"env://127.0.0.1:{find_free_port()}",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=259200),
    )
    if rank == 0:
        print("Backend initialized")
    device_id = torch.device(f"cuda:{rank}")
    
    experiment_name,ablations_dict=generate_experiment_name(args)

    merged_config_object = merge_config(args, experiment_name)
    basepath=os.path.join(os.environ.get("WORK_DIR"),
                    "_logs",
                    merged_config_object.baseline_folder_name,
                    args.experiment_id, 
                    f"repetition_{str(args.training_repetition)}",
                    args.setting)
    # introduce new dataset from the Paper TransFuser++
    dataset = CARLA_Data(
        root=merged_config_object.train_data,
        config=merged_config_object,
        rank=rank,
        baseline=args.baseline_folder_name
    )

    print("Loaded dataset")
    if args.debug:
        sampler=SequentialSampler(dataset)
    else:
        sampler = DistributedSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.number_of_workers,
        pin_memory=True,
        shuffle=False,  # because of DDP
        drop_last=True,
        sampler=sampler,
    )
    vel_vec_lst=[]
    accel_vecs_lst=[]
    for iteration, data in enumerate(tqdm(data_loader, disable=rank != 0), start=1):
        capture_time = time.time()
        all_images, all_speeds, target_point, targets, previous_targets, additional_previous_waypoints, bev_semantic_labels, targets_bb,bb,vel_vecs, accel_vecs= extract_and_normalize_data(args, device_id, merged_config_object, data)
        vel_vecs=vel_vecs[vel_vecs[:,:,-1]!=0][:,:3]
        accel_vecs=accel_vecs[accel_vecs[:,:,-1]!=0][:,:3]
        vel_vec_lst.append(vel_vecs)
        accel_vecs_lst.append(accel_vecs)
    vel_vec_lst=torch.cat(vel_vec_lst)
    accel_vecs_lst=torch.cat(accel_vecs_lst)
    mean_velocity_vector=torch.mean(vel_vec_lst, axis=0)
    std_velocity_vector=torch.std(vel_vec_lst, axis=0)

    mean_acceleration_vector=torch.mean(accel_vecs, axis=0)
    std_acceleration_vector=torch.std(accel_vecs, axis=0)

    print(f"mean_velocity_vector:{mean_velocity_vector}")
    print(f"std_velocity_vector:{std_velocity_vector}")
    
    print(f"mean_acceleration_vector:{mean_acceleration_vector}")
    print(f"std_acceleration_vector:{std_acceleration_vector}")
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", dest="seed", required=True, type=int, default=345345)
    parser.add_argument(
        "--training-repetition",
        type=int,
        default=0,
        required=True,
    )
    parser.add_argument(
        "--baseline-folder-name",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--norm",
        default=2,
        type=int
    )
    parser.add_argument(
        "--number-of-workers",
        default=12,
        type=int,
        required=True,
    )
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--use-disk-cache", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--printing-step", type=int, default=1000)
    parser.add_argument(
        "--adapt-lr-milestones",
        nargs="+",
        type=int,
        default=[30],
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="02_withheld",
        help="coil requires to be trained on Town01 only, so Town01 are train conditions and Town02 is Test Condition",
    )
    parser.add_argument(
        "--custom-validation",
        default=0,
        type=int
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
        "--framehandling",
        type=str,
        choices=["stacking","unrolling"],
        default="unrolling",

    )
    parser.add_argument(
        "--show-model-complexity",
        type=int,
        choices=[0,1],
        default=0

    )
    parser.add_argument(
        "--bev",
        type=int,
        choices=[0,1],
        default=0

    )
    parser.add_argument(
        "--augment",
        type=int,
        choices=[0,1],
        default=0

    )
    parser.add_argument(
        "--ego_velocity_prediction",
        type=int,
        choices=[0,1],
        default=0

    )
    parser.add_argument(
        "--detectboxes",
        type=int,
        choices=[0,1],
        default=0

    )
    parser.add_argument(
        "--predict_vectors",
        type=int,
        choices=[0,1],
        default=0

    )
    parser.add_argument(
        "--velocity_brake_prediction",
        type=int,
        choices=[0,1],
        default=0

    )

    parser.add_argument(
        "--zero-redundancy-optim",
        type=int,
        choices=[0,1],
        default=0

    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=0

    )
    parser.add_argument(
        "--visualize-combined",
        type=int,
        default=0

    )
    parser.add_argument(
        "--visualize-without-rgb",
        type=int,
        default=0

    )
    parser.add_argument(
        "--lossweights",
        nargs="+",
        type=float,
        default=[0.33, 0.33, 0.33]

    )
    parser.add_argument(
        "--init",
        type=int,
        choices=[0,1],
        default=0

    )
    parser.add_argument(
        "--pretrained",
        type=int,
        choices=[0,1],
        default=1

    )
    parser.add_argument(
        "--subsampling",
        type=int,
        choices=[0,1],
        default=0

    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        required=True

    )
    parser.add_argument("--datarep",type=int, default=1)
    parser.add_argument("--backbone",type=str, default="resnet", choices=["videoresnet", "resnet", "swin", "x3d_xs", "x3d_s"])
    arguments = parser.parse_args()
    if not arguments.detectboxes and arguments.velocity_brake_prediction:
        parser.error("When velocity_brake prediction is activated, detectboxes has to be true to")
    if not arguments.bev and arguments.detectboxes:
        parser.error("When detectboxes prediction is activated, bev queries have to be formed first")
    if not arguments.bev and arguments.velocity_brake_prediction:
        parser.error("When velocity_brake_prediction prediction is activated, bev queries have to be formed first")
    if not arguments.detectboxes and arguments.predict_vectors:
        parser.error("When predict_vectors is activated,  detectboxes has to be true to")
    if not arguments.bev and arguments.predict_vectors:
        parser.error("When predict_vectors is activated, bev queries have to be formed first")
    if arguments.velocity_brake_prediction and arguments.predict_vectors:
        parser.error("When predict_vectors is activated, velocity brake prediction is not possible")
    main(arguments)
