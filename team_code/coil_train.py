import os
import time
import traceback
import torch
from tqdm import tqdm
import torch.optim as optim
from diskcache import Cache
from coil_network.coil_model import CoILModel
import numpy as np
import pandas as pd
from coil_utils.baseline_helpers import get_copycat_criteria
from team_code.data import CARLA_Data
import csv
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import Sampler
class SequentialSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)
from torch.distributed.optim import ZeroRedundancyOptimizer
from coil_utils.baseline_logging import Logger
import pickle
from coil_utils.baseline_helpers import find_free_port
import datetime
from tools.visualize_copycat import norm
from coil_utils.baseline_helpers import (
    set_seed,
    get_controls_from_data,
    merge_config_files,
    is_ready_to_save,
    get_latest_saved_checkpoint,
    get_action_predict_loss_threshold,
    
)


def main(args):
    torch.cuda.empty_cache()
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    print(f"World-size {world_size}, Rank {rank}")
    dist.init_process_group(
        backend="nccl",
        init_method=f"env://127.0.0.1:{find_free_port()}",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=3600),
    )
    if rank == 0:
        print("Backend initialized")
    device_id = torch.device(f"cuda:{rank}")

    merged_config_object = merge_config_files(args)
    basepath=os.path.join(os.environ.get("WORK_DIR"),
                    "_logs",
                    merged_config_object.baseline_folder_name,
                    merged_config_object.experiment,
                    f"repetition_{str(args.training_repetition)}",
                    args.setting)
    logger = Logger(
        merged_config_object.baseline_folder_name,
        merged_config_object.experiment,
        args.training_repetition,
        args.setting,
    )
    if rank == 0:
        logger.create_tensorboard_logs()
        print(f"Start of Training {args.baseline_folder_name}, {args.experiment}, {args.training_repetition}")
    logger.create_checkpoint_logs()
    try:
        set_seed(args.seed)
        checkpoint_file = get_latest_saved_checkpoint(
            basepath
        )
        if checkpoint_file is not None:
            checkpoint = torch.load(
                os.path.join(basepath,
                    "checkpoints",
                    f"{checkpoint_file}.pth",
                ),
                map_location=lambda storage, loc: storage,
            )
            print(f"loaded checkpoint_{checkpoint_file}")
            epoch = checkpoint["epoch"]
            best_loss = checkpoint["best_loss"]
            best_loss_epoch = checkpoint["best_loss_epoch"]
        else:
            epoch = 0
            best_loss = 10000.0
            best_loss_epoch = 0
        if bool(args.use_disk_cache):
            # NOTE: This is specific to our cluster setup where the data is stored on slow storage.
            # During training, we cache the dataset on the fast storage of the local compute nodes.
            # Adapt to your cluster setup as needed. Important initialize the parallel threads from torch run to the
            # same folder (so they can share the cache).
            tmp_folder = str(os.environ.get("SCRATCH", "/tmp"))
            print("Tmp folder for dataset cache: ", tmp_folder)
            tmp_folder = tmp_folder + "/dataset_cache"
            shared_dict = Cache(directory=tmp_folder, size_limit=int(768 * 1024**3))
        else:
            shared_dict = None
        # introduce new dataset from the Paper TransFuser++
        dataset = CARLA_Data(
            root=merged_config_object.train_data,
            config=merged_config_object,
            shared_dict=shared_dict,
            rank=rank,
            baseline=args.baseline_folder_name
        )
    

        if "keyframes" in args.experiment:
            # load the correlation weights and reshape them, that the last 3 elements that do not fit into the batch size dimension get dropped, because the dataloader of Carla_Dataset does the same, it should fit
            list_of_files_path = os.path.join(
                os.environ.get("WORK_DIR"),
                "_logs",
                merged_config_object.baseline_folder_name,
                f"repetition_{str(args.training_repetition)}",
            )
            for file in os.listdir(list_of_files_path):
                full_filename = os.path.join(list_of_files_path, file)
                if f"rep{str(args.training_repetition)}" in file:
                    dataset.set_correlation_weights(path=full_filename)
            action_predict_threshold = get_action_predict_loss_threshold(
                dataset.get_correlation_weights(), merged_config_object.threshold_ratio
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
        if "arp" in args.experiment:
            policy = CoILModel(merged_config_object.model_type, merged_config_object)
            policy.to(device_id)
            policy = DDP(policy, device_ids=[device_id])
            mem_extract = CoILModel(merged_config_object.mem_extract_model_type, merged_config_object)
            mem_extract.to(device_id)
            mem_extract = DDP(mem_extract, device_ids=[device_id])
        else:
            model = CoILModel(merged_config_object.model_type, merged_config_object)
            model.to(device_id)
            model = DDP(model, device_ids=[device_id])
        if merged_config_object.optimizer_baselines == "Adam":
            if "arp" in args.experiment:
                policy_optimizer = optim.Adam(policy.parameters(), lr=merged_config_object.learning_rate)
                mem_extract_optimizer = optim.Adam(mem_extract.parameters(), lr=merged_config_object.learning_rate)

                mem_extract_scheduler = MultiStepLR(
                    mem_extract_optimizer,
                    milestones=args.adapt_lr_milestones,
                    gamma=0.1,
                )
                policy_scheduler = MultiStepLR(policy_optimizer, milestones=args.adapt_lr_milestones, gamma=0.1)
            else:
                optimizer = optim.Adam(model.parameters(), lr=merged_config_object.learning_rate)
                scheduler = MultiStepLR(optimizer, milestones=args.adapt_lr_milestones, gamma=0.1)
        elif merged_config_object.optimizer == "SGD":
            if "arp" in args.experiment:
                policy_optimizer = optim.SGD(
                    policy.parameters(),
                    lr=merged_config_object.learning_rate,
                    momentum=0.9,
                )
                mem_extract_optimizer = optim.SGD(
                    mem_extract.parameters(),
                    lr=merged_config_object.learning_rate,
                    momentum=0.9,
                )
                mem_extract_scheduler = MultiStepLR(
                    mem_extract_optimizer,
                    milestones=args.adapt_lr_milestones,
                    gamma=0.1,
                )
                policy_scheduler = MultiStepLR(policy_optimizer, milestones=args.adapt_lr_milestones, gamma=0.1)
            else:
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=merged_config_object.learning_rate,
                    momentum=0.9,
                )
                scheduler = MultiStepLR(optimizer, milestones=args.adapt_lr_milestones, gamma=0.1)
        else:
            raise ValueError

        if checkpoint_file is not None:
            accumulated_time = checkpoint["total_time"]
            already_trained_epochs = checkpoint["epoch"]
            if "arp" in args.experiment:
                policy.load_state_dict(checkpoint["policy_state_dict"])
                policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
                mem_extract.load_state_dict(checkpoint["mem_extract_state_dict"])
                mem_extract_optimizer.load_state_dict(checkpoint["mem_extract_optimizer"])
            else:
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                accumulated_time = checkpoint["total_time"]

        else:  # We accumulate iteration time and keep the average speed
            accumulated_time = 0
            already_trained_epochs = 0
        print("Before the loss")
        if "keyframes" in args.experiment:
            from coil_network.keyframes_loss import Loss
        else:
            from coil_network.loss import Loss
        criterion = Loss(merged_config_object.loss_function_baselines)
        for epoch in tqdm(
            range(1 + already_trained_epochs, merged_config_object.epochs_baselines + 1),
            disable=rank != 0,
        ):
            for iteration, data in enumerate(tqdm(data_loader, disable=rank != 0), start=1):
                # if g_conf.FINISH_ON_VALIDATION_STALE is not None and \
                #         check_loss_validation_stopped(iteration, g_conf.FINISH_ON_VALIDATION_STALE):
                #     break
                capture_time = time.time()
                controls = get_controls_from_data(data, args.batch_size, device_id)
                current_image, current_speed, target_point, targets, previous_targets, temporal_images = extract_and_normalize_data(args, device_id, merged_config_object, data)

                if "arp" in args.experiment:
                    mem_extract.zero_grad()
                    memory_branches, memory = mem_extract(temporal_images, target_point)

                    mem_extract_targets = targets - previous_targets
                    loss_function_params_memory = {
                        "branches": memory_branches,
                        "targets": mem_extract_targets,
                        "controls": controls,
                        "inputs": current_speed,
                        "branch_weights": merged_config_object.branch_loss_weight,
                        "variable_weights": merged_config_object.variable_weight,
                    }

                    mem_extract_loss, _ = criterion(loss_function_params_memory)
                    mem_extract_loss.backward()
                    mem_extract_optimizer.step()
                    policy.zero_grad()
                    policy_branches = policy(current_image, current_speed, memory, target_point)
                    loss_function_params_policy = {
                        "branches": policy_branches,
                        "targets": targets,
                        "controls": controls,
                        "inputs": current_speed,
                        "branch_weights": merged_config_object.branch_loss_weight,
                        "variable_weights": merged_config_object.variable_weight,
                    }
                    policy_loss, _ = criterion(loss_function_params_policy)
                    policy_loss.backward()
                    policy_optimizer.step()
                    if is_ready_to_save(epoch, iteration, data_loader, merged_config_object) and rank == 0:
                        state = {
                            "epoch": epoch,
                            "policy_state_dict": policy.state_dict(),
                            "mem_extract_state_dict": mem_extract.state_dict(),
                            "best_loss": best_loss,
                            "total_time": accumulated_time,
                            "policy_optimizer": policy_optimizer.state_dict(),
                            "mem_extract_optimizer": mem_extract_optimizer.state_dict(),
                            "best_loss_epoch": best_loss_epoch,
                        }
                        torch.save(
                            state,
                            os.path.join(
                                os.environ.get("WORK_DIR"),
                                "_logs",
                                merged_config_object.baseline_folder_name,
                                merged_config_object.experiment,
                                f"repetition_{str(args.training_repetition)}",
                                args.setting,
                                "checkpoints",
                                str(epoch) + ".pth",
                            ),
                        )
                    if rank == 0:
                        logger.add_scalar(
                            "Policy_Loss_Iterations",
                            policy_loss.data,
                            (epoch - 1) * len(data_loader) + iteration,
                        )
                        logger.add_scalar("Policy_Loss_Epochs", policy_loss.data, (epoch - 1))
                        logger.add_scalar(
                            "Mem_Extract_Loss_Iterations",
                            mem_extract_loss.data,
                            (epoch - 1) * len(data_loader) + iteration,
                        )
                        logger.add_scalar(
                            "Mem_Extract_Loss_Epochs",
                            mem_extract_loss.data,
                            (epoch - 1),
                        )
                    if policy_loss.data < best_loss:
                        best_loss = policy_loss.data.tolist()
                        best_loss_epoch = epoch
                    accumulated_time += time.time() - capture_time
                    if iteration % args.printing_step == 0 and rank == 0:
                        print(f"Epoch: {epoch} // Iteration: {iteration} // Policy_Loss: {policy_loss.data}\n")
                        print(
                            f"Epoch: {epoch} // Iteration: {iteration} // Mem_Extract_Loss: {mem_extract_loss.data}\n"
                        )
                    policy_scheduler.step()
                    mem_extract_scheduler.step()

                else:
                    model.zero_grad()
                    optimizer.zero_grad()
                # TODO WHY ARE THE PREVIOUS ACTIONS INPUT TO THE BCOH BASELINE??????!!!!#######################################################
                if "bcoh" in args.experiment or "keyframes" in args.experiment:
                    temporal_and_current_images = torch.cat([temporal_images, current_image], axis=1)
                    branches = model(
                        temporal_and_current_images,
                        current_speed,
                        target_point=target_point,
                    )
                if "bcso" in args.experiment:
                    branches = model(x=current_image, a=current_speed, target_point=target_point)
                if "keyframes" in args.experiment:
                    reweight_params = {
                        "importance_sampling_softmax_temper": merged_config_object.softmax_temper,
                        "importance_sampling_threshold": action_predict_threshold,
                        "importance_sampling_method": merged_config_object.importance_sample_method,
                        "importance_sampling_threshold_weight": merged_config_object.threshold_weight,
                        "action_predict_loss": data["correlation_weight"].squeeze().to(device_id),
                    }
                else:
                    reweight_params = {}
                if "arp" not in args.experiment:
                    loss_function_params = {
                        "branches": branches,
                        "targets": targets,
                        **reweight_params,
                        "controls": controls,
                        "inputs": current_speed,
                        "branch_weights": merged_config_object.branch_loss_weight,
                        "variable_weights": merged_config_object.variable_weight,
                        "config": merged_config_object,
                    }
                    if "keyframes" in args.experiment:
                        loss, _ = criterion(loss_function_params)
                    else:
                        loss, _ = criterion(loss_function_params)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    if is_ready_to_save(epoch, iteration, data_loader, merged_config_object) and rank == 0:
                        state = {
                            "epoch": epoch,
                            "state_dict": model.state_dict(),
                            "best_loss": best_loss,
                            "total_time": accumulated_time,
                            "optimizer": optimizer.state_dict(),
                            "best_loss_epoch": best_loss_epoch,
                        }

                        torch.save(
                            state,
                            os.path.join(basepath,
                                "checkpoints",
                                str(epoch) + ".pth",
                            ),
                        )

                    if loss.data < best_loss:
                        best_loss = loss.data.tolist()
                        best_loss_epoch = epoch
                    accumulated_time += time.time() - capture_time
                    if rank == 0:
                        if iteration % args.printing_step == 0:
                            print(f"Epoch: {epoch} // Iteration: {iteration} // Loss:{loss.data}\n")
                        logger.add_scalar(
                            f"{merged_config_object.experiment}_loss",
                            loss.data,
                            (epoch - 1) * len(data_loader) + iteration,
                        )
                        logger.add_scalar(
                            f"{merged_config_object.experiment}_loss_Epochs",
                            loss.data,
                            (epoch - 1),
                        )
            torch.cuda.empty_cache()
        dist.destroy_process_group()
        merged_config_object.number_previous_waypoints=7
        merged_config_object.visualize_copycat=True
        if args.custom_validation:
            with open(os.path.join(os.environ.get("WORK_DIR"),
                            "_logs",
                            "detected_cc_dirs.csv"), "r", newline="") as file:
                reader = csv.reader(file)
                val_lst=[]
                for row in reader:
                    val_lst.append(row)
            val_set = CARLA_Data(root=merged_config_object.val_data, config=merged_config_object, shared_dict=shared_dict, rank=rank,baseline=args.baseline_folder_name, custom_validation_lst=val_lst[0])
        else:
            val_set = CARLA_Data(root=merged_config_object.val_data, config=merged_config_object, shared_dict=shared_dict, rank=rank,baseline=args.baseline_folder_name)
        sampler_val=SequentialSampler(val_set)
        data_loader_val = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            num_workers=args.number_of_workers,
            pin_memory=True,
            shuffle=False,  # because of DDP
            drop_last=True,
            sampler=sampler_val,
        )

        if args.metric:
            print("Start of Evaluation")
            wp_dict={}
            for iteration, (data,image) in enumerate(zip(tqdm(data_loader_val), data_loader_val.dataset.images), start=7):
                image=str(image, encoding="utf-8").replace("\x00", "")
                current_image,current_speed,target_point,targets,previous_targets,temporal_images=extract_and_normalize_data(args=args, device_id="cuda:0", merged_config_object=merged_config_object, data=data)
                
                if "arp" in args.experiment:
                    policy.eval()
                    mem_extract.eval()
                    _,mem=mem_extract(temporal_images,target_point)
                    predictions = policy(current_image, current_speed, mem, target_point)
                    loss_function_params = {
                        "branches": predictions,
                        "targets": targets,
                        "inputs": current_speed,
                        "branch_weights": merged_config_object.branch_loss_weight,
                        "variable_weights": merged_config_object.variable_weight,
                    }
    
                
                    
                if "bcoh" in args.experiment:
                    model.eval()
                    temporal_and_current_images = torch.cat([temporal_images, current_image], axis=1)    
                    predictions = model(x=temporal_and_current_images, a=current_speed, target_point=target_point)
                    loss_function_params = {
                        "branches": predictions,
                        "targets": targets,
                        "inputs": current_speed,
                        "branch_weights": merged_config_object.branch_loss_weight,
                        "variable_weights": merged_config_object.variable_weight,
                        "config": merged_config_object,
                    }
                    
                    
                if "bcso" in args.experiment:
                    model.eval()
                    predictions = model(x=current_image, a=current_speed, target_point=target_point)
                    loss_function_params = {
                        "branches": predictions,
                        "targets": targets,
                        "inputs": current_speed,
                        "branch_weights": merged_config_object.branch_loss_weight,
                        "variable_weights": merged_config_object.variable_weight,
                        "config": merged_config_object,
                    }
                
                    
                loss, _ = criterion(loss_function_params)
                #this is only a viable comparison, if the batch_size is set to 1, because it will be marginalized over the batch dimension before the loss is returned!
                wp_dict.update({iteration:{"image": image,"pred":predictions[0].cpu().detach().numpy(), "gt":targets.cpu().detach().numpy(), "loss":loss.cpu().detach().numpy()}})
            data_df = pd.DataFrame.from_dict(wp_dict, orient='index', columns=['image','pred', 'gt', 'loss'])
            criterion_dict=get_copycat_criteria(data_df, args.norm)
           
            if not args.custom_validation:
                with open(os.path.join(basepath,f"predictions_all.pkl"), "wb") as file:
                        pickle.dump(wp_dict, file)
            else:
                with open(os.path.join(basepath,"predictions_cc_routes_only.pkl"), "wb") as file:
                    pickle.dump(wp_dict, file)
            if not args.custom_validation:
                with open(os.path.join(basepath,"predictions_std_all.pkl"), "wb") as file:
                    pickle.dump(criterion_dict, file)
        with open(os.path.join(basepath,"config.pkl"), "wb") as file:
                pickle.dump(merged_config_object, file)
    except RuntimeError as e:
        traceback.print_exc()

    except:
        traceback.print_exc()

def extract_and_normalize_data(args, device_id, merged_config_object, data):
    temporal_images=[]
    current_speed=[]
    previous_targets=[]
    current_image = torch.reshape(
                    data["rgb"].to(device_id).to(torch.float32) / 255.0,
                    (
                        args.batch_size,
                        -1,
                        merged_config_object.camera_height,
                        merged_config_object.camera_width,
                    ),
                )
    current_speed = data["speed"].to(device_id).reshape(args.batch_size, 1)
    if not merged_config_object.speed_input:
        current_speed = torch.zeros_like(current_speed)
    target_point = data["target_point"].to(device_id)
    targets = data["ego_waypoints"].to(device_id)
    if merged_config_object.number_previous_waypoints>0:
        previous_targets = data["previous_ego_waypoints"].to(device_id)
    if "arp" in args.experiment or "bcoh" in args.experiment or "keyframes" in args.experiment:
        temporal_images = data["temporal_rgb"].to(device_id) / 255.0
    return current_image,current_speed,target_point,targets,previous_targets,temporal_images


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
        "--experiment",
        default=None,
        required=True,
        help="filename of experiment without .yaml",
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
    parser.add_argument("--printing-step", type=int, default=100)
    parser.add_argument(
        "--adapt-lr-milestones",
        nargs="+",
        type=int,
        default=[30],
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="all",
        help="coil requires to be trained on Town01 only, so Town01 are train conditions and Town02 is Test Condition",
    )
    parser.add_argument(
        "--metric",
        type=int,
        default=0,
        help="make eval over eval dataset and save errors to disk",
    )
    parser.add_argument(
        "--custom-validation",
        default=0,
        type=int
    )
    parser.add_argument("--dataset-repetition", type=int, default=1)

    arguments = parser.parse_args()
    main(arguments)
