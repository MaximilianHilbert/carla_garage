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
                    experiment_name, 
                    f"repetition_{str(args.training_repetition)}",
                    args.setting)
    logger = Logger(
        merged_config_object.baseline_folder_name,
        experiment_name,
        args.training_repetition,
        args.setting,
    )
    learning_rate=merged_config_object.learning_rates[merged_config_object.baseline_folder_name][merged_config_object.backbone]
    if rank == 0:
        logger.create_tensorboard_logs()
        print(f"Start of Training {merged_config_object.baseline_folder_name}, {experiment_name}, {merged_config_object.training_repetition}")
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
            if args.debug:
                tmp_folder = str("/home/maximilian/Master/tmp")
            else:
                tmp_folder = str(os.environ.get("SCRATCH", "/tmp"))
            print("Tmp folder for dataset cache: ", tmp_folder)
            tmp_folder = tmp_folder + "/dataset_cache"
            shared_dict = Cache(directory=tmp_folder, size_limit=int(1800 * 1024**3))
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
    

        if "keyframes" in merged_config_object.baseline_folder_name:
            filename = os.path.join(
                os.environ.get("WORK_DIR"),
                    "keyframes",
                    "importance_weights",
                    "bcoh_weights_training_prev9_rep0_neurons300.npy")
            # load the correlation weights and reshape them, that the last 3 elements that do not fit into the batch size dimension get dropped, because the dataloader of Carla_Dataset does the same, it should fit
            dataset.set_correlation_weights(path=filename)
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
        if "arp" in merged_config_object.baseline_folder_name:
            policy = TimeFuser("arp-policy", merged_config_object, rank=rank)
            policy.to(device_id)
            policy = DDP(policy, device_ids=[device_id])

            mem_extract = TimeFuser("arp-memory", merged_config_object, rank=rank)
            mem_extract.to(device_id)
            mem_extract = DDP(mem_extract, device_ids=[device_id])
        else:
            
            model = TimeFuser(merged_config_object.baseline_folder_name, merged_config_object, rank=rank)
            model.to(device_id)
            model = DDP(model, device_ids=[device_id],find_unused_parameters=True if args.freeze else False)
        if "arp" in merged_config_object.baseline_folder_name:
            if bool(args.zero_redundancy_optim):
                summary(policy, (merged_config_object.rgb_input_channels, 1, merged_config_object.height_rgb,  merged_config_object.width_rgb))
                summary(mem_extract, (merged_config_object.rgb_input_channels, merged_config_object.considered_images_incl_current-1, merged_config_object.height_rgb,  merged_config_object.width_rgb))
                policy_optimizer = ZeroRedundancyOptimizer(params=policy.parameters(),optimizer_class=optim.Adam, parameters_as_bucket_view=True,lr=learning_rate, amsgrad=True)
                mem_extract_optimizer = ZeroRedundancyOptimizer(params=mem_extract.parameters(),optimizer_class=optim.Adam,parameters_as_bucket_view=True, lr=learning_rate, amsgrad=True)
            else:
                policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
                mem_extract_optimizer = optim.Adam(mem_extract.parameters(), lr=learning_rate)
           
            mem_extract_scheduler = MultiStepLR(
                mem_extract_optimizer,
                milestones=args.adapt_lr_milestones,
                gamma=0.1,
        
            )
            policy_scheduler = MultiStepLR(policy_optimizer, milestones=args.adapt_lr_milestones, gamma=0.1)
        if "bcoh" in merged_config_object.baseline_folder_name or "keyframes" in merged_config_object.baseline_folder_name:
            if bool(args.zero_redundancy_optim):
                optimizer = ZeroRedundancyOptimizer(params=model.parameters(),optimizer_class=optim.Adam,parameters_as_bucket_view=True,lr=learning_rate, amsgrad=True)
            else:
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = MultiStepLR(optimizer, milestones=args.adapt_lr_milestones, gamma=0.1)
        if "bcso" in merged_config_object.baseline_folder_name:
            if bool(args.zero_redundancy_optim):
                optimizer = ZeroRedundancyOptimizer(params=model.parameters(),optimizer_class=optim.Adam, parameters_as_bucket_view=True, lr=learning_rate, amsgrad=True)
            else:
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = MultiStepLR(optimizer, milestones=args.adapt_lr_milestones, gamma=0.1)
            
        if checkpoint_file is not None:
            accumulated_time = checkpoint["total_time"]
            already_trained_epochs = checkpoint["epoch"]
            if "arp" in merged_config_object.baseline_folder_name:
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
        for epoch in tqdm(
            range(1 + already_trained_epochs, merged_config_object.epochs_baselines + 1),
            disable=rank != 0,
        ):
            if rank==0:
                if "arp" in merged_config_object.baseline_folder_name:
                    combined_losses_policy=[]
                    combined_losses_memory=[]
                else:
                    combined_losses=[]
                detailed_losses=[]
                head_losses_lst=[]
            for iteration, data in enumerate(tqdm(data_loader, disable=rank != 0), start=1):
                capture_time = time.time()
                all_images, all_speeds, target_point, targets, previous_targets, additional_previous_waypoints, bev_semantic_labels, targets_bb,bb= extract_and_normalize_data(args, device_id, merged_config_object, data)
                if "arp" in merged_config_object.baseline_folder_name:
                    mem_extract.zero_grad()
                    #the baseline is defined as getting everything except the last (current) frame into the memory stream, same for speed input
                    if iteration>100 and iteration<150 and epoch==1:
                        timer.tic()
                    pred_dict_memory = mem_extract(x=all_images[:,:-1,...], speed=all_speeds[:,:-1,...] if all_speeds is not None else None, target_point=target_point, prev_wp=additional_previous_waypoints)
                    if iteration>100 and iteration<150 and epoch==1:
                        timer_memory=timer.tocvalue(restart=True)
                    mem_extract_targets = targets - previous_targets
                    loss_function_params_memory = {
                        **pred_dict_memory,
                        "targets": mem_extract_targets,
                        "bev_targets": bev_semantic_labels,
                        "targets_bb": targets_bb,
                        "device_id": device_id,
                        "epoch": epoch,
                        "ego_velocity": all_speeds[:,:-1] if all_speeds is not None else None#only previous velocities will be learned by the memory stream
                    }

                    mem_extract_loss,_, head_losses= mem_extract.module.compute_loss(params=loss_function_params_memory)
                    mem_extract_loss.backward()
                    mem_extract_optimizer.step()
                    policy.zero_grad()
                    #the baseline is defined as getting the last (current) frame in the policy stream only, same for speed input
                    if iteration>100 and iteration<150:
                        timer.tic()
                    pred_dict_policy = policy(x=all_images[:,-1:,...], speed=all_speeds[:,-1:,...] if all_speeds is not None else None, target_point=target_point, prev_wp=None, memory_to_fuse=pred_dict_memory["memory"].detach())
                    if iteration>100 and iteration<150 and epoch==1:
                        timer_policy=timer.tocvalue()
                        timing.append(timer_policy+timer_memory)

                    loss_function_params_policy = {
                        **pred_dict_policy,
                        "targets": targets,
                        "bev_targets": bev_semantic_labels,
                        "targets_bb": targets_bb,
                        "device_id": device_id,
                        "epoch": epoch,
                        "ego_velocity": all_speeds[:,-1] if all_speeds is not None else None#policy stream only learns current velocity
                        
                    }
                    policy_loss,plotable_losses,head_losses= policy.module.compute_loss(params=loss_function_params_policy)
                    policy_loss.backward()
                    policy_optimizer.step()
                    if is_ready_to_save(epoch, iteration, data_loader, merged_config_object) and rank == 0:
                        if bool(args.zero_redundancy_optim):
                            # To save the whole optimizer we need to gather it on GPU 0.
                            policy_optimizer.consolidate_state_dict(0)
                            mem_extract_optimizer.consolidate_state_dict(0)
                        state = {
                            "epoch": epoch,
                            "policy_state_dict": policy.state_dict(),
                            "mem_extract_state_dict": mem_extract.state_dict(),
                            "best_loss": best_loss,
                            "total_time": accumulated_time,
                            "policy_optimizer": policy_optimizer.state_dict(),
                            "mem_extract_optimizer": mem_extract_optimizer.state_dict(),
                            "best_loss_epoch": best_loss_epoch,
                            "timing": np.array(timing).mean()
                        }
                        save_checkpoint_and_delete_prior(state, merged_config_object, args, epoch)
                    if rank == 0:
                        combined_losses_policy.append(policy_loss.cpu().item())
                        combined_losses_memory.append(mem_extract_loss.cpu().item())
                        detailed_losses.append(plotable_losses)
                        head_losses_lst.append(head_losses)
                    
                    accumulated_time += time.time() - capture_time
                   
                    policy_scheduler.step()
                    mem_extract_scheduler.step()

                else:
                    if epoch==1 and iteration==1 and args.show_model_complexity==1:
                        dummy_input=torch.ones_like(all_images)
                        summary(model=model, input_data=[dummy_input,torch.ones((merged_config_object.batch_size,1)),
                                                         torch.ones_like(target_point) ,
                                                         torch.ones((merged_config_object.batch_size,8,2)),
                                                           torch.ones((merged_config_object.batch_size,256,))])
                    if args.freeze:
                        #if we freeze the backbone, we first turn off the bev head and queries, train for the given amount of epochs and later turn off everything, afterwards turning on only bev head and 
                        detector_components=[model.module.head.parameters(),
                                            model.module.change_channel_bev_to_bb_and_upscale.parameters(),
                                           model.module.bev_semantic_decoder.parameters(),
                
                                           ]
                        wp_components=[model.module.wp_gru.parameters(), model.module.transformer_decoder.parameters()]
                        if epoch==1:
                            model.module.bev_query.requires_grad=False
                            for module in detector_components:
                                for param in module:
                                    param.requires_grad=False
                        if epoch==merged_config_object.epochs_baselines-merged_config_object.epochs_after_freeze+1:
                            #first we set every param to be frozen, afterwards we reactivate the detector head, that is easier than picking our individual components of the model
                            #turn of everything
                            for param in model.module.parameters():
                                param.requires_grad=False
                            #turn on wp_predictions
                            model.module.wp_query.requires_grad=True
                            model.module.bev_query.requires_grad=True
                            for module in wp_components:
                                for param in module:
                                    param.requires_grad=True
                            #turn on detector head
                            for module in detector_components:
                                for param in module:
                                    param.requires_grad=True
                    model.zero_grad()
                    optimizer.zero_grad()
                
                if "bcso" in merged_config_object.baseline_folder_name or "bcoh" in merged_config_object.baseline_folder_name or "keyframes" in merged_config_object.baseline_folder_name:
                    if iteration>100 and iteration<150 and epoch==1:
                        timer.tic()
                    pred_dict= model(x=all_images, speed=all_speeds, target_point=target_point, prev_wp=additional_previous_waypoints)
                    
                    if iteration>100 and iteration<150 and epoch==1:
                        forward_time=timer.tocvalue(restart=True)
                        timing.append(forward_time)
                if "keyframes" in merged_config_object.baseline_folder_name:
                    reweight_params = {
                        "importance_sampling_softmax_temper": merged_config_object.softmax_temper,
                        "importance_sampling_threshold": action_predict_threshold,
                        "importance_sampling_method": merged_config_object.importance_sample_method,
                        "importance_sampling_threshold_weight": merged_config_object.threshold_weight,
                        "action_predict_loss": data["correlation_weight"].squeeze().to(device_id),
                    }
                else:
                    reweight_params = {}
                if "arp" not in merged_config_object.baseline_folder_name:
                    loss_function_params = {
                        **pred_dict,
                        "bev_targets": bev_semantic_labels if merged_config_object.bev else None,
                        "targets": targets,
                        "targets_bb": targets_bb,
                        **reweight_params,
                        "device_id": device_id,
                        "epoch": epoch,
                        "ego_velocity": all_speeds[:,-1] if all_speeds is not None else None#only current velocity is relevant
                        
                    }
                    if "keyframes" in merged_config_object.baseline_folder_name:
                        loss,plotable_losses,head_losses = model.module.compute_loss(params=loss_function_params, keyframes=True)
                    else:
                        loss,plotable_losses,head_losses = model.module.compute_loss(params=loss_function_params)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    if rank == 0:
                        combined_losses.append(loss.cpu().item())
                        detailed_losses.append(plotable_losses)
                        head_losses_lst.append(head_losses)
                if args.debug and epoch%2==0:
                    if merged_config_object.detectboxes:
                        if "arp" in merged_config_object.baseline_folder_name:
                            batch_of_bbs_pred=policy.module.convert_features_to_bb_metric(pred_dict_policy["pred_bb"])
                        else:
                            batch_of_bbs_pred=model.module.convert_features_to_bb_metric(pred_dict["pred_bb"])
                    else:
                        batch_of_bbs_pred=None
                    if "arp" in merged_config_object.baseline_folder_name:
                        visualize_model(training=True,args=args,rgb=torch.cat([image for image in all_images.squeeze(0)],axis=1).permute(1, 2, 0).detach().cpu().numpy(),config=merged_config_object,
                                save_path_root=os.path.join(os.environ.get("WORK_DIR"), "test"),
                                gt_bev_semantic=None,lidar_bev=torch.squeeze(data["lidar"],0).detach().cpu().numpy(),
                                target_point=torch.squeeze(target_point,0).detach().cpu().numpy(),
                                pred_wp=torch.squeeze(pred_dict_policy["wp_predictions"],0).detach().cpu().numpy(),
                                step=iteration,
                                gt_wp=targets.squeeze(0),
                                pred_bb=batch_of_bbs_pred,
                                gt_bbs=bb.detach().cpu().numpy() if bb is not None else None,
                                pred_bev_semantic=pred_dict_policy["pred_bev_semantic"].squeeze(0).detach().cpu().numpy() if "pred_bev_semantic" in pred_dict_policy.keys() else None,
                                )
                    else:
                        visualize_model(training=True,args=args,rgb=torch.cat([image for image in all_images.squeeze(0)],axis=1).permute(1, 2, 0).detach().cpu().numpy(),config=merged_config_object,
                                        save_path_root=os.path.join(os.environ.get("WORK_DIR"), "visualisation", "training"),
                                        gt_bev_semantic=None,lidar_bev=torch.squeeze(data["lidar"],0).detach().cpu().numpy(),
                                        target_point=torch.squeeze(target_point,0).detach().cpu().numpy(),
                                        pred_wp=torch.squeeze(pred_dict["wp_predictions"],0).detach().cpu().numpy(),
                                        step=iteration,
                                        gt_wp=targets.squeeze(0),
                                        pred_bb=batch_of_bbs_pred,
                                        gt_bbs=bb.detach().cpu().numpy() if bb is not None else None,
                                        pred_bev_semantic=pred_dict["pred_bev_semantic"].squeeze(0).detach().cpu().numpy() if "pred_bev_semantic" in pred_dict.keys() else None,
                                        )
                if merged_config_object.baseline_folder_name!="arp":
                    if is_ready_to_save(epoch, iteration, data_loader, merged_config_object):
                        if bool(args.zero_redundancy_optim):
                            # To save the whole optimizer we need to gather it on GPU 0.
                            optimizer.consolidate_state_dict(0)
                        if rank==0:
                            state = {
                                "epoch": epoch,
                                "state_dict": model.state_dict(),
                                "best_loss": best_loss,
                                "total_time": accumulated_time,
                                "optimizer": optimizer.state_dict(),
                                "best_loss_epoch": best_loss_epoch,
                                "timing": np.array(timing).mean() if timing else 0
                            }

                            save_checkpoint_and_delete_prior(state, merged_config_object, args, epoch)

                    accumulated_time += time.time() - capture_time
            if rank==0:
                if "arp" in merged_config_object.baseline_folder_name:
                    logger.add_scalar(
                                    "policy_loss_epochs",
                                    np.array(combined_losses_policy).mean(),
                                    (epoch - 1),
                                )
                    logger.add_scalar(
                                    "memory_loss_epochs",
                                    np.array(combined_losses_memory).mean(),
                                    (epoch - 1),
                                )
                else:
                    logger.add_scalar(
                                    "loss_epochs",
                                    np.array(combined_losses).mean(),
                                    (epoch - 1),
                                )


            
                sums=dict.fromkeys(detailed_losses[0].keys(), 0)
                counts=dict.fromkeys(detailed_losses[0].keys(), 0)
                for dic in detailed_losses:
                    for key, value in dic.items():
                        sums[key]+=value
                        counts[key] += 1
                for key in sums.keys():
                    sums[key]=sums[key]/counts[key]
                for key, value in sums.items():
                    logger.add_scalar(
                                        f"{key}_loss",
                                        value.item(),
                                        (epoch - 1),
                                    )
                    logger.flush()
                if merged_config_object.detectboxes:
                    sums=dict.fromkeys(head_losses_lst[0].keys(), 0)
                    counts=dict.fromkeys(head_losses_lst[0].keys(), 0)
                    for dic in head_losses_lst:
                        for key, value in dic.items():
                            sums[key]+=value
                            counts[key] += 1
                    for key in sums.keys():
                        sums[key]=sums[key]/counts[key]

                    for key, value in sums.items():
                        logger.add_scalar(
                                            f"{key}_loss",
                                            value.item(),
                                            (epoch - 1),
                                        )
                        logger.flush()
            
                torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        with open(os.path.join(basepath,"config_training.pkl"), "wb") as file:
            pickle.dump(merged_config_object, file)
        with open(os.path.join(basepath,"training_time.csv"), mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["total_training_time_in_h", "forward_time_in_s"])
            writer.writeheader()
            writer.writerow({"total_training_time_in_h": accumulated_time/3600, "forward_time_in_s": np.array(timing).mean() if timing else 0})
        dist.destroy_process_group()
        
    except RuntimeError as e:
        traceback.print_exc()

    except:
        traceback.print_exc()

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
        "--ego-velocity-prediction",
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
        "--velocity-brake-prediction",
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
        "--pretrained",
        type=int,
        required=1

    )
    parser.add_argument(
        "--subsampling",
        type=int,
        choices=[0,1],
        default=0

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
    main(arguments)
