import os
import time
import traceback
import torch
from tqdm import tqdm
import torch.optim as optim
from pathlib import Path
import cv2
import transfuser_utils as t_u
import numpy as np
from PIL import Image
from copy import deepcopy
from diskcache import Cache
from coil_utils.baseline_helpers import get_copycat_criteria,generate_experiment_name
from team_code.data import CARLA_Data
from team_code.timefuser_model import TimeFuser
import csv
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.distributed import DistributedSampler
from tools.visualize_copycat import align_previous_prediction
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
from coil_utils.baseline_logging import Logger
import pickle
from coil_utils.baseline_helpers import find_free_port
import datetime
from coil_utils.baseline_helpers import (
    set_seed,
    merge_config,
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
        timeout=datetime.timedelta(seconds=259200),
    )
    if rank == 0:
        print("Backend initialized")
    device_id = torch.device(f"cuda:{rank}")
    experiment_name=generate_experiment_name({"training_repetition": args.training_repetition,
                                              "baseline_folder_name": args.baseline_folder_name, "prevnum": args.prevnum,
                                              "speed": args.speed, "backbone": args.backbone, "head": (args.detectboxes or args.bev), "lossweights": args.lossweights})
    merged_config_object = merge_config(args, experiment_name)
    
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
        print(f"Start of Training {merged_config_object.baseline_folder_name}, {merged_config_object.experiment}, {merged_config_object.training_repetition}")
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
    

        if "keyframes" in merged_config_object.experiment:
            if not args.metric:
                filename = os.path.join(
                    os.environ.get("WORK_DIR"),
                        "_logs",
                        "waypoint_weight_generation",
                        "repetition_0",
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
        if "arp" in merged_config_object.experiment:
            policy = TimeFuser("arp-policy", merged_config_object)
            policy.to(device_id)
            policy = DDP(policy, device_ids=[device_id])

            mem_extract = TimeFuser("arp-memory", merged_config_object)
            mem_extract.to(device_id)
            mem_extract = DDP(mem_extract, device_ids=[device_id])
        else:
            model = TimeFuser(merged_config_object.baseline_folder_name, merged_config_object)
            model.to(device_id)
            model = DDP(model, device_ids=[device_id])

        if "arp" in merged_config_object.experiment:
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


        if checkpoint_file is not None:
            accumulated_time = checkpoint["total_time"]
            already_trained_epochs = checkpoint["epoch"]
            if "arp" in merged_config_object.experiment:
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
        if not args.metric:
            for epoch in tqdm(
                range(1 + already_trained_epochs, merged_config_object.epochs_baselines + 1),
                disable=rank != 0,
            ):
                for iteration, data in enumerate(tqdm(data_loader, disable=rank != 0), start=1):
                    # if g_conf.FINISH_ON_VALIDATION_STALE is not None and \
                    #         check_loss_validation_stopped(iteration, g_conf.FINISH_ON_VALIDATION_STALE):
                    #     break
                    capture_time = time.time()
                    all_images, all_speeds, target_point, targets, previous_targets, additional_previous_waypoints, bev_semantic_labels, targets_bb,bb= extract_and_normalize_data(args, device_id, merged_config_object, data)

                    if "arp" in merged_config_object.experiment:
                        mem_extract.zero_grad()
                        #the baseline is defined as getting everything except the last (current) frame into the memory stream, same for speed input
                        pred_dict_memory = mem_extract(x=all_images[:,:-1,...], speed=all_speeds[:,:-1,...] if all_speeds is not None else None, target_point=target_point, prev_wp=additional_previous_waypoints)

                        mem_extract_targets = targets - previous_targets
                        loss_function_params_memory = {
                            **pred_dict_memory,
                            "targets": mem_extract_targets,
                            "bev_targets": bev_semantic_labels,
                            "targets_bb": targets_bb,
                            "device_id": device_id,
                        }

                        mem_extract_loss,_= mem_extract.module.compute_loss(params=loss_function_params_memory, logger=logger, logging_step=(epoch - 1) * len(data_loader) + iteration, rank=rank)
                        mem_extract_loss.backward()
                        mem_extract_optimizer.step()
                        policy.zero_grad()
                        #the baseline is defined as getting the last (current) frame in the policy stream only, same for speed input
                        pred_dict_policy = policy(x=all_images[:,-1:,...], speed=all_speeds[:,-1:,...] if all_speeds is not None else None, target_point=target_point, prev_wp=None, memory_to_fuse=pred_dict_memory["memory"].detach())
                         
                        loss_function_params_policy = {
                            **pred_dict_policy,
                            "targets": targets,
                            "bev_targets": bev_semantic_labels,
                            "targets_bb": targets_bb,
                            "device_id": device_id,
                            
                        }
                        policy_loss,_= policy.module.compute_loss(params=loss_function_params_policy, logger=logger, logging_step=(epoch - 1) * len(data_loader) + iteration, rank=rank)
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
                                os.path.join(basepath,
                                    "checkpoints",
                                    str(epoch) + ".pth",
                                ),
                            )
                        if rank == 0 and iteration%100==0:
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
                            print(f"Epoch: {epoch} // Iteration: {iteration} // Policy_Loss: {policy_loss.data}")
                            print(
                                f"Epoch: {epoch} // Iteration: {iteration} // Mem_Extract_Loss: {mem_extract_loss.data}"
                            )
                        policy_scheduler.step()
                        mem_extract_scheduler.step()

                    else:
                        model.zero_grad()
                        optimizer.zero_grad()
                    
                    if "bcso" in merged_config_object.experiment or "bcoh" in merged_config_object.experiment or "keyframes" in merged_config_object.experiment:
                        pred_dict= model(x=all_images, speed=all_speeds, target_point=target_point, prev_wp=additional_previous_waypoints)
    
                    if "keyframes" in merged_config_object.experiment:
                        reweight_params = {
                            "importance_sampling_softmax_temper": merged_config_object.softmax_temper,
                            "importance_sampling_threshold": action_predict_threshold,
                            "importance_sampling_method": merged_config_object.importance_sample_method,
                            "importance_sampling_threshold_weight": merged_config_object.threshold_weight,
                            "action_predict_loss": data["correlation_weight"].squeeze().to(device_id),
                        }
                    else:
                        reweight_params = {}
                    if "arp" not in merged_config_object.experiment:
                        loss_function_params = {
                            **pred_dict,
                            "bev_targets": bev_semantic_labels if merged_config_object.bev else None,
                            "targets": targets,
                            "targets_bb": targets_bb,
                            **reweight_params,
                            "device_id": device_id,
                            
                        }
                        if "keyframes" in merged_config_object.experiment:
                            loss,_ = model.module.compute_loss(params=loss_function_params,logger=logger, logging_step=(epoch - 1) * len(data_loader) + iteration, rank=rank, keyframes=True)
                        else:
                            loss,_ = model.module.compute_loss(params=loss_function_params,logger=logger, logging_step=(epoch - 1) * len(data_loader) + iteration,rank=rank)
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        
                        if args.debug and iteration%1==0:
                            if merged_config_object.detectboxes:
                                batch_of_bbs_pred=model.module.convert_features_to_bb_metric(pred_dict["pred_bb"])
                            else:
                                batch_of_bbs_pred=None
                            visualize_model(rgb=torch.squeeze(data["rgb"],0),config=merged_config_object, save_path=os.path.join(os.environ.get("WORK_DIR"), "test"), gt_bev_semantic=None,lidar_bev=data["lidar"], target_point=
                                                data["target_point"], pred_wp=pred_dict["wp_predictions"],pred_bb=batch_of_bbs_pred,gt_bbs=bb,step=iteration,pred_bev_semantic=pred_dict["pred_bev_semantic"] if "pred_bev_semantic" in pred_dict.keys() else None, gt_wp=data["ego_waypoints"])
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
                                print(f"Epoch: {epoch} // Iteration: {iteration} // Loss:{loss.data}")
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
            with open(os.path.join(basepath,"config_training.pkl"), "wb") as file:
                pickle.dump(merged_config_object, file)
        dist.destroy_process_group()
        
        if args.metric:
            merged_config_object.number_previous_waypoints=1
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
            if "keyframes" in merged_config_object.experiment:
                filename = os.path.join(
                    os.environ.get("WORK_DIR"),
                        "_logs",
                        "keyframes",
                        "repetition_0",
                        "bcoh_weights_copycat_prev9_rep0_neurons300.npy")
                val_set.set_correlation_weights(path=filename)
                action_predict_threshold = get_action_predict_loss_threshold(
                    val_set.get_correlation_weights(), merged_config_object.threshold_ratio
                )
            data_loader_val = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size,
                num_workers=args.number_of_workers,
                pin_memory=True,
                shuffle=False,  # because of DDP
                drop_last=True,
                sampler=sampler_val,
            )
            
            print("Start of Evaluation")
            cc_lst=[]
            for data,image in zip(tqdm(data_loader_val), data_loader_val.dataset.images):
                image=str(image, encoding="utf-8").replace("\x00", "")
                all_images,current_speed,target_point,targets,previous_targets,additional_previous_waypoints, bev_semantic_labels=extract_and_normalize_data(args=args, device_id="cuda:0", merged_config_object=merged_config_object, data=data)
                
                if "arp" in merged_config_object.experiment:
                    policy.eval()
                    mem_extract.eval()
                    predictions_dict_memory=mem_extract(x=temporal_images, speed=current_speed, target_point=target_point, prev_wp=additional_previous_waypoints,bev_semantic_labels=bev_semantic_labels)
                    predictions_dict_policy = policy(x=current_image, speed=current_speed, target_point=target_point, prev_wp=additional_previous_waypoints, memory_to_fuse=predictions_dict_memory["memory"])
                    loss_function_params = {
                        "branches": predictions,
                        "targets": targets,
                        "speed": current_speed,
                        "config": merged_config_object
                    }
    
                
                    
                if "bcoh" in merged_config_object.experiment or "keyframes" in merged_config_object.experiment:
                    model.eval()
                    temporal_and_current_images = torch.cat([temporal_images, current_image], axis=1)
                    predictions = model(x=temporal_and_current_images, a=current_speed, target_point=target_point, prev_wp=additional_previous_waypoints)
                    loss_function_params = {
                        "branches": predictions,
                        "targets": targets,
                        "speed": current_speed,
                        "config": merged_config_object,
                    }
                if "keyframes" in merged_config_object.experiment:
                    reweight_params = {
                        "importance_sampling_softmax_temper": merged_config_object.softmax_temper,
                        "importance_sampling_threshold": action_predict_threshold,
                        "importance_sampling_method": merged_config_object.importance_sample_method,
                        "importance_sampling_threshold_weight": merged_config_object.threshold_weight,
                        "action_predict_loss": data["correlation_weight"].squeeze().to(device_id),
                    }
                    loss_function_params = {
                    "branches": predictions,
                    "targets": targets,
                    "inputs": current_speed,
                    **reweight_params,
                    "branch_weights": merged_config_object.branch_loss_weight,
                    "variable_weights": merged_config_object.variable_weight,
                    "config": merged_config_object,
                }
                    
                if "bcso" in merged_config_object.experiment:
                    model.eval()
                    predictions = model(x=current_image, a=current_speed, target_point=target_point, prev_wp=additional_previous_waypoints)
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
                cc_lst.append({"image": image, "pred":predictions[0].squeeze().cpu().detach().numpy(),
                                           "gt":targets.squeeze().cpu().detach().numpy(), "loss":loss.cpu().detach().numpy(),
                                           "previous_matrix": data["ego_matrix_previous"].detach().cpu().numpy()[0],
                                           "current_matrix": data["ego_matrix_current"].detach().cpu().numpy()[0]})
            prev_predictions_aligned_lst=[]
            prev_gt_aligned_lst=[]
            for i in range(1,len(cc_lst)):
                prev_predictions_aligned=align_previous_prediction(pred=cc_lst[i-1]["pred"], matrix_previous=cc_lst[i]["previous_matrix"],
                                                                   matrix_current=cc_lst[i]["current_matrix"])
                prev_gt_aligned=align_previous_prediction(pred=cc_lst[i-1]["gt"], matrix_previous=cc_lst[i]["previous_matrix"],
                                                                   matrix_current=cc_lst[i]["current_matrix"])
                prev_predictions_aligned_lst.append(prev_predictions_aligned)
                prev_gt_aligned_lst.append(prev_gt_aligned)
            #data_df = pd.DataFrame.from_dict(wp_dict, orient='index', columns=['image','pred', 'gt', 'loss'])
            criterion_dict=get_copycat_criteria(cc_lst, prev_predictions_aligned_lst,prev_gt_aligned_lst,args.norm)
           
            if not args.custom_validation:
                with open(os.path.join(basepath,f"predictions_all.pkl"), "wb") as file:
                        pickle.dump(cc_lst, file)
            
                with open(os.path.join(basepath,f"aligned_predictions_all.pkl"), "wb") as file:
                        pickle.dump(prev_predictions_aligned_lst, file)
                with open(os.path.join(basepath,f"aligned_gt_all.pkl"), "wb") as file:
                        pickle.dump(prev_gt_aligned_lst, file)
            else:
                with open(os.path.join(basepath,"predictions_cc_routes_only.pkl"), "wb") as file:
                    pickle.dump(wp_dict, file)
            if not args.custom_validation:
                with open(os.path.join(basepath,"predictions_std_all.pkl"), "wb") as file:
                    pickle.dump(criterion_dict, file)
        with open(os.path.join(basepath,"config_cc.pkl"), "wb") as file:
            pickle.dump(merged_config_object, file)
    except RuntimeError as e:
        traceback.print_exc()

    except:
        traceback.print_exc()
@torch.no_grad()
def visualize_model(  # pylint: disable=locally-disabled, unused-argument
    config,
    save_path,
    step,
    rgb,
    lidar_bev,
    target_point,
    pred_wp,
    pred_semantic=None,
    pred_bev_semantic=None,
    pred_depth=None,
    pred_checkpoint=None,
    pred_speed=None,
    pred_bb=None,
    gt_wp=None,
    gt_bbs=None,
    gt_speed=None,
    gt_bev_semantic=None,
    wp_selected=None,
):
    # 0 Car, 1 Pedestrian, 2 Red light, 3 Stop sign
    color_classes = [
        np.array([255, 165, 0]),
        np.array([0, 255, 0]),
        np.array([255, 0, 0]),
        np.array([250, 160, 160]),
    ]

    size_width = int((config.max_y - config.min_y) * config.pixels_per_meter)
    size_height = int((config.max_x - config.min_x) * config.pixels_per_meter)

    scale_factor = 4
    origin = ((size_width * scale_factor) // 2, (size_height * scale_factor) // 2)
    loc_pixels_per_meter = config.pixels_per_meter * scale_factor

    ## add rgb image and lidar
    if config.use_ground_plane:
        images_lidar = np.concatenate(list(lidar_bev.detach().cpu().numpy()[0][:1]), axis=1)
    else:
        images_lidar = np.concatenate(list(lidar_bev.detach().cpu().numpy()[0][:1]), axis=1)

    images_lidar = 255 - (images_lidar * 255).astype(np.uint8)
    images_lidar = np.stack([images_lidar, images_lidar, images_lidar], axis=-1)

    images_lidar = cv2.resize(
        images_lidar,
        dsize=(
            images_lidar.shape[1] * scale_factor,
            images_lidar.shape[0] * scale_factor,
        ),
        interpolation=cv2.INTER_NEAREST,
    )
    # # Render road over image
    # road = self.ss_bev_manager.get_road()
    # # Alpha blending the road over the LiDAR
    # images_lidar = road[:, :, 3:4] * road[:, :, :3] + (1 - road[:, :, 3:4]) * images_lidar

    if pred_bev_semantic is not None:
        bev_semantic_indices = np.argmax(pred_bev_semantic[0].detach().cpu().numpy(), axis=0)
        converter = np.array(config.bev_classes_list)
        converter[1][0:3] = 40
        bev_semantic_image = converter[bev_semantic_indices, ...].astype("uint8")
        alpha = np.ones_like(bev_semantic_indices) * 0.33
        alpha = alpha.astype(float)
        alpha[bev_semantic_indices == 0] = 0.0
        alpha[bev_semantic_indices == 1] = 0.1

        alpha = cv2.resize(
            alpha,
            dsize=(alpha.shape[1] * 4, alpha.shape[0] * 4),
            interpolation=cv2.INTER_NEAREST,
        )
        alpha = np.expand_dims(alpha, 2)
        bev_semantic_image = cv2.resize(
            bev_semantic_image,
            dsize=(
                bev_semantic_image.shape[1] * 4,
                bev_semantic_image.shape[0] * 4,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        images_lidar = bev_semantic_image * alpha + (1 - alpha) * images_lidar

    if gt_bev_semantic is not None:
        bev_semantic_indices = gt_bev_semantic[0].detach().cpu().numpy()
        converter = np.array(config.bev_classes_list)
        converter[1][0:3] = 40
        bev_semantic_image = converter[bev_semantic_indices, ...].astype("uint8")
        alpha = np.ones_like(bev_semantic_indices) * 0.33
        alpha = alpha.astype(float)
        alpha[bev_semantic_indices == 0] = 0.0
        alpha[bev_semantic_indices == 1] = 0.1

        alpha = cv2.resize(
            alpha,
            dsize=(alpha.shape[1] * 4, alpha.shape[0] * 4),
            interpolation=cv2.INTER_NEAREST,
        )
        alpha = np.expand_dims(alpha, 2)
        bev_semantic_image = cv2.resize(
            bev_semantic_image,
            dsize=(
                bev_semantic_image.shape[1] * 4,
                bev_semantic_image.shape[0] * 4,
            ),
            interpolation=cv2.INTER_NEAREST,
        )
        images_lidar = bev_semantic_image * alpha + (1 - alpha) * images_lidar

        images_lidar = np.ascontiguousarray(images_lidar, dtype=np.uint8)

    # Draw wps
    # Red ground truth
    if gt_wp is not None:
        gt_wp_color = (255, 255, 0)
        for wp in gt_wp.detach().cpu().numpy()[0]:
            wp_x = wp[0] * loc_pixels_per_meter + origin[0]
            wp_y = wp[1] * loc_pixels_per_meter + origin[1]
            cv2.circle(
                images_lidar,
                (int(wp_x), int(wp_y)),
                radius=10,
                color=gt_wp_color,
                thickness=-1,
            )

    # Green predicted checkpoint
    if pred_checkpoint is not None:
        for wp in pred_checkpoint.detach().cpu().numpy()[0]:
            wp_x = wp[0] * loc_pixels_per_meter + origin[0]
            wp_y = wp[1] * loc_pixels_per_meter + origin[1]
            cv2.circle(
                images_lidar,
                (int(wp_x), int(wp_y)),
                radius=8,
                lineType=cv2.LINE_AA,
                color=(0, 128, 255),
                thickness=-1,
            )

    # Blue predicted wp
    if pred_wp is not None:
        pred_wps = pred_wp.detach().cpu().numpy()[0]
        num_wp = len(pred_wps)
        for idx, wp in enumerate(pred_wps):
            color_weight = 0.5 + 0.5 * float(idx) / num_wp
            wp_x = wp[0] * loc_pixels_per_meter + origin[0]
            wp_y = wp[1] * loc_pixels_per_meter + origin[1]
            cv2.circle(
                images_lidar,
                (int(wp_x), int(wp_y)),
                radius=8,
                lineType=cv2.LINE_AA,
                color=(0, 0, int(color_weight * 255)),
                thickness=-1,
            )

    # Draw target points
    if config.use_tp:
        x_tp = target_point[0][0] * loc_pixels_per_meter + origin[0]
        y_tp = target_point[0][1] * loc_pixels_per_meter + origin[1]
        cv2.circle(
            images_lidar,
            (int(x_tp), int(y_tp)),
            radius=12,
            lineType=cv2.LINE_AA,
            color=(255, 0, 0),
            thickness=-1,
        )

    # Visualize Ego vehicle
    sample_box = np.array(
        [
            int(images_lidar.shape[0] / 2),
            int(images_lidar.shape[1] / 2),
            config.ego_extent_x * loc_pixels_per_meter,
            config.ego_extent_y * loc_pixels_per_meter,
            np.deg2rad(90.0),
            0.0,
        ]
    )
    images_lidar = t_u.draw_box(images_lidar, sample_box, color=(0, 200, 0), pixel_per_meter=16, thickness=4)

    if pred_bb is not None:
        for box in pred_bb:
            inv_brake = 1.0 - box[6]
            color_box = deepcopy(color_classes[int(box[7])])
            color_box[1] = color_box[1] * inv_brake
            box = t_u.bb_vehicle_to_image_system(box, loc_pixels_per_meter, config.min_x, config.min_y)
            images_lidar = t_u.draw_box(
                images_lidar,
                box,
                color=color_box,
                pixel_per_meter=loc_pixels_per_meter,
            )

    if gt_bbs is not None:
        gt_bbs = gt_bbs.detach().cpu().numpy()[0]
        real_boxes = gt_bbs.sum(axis=-1) != 0.0
        gt_bbs = gt_bbs[real_boxes]
        for box in gt_bbs:
            box[:4] = box[:4] * scale_factor
            images_lidar = t_u.draw_box(
                images_lidar,
                box,
                color=(0, 255, 255),
                pixel_per_meter=loc_pixels_per_meter,
            )

    images_lidar = np.rot90(images_lidar, k=1)

    rgb_image = rgb[0].permute(1, 2, 0).detach().cpu().numpy()

    if wp_selected is not None:
        colors_name = ["blue", "yellow"]
        colors_idx = [(0, 0, 255), (255, 255, 0)]
        images_lidar = np.ascontiguousarray(images_lidar, dtype=np.uint8)
        cv2.putText(
            images_lidar,
            "Selected: ",
            (700, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            images_lidar,
            f"{colors_name[wp_selected]}",
            (850, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            colors_idx[wp_selected],
            2,
            cv2.LINE_AA,
        )

    if pred_speed is not None:
        pred_speed = pred_speed.detach().cpu().numpy()[0]
        images_lidar = np.ascontiguousarray(images_lidar, dtype=np.uint8)
        t_u.draw_probability_boxes(images_lidar, pred_speed, config.target_speeds)

    all_images = np.concatenate((rgb_image, images_lidar), axis=0)
    all_images = Image.fromarray(all_images.astype(np.uint8))

    store_path = str(str(save_path) + (f"/{step:04}.png"))
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)
    all_images.save(store_path)
def extract_and_normalize_data(args, device_id, merged_config_object, data):
    all_images = data["rgb"].to(device_id).to(torch.float32)/255.0
                   
    
    if merged_config_object.speed:
        all_speeds = data["speed"].to(device_id)
    else:
        all_speeds = None
    target_point = data["target_point"].to(device_id)
    wp_targets = data["ego_waypoints"].to(device_id)
    if merged_config_object.number_previous_waypoints>0:
        previous_wp_targets = data["previous_ego_waypoints"].to(device_id)
    else:
        previous_wp_targets=None
    if merged_config_object.prevnum>0:
        additional_previous_wp_targets = torch.flatten(data["additional_waypoints_ego_system"].to(device_id), start_dim=1)
    else:
        additional_previous_wp_targets=None

    bev_semantic_labels=data["bev_semantic"].to(torch.long).to(device_id)
    if merged_config_object.detectboxes:
        bb=data["bounding_boxes"].to(device_id, dtype=torch.float32)
        gt_bb=(
data["center_heatmap"].to(device_id, dtype=torch.float32),
data["wh"].to(device_id, dtype=torch.float32),
data["yaw_class"].to(device_id, dtype=torch.long),
data["yaw_res"].to(device_id, dtype=torch.float32),
data["offset"].to(device_id, dtype=torch.float32),
data["velocity"].to(device_id, dtype=torch.float32),
data["brake_target"].to(device_id, dtype=torch.long),
data["pixel_weight"].to(device_id, dtype=torch.float32),
data["avg_factor"].to(device_id, dtype=torch.float32)
        )
    else:
        gt_bb=None
        bb=None
    return all_images,all_speeds,target_point,wp_targets,previous_wp_targets,additional_previous_wp_targets,bev_semantic_labels, gt_bb,bb

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
        "--backbone",
        type=str,
        choices=["stacking","unrolling"],
        default="unrolling",

    )
    parser.add_argument(
        "--bev",
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
        "--lossweights",
        nargs="+",
        type=float,
        default=[0.33, 0.33, 0.33]

    )
    parser.add_argument("--dataset-repetition", type=int, default=1)

    arguments = parser.parse_args()

    main(arguments)
