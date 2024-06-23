
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from coil_utils.baseline_helpers import visualize_model,set_not_included_ablation_args
import os
import pickle
from tools.video_generation import generate_video_stacked
from torch.nn.parallel import DistributedDataParallel as DDP
from coil_utils.copycat_helper import evaluate_baselines_and_save_predictions, preprocess, determine_copycat
from team_code.data import CARLA_Data
from team_code.timefuser_model import TimeFuser
from coil_utils.baseline_helpers import get_latest_saved_checkpoint, SequentialSampler


def load_image_sequence(config,predictions_lst,current_iteration):
    root=os.path.dirname(predictions_lst[current_iteration]["image"])
    index_in_dataset=int(os.path.basename(predictions_lst[current_iteration]["image"]).replace(".jpg",""))
    return np.concatenate([Image.open(os.path.join(root, "0"*(4-len(str(index_in_dataset-i)))+f"{index_in_dataset-i}.jpg"))  for i in reversed(range(config.img_seq_len))],axis=0), root

@torch.no_grad()
def main(args):
    path_of_baselines=os.path.join(os.path.join(os.environ.get("WORK_DIR"),
                "_logs",
    ))
    model=evaluate_baselines_and_save_predictions(args, path_of_baselines)
    params=preprocess(args)
    paths=[]
    results=pd.DataFrame(columns=["baseline", "experiment","metric", "length", "positions"])
    
    for root, dirs, files in os.walk(path_of_baselines):
        for file in files:
            if file=="config_cc.pkl":
                print(f"current model: {root}")
                checkpoint_file = get_latest_saved_checkpoint(
                root
            )
                checkpoint = torch.load(
                    os.path.join(root,
                        "checkpoints",
                        f"{checkpoint_file}.pth",
                    ),
                    map_location=lambda storage, loc: storage,
                )
                print(f"loaded checkpoint_{checkpoint_file}")
                with open(os.path.join(root, "config_cc.pkl"), 'rb') as f:
                    config = pickle.load(f)
                config.number_previous_waypoints=1
                config.visualize_copycat=True
                set_not_included_ablation_args(config)
                if getattr(config, "freeze")==1:
                    continue
                if getattr(config, "bev")==0 or getattr(config, "detectboxes")==0:
                    continue
                if getattr(config, "training_repetition")!=0:
                    continue
                if "arp" in config.baseline_folder_name:
                    policy = TimeFuser("arp-policy", config)
                    policy.to("cuda:0")
                    policy = DDP(policy, device_ids=["cuda:0"])

                    mem_extract = TimeFuser("arp-memory", config)
                    mem_extract.to("cuda:0")
                    mem_extract = DDP(mem_extract, device_ids=["cuda:0"])
                    policy.load_state_dict(checkpoint["policy_state_dict"])
                    mem_extract.load_state_dict(checkpoint["mem_extract_state_dict"])
                   
                else:
                    model = TimeFuser(config.baseline_folder_name, config)
                    model.to("cuda:0")
                    model = DDP(model, device_ids=["cuda:0"])
                    model.load_state_dict(checkpoint["state_dict"])
                with open(os.path.join(root,"predictions_all.pkl"), 'rb') as f:
                    predictions_lst = pickle.load(f)
                with open(os.path.join(root,"aligned_gt_all.pkl"), 'rb') as f:
                    previous_gt_lst = pickle.load(f)
                with open(os.path.join(root,"aligned_predictions_all.pkl"), 'rb') as f:
                    previous_predictions_lst = pickle.load(f)
                
                val_set = CARLA_Data(root=config.val_data, config=config, rank=0,baseline=config.baseline_folder_name)
                sampler_val=SequentialSampler(val_set)
                val_set.set_correlation_weights(path=os.path.join(
                    os.environ.get("WORK_DIR"),
                    "keyframes",
                    "importance_weights",
                    "bcoh_weights_copycat_prev9_rep0_neurons300.npy",
                ))
                data_loader_val = torch.utils.data.DataLoader(
                    val_set,
                    batch_size=1,
                    num_workers=args.number_of_workers,
                    pin_memory=True,
                    shuffle=False,  # because of DDP
                    drop_last=True,
                    sampler=sampler_val,
                )
                keyframes_cc_positions=[]
                our_cc_positions=[]
                velocity_losses=[]
                accel_losses=[]
                
                assert len(params["keyframes_correlations"])==len(data_loader_val.dataset) , "wrong correlation weights selected!"
                for data_loader_position, (data, image_path, keyframe_correlation) in enumerate(zip(tqdm(data_loader_val),data_loader_val.dataset.images, params["keyframes_correlations"])):
                    
                    route_name=os.path.basename(os.path.dirname(os.path.dirname(str(image_path, encoding="utf-8"))))
                    cc_save_path=os.path.join(os.environ.get("WORK_DIR"),"visualisation", "open_loop", config.baseline_folder_name,config.experiment,route_name)
                    data_image=str(image_path, encoding="utf-8").replace("\x00", "")
                    current_index=data_loader_position
                    #previous_index=data_loader_position-1
                    pred_image=predictions_lst[current_index]["image"]
                    if data_image!=pred_image:
                        assert("not aligned")
                    # previous_prediction_aligned=align_previous_prediction(data_df.iloc[previous_index]["pred"][0],
                    #                                                       data["ego_matrix_previous"].detach().cpu().numpy()[0],
                    #                                                       data["ego_matrix_current"].detach().cpu().numpy()[0])
                    detection_ours, detection_keyframes,copycat_information=determine_copycat(args,predictions_lst,data["ego_waypoints"].squeeze().numpy(),previous_predictions_lst[current_index],previous_gt_lst,
                                                                            keyframe_correlation,current_index,params)
                    if detection_keyframes:
                        keyframes_cc_positions.append(data_loader_position)
                    if detection_ours:
                        our_cc_positions.append(data_loader_position)
                    if predictions_lst[current_index]["head_loss"] is not None:
                        velocity_losses.append(predictions_lst[current_index]["head_loss"]["loss_velocity"])
                        accel_losses.append(predictions_lst[current_index]["head_loss"]["loss_brake"])
                    if args.save_whole_scene:
                        if config.img_seq_len<7:
                            empties=np.concatenate([np.zeros_like(Image.open(predictions_lst[0]["image"]))]*(7-config.img_seq_len))
                            image_sequence,root=load_image_sequence(config,predictions_lst, data_loader_position)
                            image_sequence=np.concatenate([empties, image_sequence], axis=0)
                        else:
                            image_sequence,root=load_image_sequence(config,predictions_lst, data_loader_position)
                        if args.visualize_combined or args.visualize_without_rgb:
                                if config.detectboxes:
                                    if "arp" in config.baseline_folder_name:
                                        batch_of_bbs_pred=policy.module.convert_features_to_bb_metric(predictions_lst[current_index]["pred_bb"]["pred_bb"])
                                    else:
                                        batch_of_bbs_pred=model.module.convert_features_to_bb_metric(predictions_lst[current_index]["pred_bb"]["pred_bb"])
                                else:
                                    batch_of_bbs_pred=None
                        visualize_model(args=args,config=config, save_path_root=cc_save_path, rgb=image_sequence, lidar_bev=torch.squeeze(data["lidar"], dim=0),
                                        pred_wp_prev=previous_predictions_lst[current_index],
                                        gt_bev_semantic=torch.squeeze(data["bev_semantic"],dim=0) if not config.bev else None, step=current_index,
                                        target_point=torch.squeeze(data["target_point"],dim=0), pred_wp=predictions_lst[current_index]["pred"]["wp_predictions"],
                                        gt_wp=torch.squeeze(data["ego_waypoints"],dim=0),parameters=copycat_information,
                                        detect_our=False, detect_kf=False,frame=current_index,
                                        pred_bb=batch_of_bbs_pred,
                                        pred_bev_semantic=predictions_lst[current_index]["pred"]["pred_bev_semantic"] if config.bev else None,
                                        gt_bbs=data["bounding_boxes"] if config.detectboxes else None,
                                        prev_gt=previous_gt_lst[current_index],loss=predictions_lst[current_index]["loss"], condition=args.second_cc_condition,
                                        ego_speed=data["speed"].numpy()[0], correlation_weight=params["keyframes_correlations"][current_index],
                                        loss_brake=predictions_lst[current_index]["head_loss"]["loss_brake"] if predictions_lst[current_index]["head_loss"] is not None else None,
                                        loss_velocity=predictions_lst[current_index]["head_loss"]["loss_velocity"] if predictions_lst[current_index]["head_loss"] is not None else None)
                    
                    if detection_ours or detection_keyframes:
                        for i in range(-args.num_surrounding_frames,args.num_surrounding_frames+1):
                            detection_ours, detection_keyframes=False, False
                            #previous_index=data_loader_position+i-1
                            current_index=data_loader_position+i
                            # if current_index in already_saved_indices:
                            #     continue
                            data=data_loader_val.dataset.__getitem__(data_loader_position+i)
                            if config.img_seq_len<7:
                                empties=np.concatenate([np.zeros_like(Image.open(predictions_lst[0]["image"]))]*(7-config.img_seq_len))
                                image_sequence,root=load_image_sequence(config,predictions_lst, data_loader_position+i)
                                image_sequence=np.concatenate([empties, image_sequence], axis=0)
                            else:
                                image_sequence,root=load_image_sequence(config,predictions_lst, data_loader_position+i)
                        
                            # previous_prediction_aligned=align_previous_prediction(data_df.iloc[previous_index]["pred"].squeeze(), data["ego_matrix_previous"], data["ego_matrix_current"])

                            detection_ours, detection_keyframes, copycat_information=determine_copycat(args,predictions_lst,data["ego_waypoints"].squeeze(),previous_predictions_lst[current_index],
                                                                                                    previous_gt_lst,params["keyframes_correlations"][current_index],current_index,params)
                            if args.visualize_combined or args.visualize_without_rgb:
                                if config.detectboxes:
                                    if "arp" in config.baseline_folder_name:
                                        batch_of_bbs_pred=policy.module.convert_features_to_bb_metric(predictions_lst[current_index]["pred_bb"]["pred_bb"])
                                    else:
                                        batch_of_bbs_pred=model.module.convert_features_to_bb_metric(predictions_lst[current_index]["pred_bb"]["pred_bb"])
                                else:
                                    batch_of_bbs_pred=None
                                visualize_model(args=args,config=config, save_path_root=cc_save_path, rgb=image_sequence, lidar_bev=data["lidar"],
                                        pred_wp_prev=previous_predictions_lst[current_index],
                                        gt_bev_semantic=data["bev_semantic"] if not config.bev else None, step=current_index,
                                        target_point=data["target_point"], pred_wp=predictions_lst[current_index]["pred"]["wp_predictions"],
                                        gt_wp=data["ego_waypoints"],parameters=copycat_information,
                                        detect_our=detection_ours, detect_kf=detection_keyframes,frame=current_index,
                                        pred_bb=batch_of_bbs_pred,
                                        pred_bev_semantic=predictions_lst[current_index]["pred"]["pred_bev_semantic"] if config.bev else None,
                                        gt_bbs=data["bounding_boxes"] if config.detectboxes else None,
                                        prev_gt=previous_gt_lst[current_index],loss=predictions_lst[current_index]["loss"], condition=args.second_cc_condition,
                                        ego_speed=data["speed"], correlation_weight=params["keyframes_correlations"][current_index],
                                        loss_brake=predictions_lst[current_index]["head_loss"]["loss_brake"] if predictions_lst[current_index]["head_loss"] is not None else None,
                                        loss_velocity=predictions_lst[current_index]["head_loss"]["loss_velocity"] if predictions_lst[current_index]["head_loss"] is not None else None)
                for metric, count, pos in zip(["our", "kf"], [len(our_cc_positions), len(keyframes_cc_positions)], [our_cc_positions,keyframes_cc_positions]):
                    results=results.append({"baseline":config.baseline_folder_name,
                                            "experiment": config.experiment,
                                            "training_repetition": config.training_repetition,
                                            "metric":metric, "length": count, "positions": pos,
                                             "velocity_mean": np.array(velocity_losses).mean() if velocity_losses else None,
                                             "velocity_std": np.array(velocity_losses).std() if velocity_losses else None,
                                             "accel_mean": np.array(accel_losses).mean() if accel_losses else None,
                                             "accel_std": np.array(accel_losses).std() if accel_losses else None}, ignore_index=True)
    if args.save_whole_scene:
        generate_video_stacked()
    results.to_csv(os.path.join(os.environ.get("WORK_DIR"),"visualisation", "open_loop", "metric_results.csv"), index=False)
    
if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-whole-scene",
                        type=int,
                        default=0)
    parser.add_argument(
        "--custom-validation",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num-surrounding-frames",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--keyframes-threshold",
        type=str,
        default="relative",
        choices=['relative', 'absolute']
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="02_withheld",
    )
    parser.add_argument(
        "--training_repetition",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--pred-tuning-parameter",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--visualize-without-rgb",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--visualize-combined",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--second-cc-condition",
        type=str,
        default="loss",
    )
    parser.add_argument(
        "--tuning-parameter_2",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--norm",
        type=int,
        default=2,

    )
    parser.add_argument(
        "--number-of-workers",
        type=int,
        default=12,

    )

    
    arguments = parser.parse_args()
    main(arguments)