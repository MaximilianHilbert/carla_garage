
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from coil_utils.baseline_helpers import visualize_model, norm
import os
import csv
import pickle
from scipy.special import softmax
from tools.video_generation import generate_video_stacked
import re
from team_code.data import CARLA_Data
from coil_utils.baseline_helpers import get_action_predict_loss_threshold
from torch.utils.data import Sampler
class SequentialSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)

def align_previous_prediction(pred, matrix_previous, matrix_current):
    matrix_prev = matrix_previous[:3]
    translation_prev = matrix_prev[:, 3:4].flatten()
    rotation_prev = matrix_prev[:, :3]

    matrix_curr = matrix_current[:3]
    translation_curr = matrix_curr[:, 3:4].flatten()
    rotation_curr = matrix_curr[:, :3]

    waypoints=[]
    for waypoint in pred:
        waypoint_3d=np.append(waypoint, 0)
        waypoint_world_frame = (rotation_prev@waypoint_3d) + translation_prev
        waypoint_aligned_frame=rotation_curr.T @(waypoint_world_frame-translation_curr)
        waypoints.append(waypoint_aligned_frame[:-1])
    return np.array(waypoints)

def determine_copycat(args,predictions_lst,current_waypoints_gt,previous_prediction_aligned,previous_gt_lst,keyframe_correlation,current_index,params):
    detection_keyframes,detection_ours=False,False
    pred_residual=norm(predictions_lst[current_index]["pred"]-previous_prediction_aligned, ord=args.norm)
    gt_residual=norm(current_waypoints_gt-previous_gt_lst[current_index], ord=args.norm)
    
    condition_value_1=params["avg_of_avg_baseline_predictions"]-params["avg_of_std_baseline_predictions"]*args.pred_tuning_parameter
    condition_1=pred_residual<condition_value_1
    condition_value_keyframes=params["avg_of_kf"]+params["std_of_kf"]*args.pred_tuning_parameter
    if args.keyframes_threshold=="absolute":
        condition_keyframes=True if keyframe_correlation==5.0 else False
    else:
        condition_keyframes=keyframe_correlation>condition_value_keyframes
    if args.second_cc_condition=="loss":
        condition_value_2=params["loss_avg_of_avg"]+params["loss_avg_of_std"]*args.tuning_parameter_2
        condition_2=predictions_lst[current_index]["loss"]>condition_value_2
    else:
        condition_value_2=params["avg_gt"]+params["std_gt"]*args.tuning_parameter_2
        condition_2=gt_residual>condition_value_2
    if condition_1 and condition_2:
        detection_ours=True
    if condition_keyframes:
        detection_keyframes=True
    return detection_ours, detection_keyframes, {"pred_residual":pred_residual, "gt_residual": gt_residual, "condition_value_1": condition_value_1, "condition_value_2": condition_value_2, "condition_value_keyframes": condition_value_keyframes}


def preprocess(args):
    
    baseline_dict={}
    keyframe_correlations=np.load(os.path.join(
            os.environ.get("WORK_DIR"),
            "_logs",
            "waypoint_weight_generation",
            f"repetition_0",
            f"bcoh_weights_copycat_prev9_rep0_neurons300.npy",
        ))
    
    if args.keyframes_threshold=="absolute":
        threshold_ratio=0.1
        importance_sampling_threshold_weight=5.0
        importance_sampling_threshold=get_action_predict_loss_threshold(keyframe_correlations,threshold_ratio)
        keyframe_correlations=(keyframe_correlations > importance_sampling_threshold)* (importance_sampling_threshold_weight - 1) + 1        
    path_of_baselines=os.path.join(os.path.join(os.environ.get("WORK_DIR"),
                    "_logs",
        ))
    for baseline in os.listdir(path_of_baselines):
        if baseline!="waypoint_weight_generation":
            experiment_path=os.path.join(os.path.join(os.environ.get("WORK_DIR"),
                        "_logs",
                        baseline))
            for experiment in os.listdir(os.path.join(experiment_path)):
                basename=os.path.join(path_of_baselines,experiment_path,experiment, f"repetition_{str(args.training_repetition)}",
                        args.setting)
        
                with open(os.path.join(basename,"predictions_std_all.pkl"), 'rb') as f:
                    criterion_dict = pickle.load(f)
                baseline_dict[basename]={}
                baseline_dict[basename].update({"std_pred": criterion_dict["std_pred"], "mean_pred": criterion_dict["mean_pred"],
                                                "loss_std": criterion_dict["loss_std"], "loss_mean":criterion_dict["loss_mean"]})
            
    ret_dict= {"avg_of_std_baseline_predictions": np.mean(np.array([per_baseline_stats["std_pred"] for per_baseline_stats in baseline_dict.values()])),
            "avg_of_avg_baseline_predictions":np.mean(np.array([per_baseline_stats["mean_pred"] for per_baseline_stats in baseline_dict.values()])),
            "std_gt":criterion_dict["std_gt"], "avg_gt":criterion_dict["mean_gt"], "loss_avg_of_std":np.mean(np.array([per_baseline_stats["loss_std"] for per_baseline_stats in baseline_dict.values()])),
            "loss_avg_of_avg":np.mean(np.array([per_baseline_stats["loss_mean"] for per_baseline_stats in baseline_dict.values()])),
            "avg_of_kf":np.mean(keyframe_correlations),"std_of_kf":np.std(keyframe_correlations), "keyframes_correlations": keyframe_correlations}
    return ret_dict
def load_image_sequence(config,predictions_lst,current_iteration):
    root=os.path.dirname(predictions_lst[current_iteration]["image"])
    index_in_dataset=int(os.path.basename(predictions_lst[current_iteration]["image"]).replace(".jpg",""))
    return np.concatenate([Image.open(os.path.join(root, "0"*(4-len(str(index_in_dataset-i)))+f"{index_in_dataset-i}.jpg"))  for i in reversed(range(config.img_seq_len))],axis=0), root


def main(args):
    params=preprocess(args)
    paths=[]
    results=pd.DataFrame(columns=["baseline", "experiment","metric", "length", "positions"])
    path_of_baselines=os.path.join(os.path.join(os.environ.get("WORK_DIR"),
                    "_logs",
        ))
    for baseline in os.listdir(path_of_baselines):
        if baseline!="waypoint_weight_generation":
            experiment_path=os.path.join(os.path.join(os.environ.get("WORK_DIR"),
                        "_logs",
                        baseline))
            for experiment in os.listdir(experiment_path):
                basename=os.path.join(path_of_baselines,experiment_path,experiment, f"repetition_{str(args.training_repetition)}",
                        args.setting)

                with open(os.path.join(basename, "config_cc.pkl"), 'rb') as f:
                    config = pickle.load(f)
                config.number_previous_waypoints=1
                config.visualize_copycat=True
            
                if not args.custom_validation:
                    with open(os.path.join(basename,"predictions_all.pkl"), 'rb') as f:
                        predictions_lst = pickle.load(f)
                    with open(os.path.join(basename,"aligned_gt_all.pkl"), 'rb') as f:
                        previous_gt_lst = pickle.load(f)
                    with open(os.path.join(basename,"aligned_predictions_all.pkl"), 'rb') as f:
                        previous_predictions_lst = pickle.load(f)
                else:
                    with open(os.path.join(basename,"predictions_cc_routes_only.pkl"), 'rb') as f:
                        wp_dict = pickle.load(f)

                #data_df = pd.DataFrame.from_dict(wp_dict, orient='index', columns=['image','pred', 'gt', 'loss'])
                if args.custom_validation:
                    with open(os.path.join(os.environ.get("WORK_DIR"),
                                    "_logs",
                                    "detected_cc_dirs.csv"), "r", newline="") as file:
                        reader = csv.reader(file)
                        val_lst=[]
                        for row in reader:
                            val_lst.append(row)
                    val_set = CARLA_Data(root=config.val_data, config=config, rank=0,baseline=baseline, custom_validation_lst=val_lst[0])
                else:
                    val_set = CARLA_Data(root=config.val_data, config=config, rank=0,baseline=baseline)
                sampler_val=SequentialSampler(val_set)
                val_set.set_correlation_weights(path=os.path.join(
                    os.environ.get("WORK_DIR"),
                    "_logs",
                    "waypoint_weight_generation",
                    f"repetition_0",
                    f"bcoh_weights_copycat_prev9_rep0_neurons300.npy",
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
                
                
                assert len(params["keyframes_correlations"])==len(data_loader_val.dataset) , "wrong correlation weights selected!"
                for data_loader_position, (data, image_path, keyframe_correlation) in enumerate(zip(tqdm(data_loader_val),data_loader_val.dataset.images, params["keyframes_correlations"])):
                    
                    route_name=os.path.basename(os.path.dirname(os.path.dirname(str(image_path, encoding="utf-8"))))
                    cc_save_path=os.path.join(os.environ.get("WORK_DIR"),"visualisation", "open_loop", baseline,route_name, experiment)
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
                    if args.save_whole_scene:
                        if config.img_seq_len<7:
                            empties=np.concatenate([np.zeros_like(Image.open(predictions_lst[0]["image"]))]*(7-config.img_seq_len))
                            image_sequence,root=load_image_sequence(config,predictions_lst, data_loader_position)
                            image_sequence=np.concatenate([empties, image_sequence], axis=0)
                        else:
                            image_sequence,root=load_image_sequence(config,predictions_lst, data_loader_position)
                        visualize_model(args=args,config=config, save_path_root=cc_save_path, rgb=image_sequence, lidar_bev=torch.squeeze(data["lidar"], dim=0),
                                            pred_wp_prev=previous_predictions_lst[current_index],
                                            gt_bev_semantic=torch.squeeze(data["bev_semantic"], dim=0), step=current_index,
                                            target_point=torch.squeeze(data["target_point"], dim=0), pred_wp=predictions_lst[current_index]["pred"],
                                            gt_wp=torch.squeeze(data["ego_waypoints"], dim=0),parameters=copycat_information,
                                            detect_our=False, detect_kf=False,frame=current_index,
                                            prev_gt=previous_gt_lst[current_index],loss=predictions_lst[current_index]["loss"], condition=args.second_cc_condition,
                                            ego_speed=data["speed"].numpy()[0], correlation_weight=params["keyframes_correlations"][current_index])
                    
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
                                visualize_model(args=args,config=config, save_path_root=cc_save_path, rgb=image_sequence, lidar_bev=data["lidar"],
                                        pred_wp_prev=previous_predictions_lst[current_index],
                                        gt_bev_semantic=data["bev_semantic"], step=current_index,
                                        target_point=data["target_point"], pred_wp=predictions_lst[current_index]["pred"],
                                        gt_wp=data["ego_waypoints"],parameters=copycat_information,
                                        detect_our=detection_ours, detect_kf=detection_keyframes,frame=current_index,
                                        prev_gt=previous_gt_lst[current_index],loss=predictions_lst[current_index]["loss"], condition=args.second_cc_condition,
                                        ego_speed=data["speed"], correlation_weight=params["keyframes_correlations"][current_index])
                for metric, count, pos in zip(["our", "kf"], [len(our_cc_positions), len(keyframes_cc_positions)], [our_cc_positions,keyframes_cc_positions]):
                    results=results.append({"baseline":baseline, "experiment": experiment,"metric":metric, "length": count, "positions": pos}, ignore_index=True)
            if args.save_whole_scene:
                generate_video_stacked(args.baselines, os.path.join(os.environ.get("WORK_DIR"), "visualisation"))

            results.to_csv(os.path.join(os.environ.get("WORK_DIR"),"visualisation", "open_loop", "metric_results.csv"), index=False)
        
            if args.custom_validation:
                with open(os.path.join(os.environ.get("WORK_DIR"),
                                    "_logs",
                                    "detected_cc_dirs.csv"), "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(set(paths))
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
    parser.add_argument(
        "--baselines",
        nargs="+",
        type=str

    )
    
    arguments = parser.parse_args()
    main(arguments)