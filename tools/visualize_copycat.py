import os
import pickle
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from coil_utils.baseline_helpers import visualize_model, get_ablations_dict,set_not_included_ablation_args, get_latest_saved_checkpoint, SequentialSampler
from tools.video_generation import generate_video_stacked
from coil_utils.copycat_helper import evaluate_baselines_and_save_predictions, preprocess, determine_copycat
from team_code.data import CARLA_Data
from team_code.timefuser_model import TimeFuser


def load_image_sequence(config, predictions_lst, current_iteration):
    root = os.path.dirname(predictions_lst[current_iteration]["image"])
    index_in_dataset = int(os.path.basename(predictions_lst[current_iteration]["image"]).replace(".jpg", ""))
    return np.concatenate(
        [Image.open(os.path.join(root, f"{index_in_dataset - i:04d}.jpg")) for i in reversed(range(config.img_seq_len))], axis=0), root


@torch.no_grad()
def load_model_and_checkpoint(config, root):
    checkpoint_file = get_latest_saved_checkpoint(root)
    checkpoint_path = os.path.join(root, "checkpoints", f"{checkpoint_file}.pth")
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    print(f"Loaded checkpoint_{checkpoint_file}")

    if "arp" in config.baseline_folder_name:
        policy = TimeFuser("arp-policy", config)
        policy.to("cuda:0")
        policy = DDP(policy, device_ids=["cuda:0"])
        mem_extract = TimeFuser("arp-memory", config)
        mem_extract.to("cuda:0")
        policy = DDP(mem_extract, device_ids=["cuda:0"])
        #del checkpoint["policy_state_dict"]["module.time_position_embedding"]
        #del checkpoint["mem_extract_state_dict"]["module.time_position_embedding"]
        policy.load_state_dict(checkpoint["policy_state_dict"])
        mem_extract.load_state_dict(checkpoint["mem_extract_state_dict"])

        return policy, mem_extract, checkpoint
    else:
        model = TimeFuser(config.baseline_folder_name, config, rank=0)
        model.to("cuda:0")
        model = DDP(model, device_ids=["cuda:0"])
        #del checkpoint["state_dict"]["module.time_position_embedding"]
        return model,None, checkpoint



def initialize_data_loader(config, args):
    val_set = CARLA_Data(root=config.val_data, config=config, rank=0, baseline=config.baseline_folder_name)
    sampler_val = SequentialSampler(val_set)
    # val_set.set_correlation_weights(path=os.path.join(
    #     os.environ.get("WORK_DIR"),
    #     "keyframes",
    #     "importance_weights",
    #     "bcoh_weights_copycat_prev9_rep0_neurons300.npy",
    # ))
    return torch.utils.data.DataLoader(
        val_set,
        batch_size=1,
        num_workers=args.number_of_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
        sampler=sampler_val,
    )


def process_data_loader(data_loader_val, config, params, args, predictions_lst, previous_predictions_lst, previous_gt_lst, model, false_positives):
    keyframes_cc_positions = []
    our_cc_positions = []
    head_losses = []
    our_cc_losses = []

    for data_loader_position, (data, image_path, keyframe_correlation) in enumerate(
            zip(tqdm(data_loader_val), data_loader_val.dataset.images, params["keyframes_correlations"])):
        route_name = os.path.basename(os.path.dirname(os.path.dirname(str(image_path, encoding="utf-8"))))
        cc_save_path = os.path.join(os.environ.get("WORK_DIR"), "visualisation", "open_loop", config.baseline_folder_name, config.experiment, route_name)
        detection_ours, detection_keyframes, copycat_information = process_data_sample(
            data_loader_position, data, image_path, keyframe_correlation, config, params,
              args, predictions_lst, previous_predictions_lst, previous_gt_lst, model,cc_save_path, false_positives)

        if detection_keyframes:
            keyframes_cc_positions.append(data_loader_position)
        if detection_ours:
            our_cc_positions.append(data_loader_position)
            add_loss_to_list(predictions_lst[data_loader_position], our_cc_losses)
        if predictions_lst[data_loader_position]["head_loss"] is not None:
            head_losses.append(predictions_lst[data_loader_position]["head_loss"])
        if detection_keyframes or detection_ours:
            process_copycat_frames(detection_ours, detection_keyframes, data_loader_position, config,
                                    params, args, data_loader_val, predictions_lst, previous_predictions_lst, previous_gt_lst, model,cc_save_path)

    return keyframes_cc_positions, our_cc_positions, head_losses, our_cc_losses


def process_data_sample(data_loader_position, data, image_path, keyframe_correlation, config, params, args,
                        predictions_lst, previous_predictions_lst, previous_gt_lst, model,cc_save_path, false_positives):
    data_image = str(image_path, encoding="utf-8").replace("\x00", "")
    current_index = data_loader_position

    pred_image = predictions_lst[current_index]["image"]
    assert data_image == pred_image, "Image paths do not match"

    detection_ours, detection_keyframes, copycat_information = determine_copycat(args, predictions_lst, data["ego_waypoints"].squeeze().numpy(), previous_predictions_lst[current_index], previous_gt_lst,
                                                                                keyframe_correlation, current_index, params)
    if false_positives is not None:
        for dct in false_positives:
            if dct["metric"]=="our_cc":
                if detection_ours and (data_loader_position not in dct["positions"]):#all([i not in dct["positions"] for i in range(data_loader_position-1, data_loader_position+1)]):
                    detection_ours=True
                else:
                    detection_ours=False
    if args.save_whole_scene:
        if config.img_seq_len < 7:
            empties = np.concatenate([np.zeros_like(Image.open(predictions_lst[0]["image"]))] * (7 - config.img_seq_len))
            image_sequence, root = load_image_sequence(config, predictions_lst, data_loader_position)
            image_sequence = np.concatenate([empties, image_sequence], axis=0)
        else:
            image_sequence, root = load_image_sequence(config, predictions_lst, data_loader_position)

        batch_of_bbs_pred = get_batch_of_bbs_pred(config, predictions_lst, current_index, model)
        if args.visualize_without_rgb or args.visualize_combined:
            visualize_model(args=args, config=config, save_path_root=cc_save_path, rgb=image_sequence, lidar_bev=torch.squeeze(data["lidar"], dim=0),
                            pred_wp_prev=previous_predictions_lst[current_index],
                            gt_bev_semantic=torch.squeeze(data["bev_semantic"], dim=0) if not config.bev else None, step=current_index,
                            target_point=torch.squeeze(data["target_point"], dim=0), pred_wp=predictions_lst[current_index]["pred"]["wp_predictions"],
                            gt_wp=torch.squeeze(data["ego_waypoints"], dim=0), parameters=copycat_information,
                            detect_our=detection_ours, detect_kf=detection_keyframes, frame=current_index,
                            pred_bb=batch_of_bbs_pred,
                            pred_bev_semantic=predictions_lst[current_index]["pred"]["pred_bev_semantic"] if config.bev else None,
                            gt_bbs=data["bounding_boxes"] if config.detectboxes else None,
                            prev_gt=previous_gt_lst[current_index], loss=predictions_lst[current_index]["loss"], condition=args.second_cc_condition,
                            ego_speed=data["speed"].numpy()[0], correlation_weight=params["keyframes_correlations"][current_index],
                            loss_brake=predictions_lst[current_index]["head_loss"]["loss_brake"] if predictions_lst[current_index]["head_loss"] is not None else None,
                            loss_velocity=predictions_lst[current_index]["head_loss"]["loss_velocity"] if predictions_lst[current_index]["head_loss"] is not None else None)

    return detection_ours, detection_keyframes, copycat_information


def get_batch_of_bbs_pred(config, predictions_lst, current_index, model):
    if config.detectboxes:
        return model.module.convert_features_to_bb_metric(predictions_lst[current_index]["pred_bb"])
    return None


def add_loss_to_list(prediction, our_cc_losses):
    if "detailed_loss" in prediction:
        our_cc_losses.append(prediction["detailed_loss"]["wp_loss"].item())
    else:
        our_cc_losses.append(prediction["loss"].item())


def process_copycat_frames(detection_ours, detection_keyframes, data_loader_position, config, params, args, data_loader_val,
                           predictions_lst, previous_predictions_lst, previous_gt_lst, model, cc_save_path):
    for i in range(-args.num_surrounding_frames, args.num_surrounding_frames + 1):
        detection_ours, detection_keyframes = False, False
        current_index = data_loader_position + i

        if current_index < 0 or current_index >= len(data_loader_val.dataset):
            continue

        data = data_loader_val.dataset.__getitem__(current_index)

        if config.img_seq_len < 7:
            empties = np.concatenate([np.zeros_like(Image.open(predictions_lst[0]["image"]))] * (7 - config.img_seq_len))
            image_sequence, root = load_image_sequence(config, predictions_lst, current_index)
            image_sequence = np.concatenate([empties, image_sequence], axis=0)
        else:
            image_sequence, root = load_image_sequence(config, predictions_lst, current_index)

        detection_ours, detection_keyframes, copycat_information = determine_copycat(
            args, predictions_lst, data["ego_waypoints"].squeeze(), previous_predictions_lst[current_index],
            previous_gt_lst, params["keyframes_correlations"][current_index], current_index, params)

        batch_of_bbs_pred = get_batch_of_bbs_pred(config, predictions_lst, current_index, model)
        if args.visualize_without_rgb or args.visualize_combined:
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


def save_results(results,config, ablations_dict,keyframes_cc_positions, our_cc_positions, our_cc_losses, head_losses):
    if config.detectboxes or config.bev:
        sum_dict={key: 0 for key in head_losses[0].keys()}
        count_dict={key: 0 for key in head_losses[0].keys()}
        for loss_dict in head_losses:
            for key, value in loss_dict.items():
                sum_dict[key]+=value
                count_dict[key]+=1
        mean_dict={key: sum_dict[key]/count_dict[key] for key in sum_dict}
        results.append({
            "baseline": config.baseline_folder_name,
            **ablations_dict,
            "experiment": config.experiment,
            "metric": "our_cc",
            "length": len(our_cc_positions),
            "positions": our_cc_positions,
            "losses": np.mean(our_cc_losses) if len(our_cc_losses) > 0 else None,
            **mean_dict
        })
        results.append({
        "baseline": config.baseline_folder_name,
        **ablations_dict,
        "experiment": config.experiment,
        "metric": "kf_cc",
        "length": len(keyframes_cc_positions),
        "positions": keyframes_cc_positions,
        "losses": np.nan,
        **mean_dict
    })
    else:
        results.append({
            "baseline": config.baseline_folder_name,
            **ablations_dict,
            "experiment": config.experiment,
            "metric": "our_cc",
            "length": len(our_cc_positions),
            "positions": our_cc_positions,
            "losses": np.mean(our_cc_losses) if len(our_cc_losses) > 0 else None,
        })

        results.append({
            "baseline": config.baseline_folder_name,
            **ablations_dict,
            "experiment": config.experiment,
            "metric": "kf_cc",
            "length": len(keyframes_cc_positions),
            "positions": keyframes_cc_positions,
            "losses": np.nan,
        })
def get_false_positives(args,path, params, results):
    for root, dirs, files in os.walk(path):
        if "config_cc.pkl" in files:
            with open(os.path.join(root, "config_cc.pkl"), 'rb') as f:
                config = pickle.load(f)
            config.number_previous_waypoints = 1
            config.visualize_copycat = True
            set_not_included_ablation_args(config)
            ablations_dict=get_ablations_dict()
            current_ablations_dict={}
            for ablation, _ in ablations_dict.items():
                current_ablations_dict.update({ablation:getattr(config, ablation)})
            current_ablations_dict.update({"training_repetition": config.training_repetition})
            if config.baseline_folder_name!="bcso":
                continue
            else:
                # if not is_valid_config(config):
                #     continue
                model, _, checkpoint = load_model_and_checkpoint(config, root)
                with open(os.path.join(root, "predictions_all.pkl"), 'rb') as f:
                    predictions_lst = pickle.load(f)
                with open(os.path.join(root, "aligned_gt_all.pkl"), 'rb') as f:
                    previous_gt_lst = pickle.load(f)
                with open(os.path.join(root, "aligned_predictions_all.pkl"), 'rb') as f:
                    previous_predictions_lst = pickle.load(f)

                    data_loader_val = initialize_data_loader(config, args)
                keyframes_cc_positions, our_cc_positions, head_losses, our_cc_losses = process_data_loader(
                    data_loader_val, config, params, args, predictions_lst, previous_predictions_lst, previous_gt_lst, model, None)

            save_results(results,config,current_ablations_dict, keyframes_cc_positions, our_cc_positions, our_cc_losses, head_losses)
def get_false_positives_for_specific_ablation(bcso_results,current_ablations_dict):
    ret_lst=[]
    for dct in bcso_results:
        if all([dct[ablation]==value for ablation, value in current_ablations_dict.items()]):
            ret_lst.append(dct)
    return ret_lst

@torch.no_grad()
def main(args):
    bcso_results=[]
    path_of_baselines = os.path.join(os.path.join(os.environ.get("WORK_DIR"), "_logs"))
    model = evaluate_baselines_and_save_predictions(args, path_of_baselines)
    params = preprocess(args)
    get_false_positives(args,path_of_baselines, params, bcso_results)
    results=[]
    for root, dirs, files in os.walk(path_of_baselines):
        if "config_cc.pkl" in files:
            with open(os.path.join(root, "config_cc.pkl"), 'rb') as f:
                config = pickle.load(f)
            config.number_previous_waypoints = 1
            config.visualize_copycat = True
            config.batch_size=1
            set_not_included_ablation_args(config)
            ablations_dict=get_ablations_dict()
            current_ablations_dict={}
            for ablation, _ in ablations_dict.items():
                current_ablations_dict.update({ablation:getattr(config, ablation)})
            current_ablations_dict.update({"training_repetition": config.training_repetition})
            # if not is_valid_config(config):
            #     continue

            model, _, checkpoint = load_model_and_checkpoint(config, root)

            with open(os.path.join(root, "predictions_all.pkl"), 'rb') as f:
                predictions_lst = pickle.load(f)
            with open(os.path.join(root, "aligned_gt_all.pkl"), 'rb') as f:
                previous_gt_lst = pickle.load(f)
            with open(os.path.join(root, "aligned_predictions_all.pkl"), 'rb') as f:
                previous_predictions_lst = pickle.load(f)

            data_loader_val = initialize_data_loader(config, args)
            false_positives=get_false_positives_for_specific_ablation(bcso_results,current_ablations_dict)
            keyframes_cc_positions, our_cc_positions, head_losses, our_cc_losses = process_data_loader(
                data_loader_val, config, params, args, predictions_lst, previous_predictions_lst, previous_gt_lst, model, false_positives)

            save_results(results,config, current_ablations_dict,keyframes_cc_positions, 
                         our_cc_positions, our_cc_losses, head_losses)
    # if args.save_whole_scene:
    #     generate_video_stacked()
    results=pd.DataFrame(results)
    results.to_csv(os.path.join(os.environ.get("WORK_DIR"), "visualisation", "open_loop", "metric_results.csv"), index=False)


# def is_valid_config(config):
#     if config.freeze == 1 or config.bev == 0 or config.detectboxes == 0 or config.training_repetition != 0:
#         return False
#     return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-whole-scene", type=int, default=0)
    parser.add_argument("--custom-validation", type=int, default=0)
    parser.add_argument("--num-surrounding-frames", type=int, default=0)
    parser.add_argument("--keyframes-threshold", type=str, default="relative", choices=['relative', 'absolute'])
    parser.add_argument("--setting", type=str, default="02_withheld")
    parser.add_argument("--training_repetition", type=int, default=0)
    parser.add_argument("--pred-tuning-parameter", type=float, default=1)
    parser.add_argument("--visualize-without-rgb", type=float, default=1)
    parser.add_argument("--visualize-combined", type=float, default=1)
    parser.add_argument("--second-cc-condition", type=str, default="loss")
    parser.add_argument("--tuning-parameter_2", type=float, default=1)
    parser.add_argument("--norm", type=int, default=2)
    parser.add_argument("--number-of-workers", type=int, default=12)

    arguments = parser.parse_args()
    main(arguments)
