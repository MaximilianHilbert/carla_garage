
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from coil_utils.baseline_helpers import visualize_model, norm
import os
import csv
import pickle
from coil_utils.baseline_helpers import get_copycat_criteria
from team_code.data import CARLA_Data
from torch.utils.data import Sampler
class SequentialSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)

    
def preprocess(args):
    std_lst=[]
    mean_lst=[]

    loss_std_lst=[]
    loss_mean_lst=[]
    for baseline in args.baselines:
        basename=os.path.join(os.environ.get("WORK_DIR"),
                            "_logs",
                            baseline,
                            f"{baseline}")
        with open(f"{basename}_predictions_std_all.pkl", 'rb') as f:
            criterion_dict = pickle.load(f)
            std_lst.append(criterion_dict["std_pred"])
            mean_lst.append(criterion_dict["mean_pred"])

            loss_std_lst.append(criterion_dict["loss_std"])
            loss_mean_lst.append(criterion_dict["loss_mean"])
            
    return np.mean(np.array(std_lst)),np.mean(np.array(mean_lst)), criterion_dict["std_gt"], criterion_dict["mean_gt"], np.mean(np.array(loss_std_lst)), np.mean(np.array(loss_mean_lst))

def load_image_sequence(config,df_data,current_iteration):
    root=os.path.dirname(df_data.iloc[current_iteration]["image"])
    index_in_dataset=int(os.path.basename(df_data.iloc[current_iteration]["image"]).replace(".jpg",""))
    return np.concatenate([Image.open(os.path.join(root, "0"*(4-len(str(index_in_dataset-i)))+f"{index_in_dataset-i}.jpg"))  for i in reversed(range(config.img_seq_len))],axis=0), root


def main(args):
    avg_of_std_baseline_predictions, avg_of_avg_baseline_predictions, std_gt, avg_gt, loss_avg_of_std,loss_avg_of_avg=preprocess(args)
    
    keyframe_correlations=np.load(os.path.join(
            os.environ.get("WORK_DIR"),
            "_logs",
            "keyframes",
            f"repetition_0",
            f"bcoh_weights_prev9_rep0_neurons300.npy",
        ))
    
    paths=[]
    for baseline in args.baselines:
        basename=os.path.join(os.environ.get("WORK_DIR"),
                            "_logs",
                            baseline,
                            baseline)
    
        with open(f"{basename}_config.pkl", 'rb') as f:
            config = pickle.load(f)
        config.number_previous_waypoints=1
        config.visualize_copycat=True
        if not args.custom_validation:
            with open(f"{basename}_predictions_all.pkl", 'rb') as f:
                wp_dict = pickle.load(f)
        else:
            with open(f"{basename}_predictions_cc_routes_only.pkl", 'rb') as f:
                wp_dict = pickle.load(f)

        data_df = pd.DataFrame.from_dict(wp_dict, orient='index', columns=['image','pred', 'gt', 'loss'])
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
        data_loader_val = torch.utils.data.DataLoader(
            val_set,
            batch_size=1,
            num_workers=args.number_of_workers,
            pin_memory=True,
            shuffle=False,  # because of DDP
            drop_last=True,
            sampler=sampler_val,
        )
        count=0
        
        for data_loader_position, (data, image_path) in enumerate(zip(tqdm(data_loader_val),data_loader_val.dataset.images)):
            if data_loader_position==0:
                continue
            data_image=str(image_path, encoding="utf-8").replace("\x00", "")
            current_index=data_loader_position
            previous_index=data_loader_position-1
            pred_image=data_df.iloc[current_index]["image"]
            if data_image!=pred_image:
                assert("not aligned")
    
            #only red lights 0.01 1
            #0.2 0.5
            #single curve 0.2 ung std_value gt 1
            #oder 0.1
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

            
            previous_prediction_aligned=align_previous_prediction(data_df.iloc[previous_index]["pred"][0], data["ego_matrix_previous"].detach().cpu().numpy()[0], data["ego_matrix_current"].detach().cpu().numpy()[0])
            if config.img_seq_len<7:
                empties=np.concatenate([np.zeros_like(Image.open(data_df.iloc[0]["image"]))]*(7-config.img_seq_len))
                image_sequence,root=load_image_sequence(config,data_df, data_loader_position)
                image_sequence=np.concatenate([empties, image_sequence], axis=0)
            else:
                image_sequence,root=load_image_sequence(config,data_df, data_loader_position)
            pred_residual=norm(data_df.iloc[current_index]["pred"]-previous_prediction_aligned, ord=args.norm)
            gt_residual=norm(data["ego_waypoints"][0].detach().cpu().numpy()-data["previous_ego_waypoints"][0].detach().cpu().numpy(), ord=args.norm)
            condition_value_1=avg_of_avg_baseline_predictions-avg_of_std_baseline_predictions*args.pred_tuning_parameter
            condition_1=pred_residual<condition_value_1
            if args.second_cc_condition=="loss":
                condition_value_2=loss_avg_of_avg+loss_avg_of_std*args.tuning_parameter_2
                condition_2=data_df.iloc[current_index]["loss"]>condition_value_2
            else:
                condition_value_2=avg_gt+std_gt*args.tuning_parameter_2
                condition_2=gt_residual>condition_value_2
            
            if condition_1 and condition_2 and data["speed"].numpy()[0]>0.05:
                #0.15 and 1 for the one curve only
                count+=1
                if not args.custom_validation:
                    paths.append(os.path.dirname(root))
                for i in range(-5,6):
                    data=data_loader_val.dataset.__getitem__(data_loader_position+i)
                    
                    previous_index=data_loader_position+i-1
                    current_index=data_loader_position+i
                    if config.img_seq_len<7:
                        empties=np.concatenate([np.zeros_like(Image.open(data_df.iloc[0]["image"]))]*(7-config.img_seq_len))
                        image_sequence,root=load_image_sequence(config,data_df, data_loader_position+i)
                        image_sequence=np.concatenate([empties, image_sequence], axis=0)
                    else:
                        image_sequence,root=load_image_sequence(config,data_df, data_loader_position+i)
                    if i==0:
                        detection=True
                    else:
                        detection=False
                    previous_prediction_aligned=align_previous_prediction(data_df.iloc[previous_index]["pred"].squeeze(), data["ego_matrix_previous"], data["ego_matrix_current"])
                    visualize_model(config=config, save_path_with_rgb=os.path.join(os.environ.get("WORK_DIR"),"vis_rgb",baseline),save_path_without_rgb=os.path.join(os.environ.get("WORK_DIR"),"vis_no_rgb",baseline), rgb=image_sequence, lidar_bev=data["lidar"],
                            pred_wp_prev=np.squeeze(previous_prediction_aligned),
                            gt_bev_semantic=data["bev_semantic"], step=current_index,
                            target_point=data["target_point"], pred_wp=np.squeeze(data_df.iloc[current_index]["pred"]),
                            gt_wp=data["ego_waypoints"],pred_residual=pred_residual,
                            gt_residual=gt_residual,copycat_count=count, detect=detection, frame=data_loader_position,
                            prev_gt=data["previous_ego_waypoints"],loss=data_df.iloc[current_index]["loss"], condition=args.second_cc_condition,
                            condition_value_1=condition_value_1, condition_value_2=condition_value_2, ego_speed=data["speed"], correlation_weight=keyframe_correlations[data_loader_position-i])
        print(f"count for real copycat for baseline {baseline}: {count}")
    if not args.custom_validation:
        with open(os.path.join(os.environ.get("WORK_DIR"),
                            "_logs",
                            "detected_cc_dirs.csv"), "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(set(paths))
if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--custom-validation",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--pred-tuning-parameter",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--second-cc-condition",
        type=str,
        default="gt",
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