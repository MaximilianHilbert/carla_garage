
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from coil_utils.baseline_helpers import visualize_model
import matplotlib.pyplot as plt
import os

import pickle
# Read the CSV file into a DataFrame
def norm(differences, ord):
    if ord==1:
        return np.sum(np.absolute(differences))
    if ord==2:
        return np.sqrt(np.sum(np.absolute(differences)**2))
    
def visualize_everything(args,config, data, branches, targets, iteration):
    rgb=np.concatenate()
    visualize_model(config=config,save_path=os.path.join(os.environ.get("WORK_DIR"),"vis",args.baseline_folder_name),
                                        rgb=rgb.detach().cpu().numpy(),lidar_bev=data["lidar"],gt_bev_semantic=data["bev_semantic"],
                                        step=iteration, target_point=data["target_point"],pred_wp=branches, gt_wp=targets)
def generate_metric(config,data, vis_dict, baseline, visualize_non_copycat, visualize_copycat):
    # with open(f"{basename}_config.pkl", 'rb') as f:
    #     config_dict = pickle.load(f)
    
    # basename=os.path.join(os.environ.get("WORK_DIR"),
    #                     "_logs",
    #                     args.baseline_folder_name,args.experiment,
    #                     f"repetition_{str(args.training_repetition)}", f"{args.setting}",f"{args.baseline_folder_name}_{args.experiment}")
    # with open(os.path.join(f"{basename}_wp.pkl"),'rb') as f:
    #     data = pickle.load(f)
    #     data_df = pd.DataFrame.from_dict(data, orient='index', columns=['pred', 'gt', 'loss']).reset_index()
    # with open(f"{basename}_vis.pkl", 'rb') as f:
    #     vis_dict = pickle.load(f)


    last_and_future=0
    which_norm=2
    #fig,axs=plt.subplots(3,2, figsize=(15,10), sharex=True, sharey=True)
    data_df = pd.DataFrame.from_dict(data, orient='index', columns=['pred', 'gt', 'loss']).reset_index()
    data_df['running_diff'] = data_df['pred'].diff()
    data_df['running_diff']=data_df['running_diff'].apply(lambda x: norm(x,ord=which_norm))
    data_df['running_diff_gt'] = data_df['gt'].diff()
    data_df['running_diff_gt']=data_df['running_diff_gt'].apply(lambda x: norm(x,ord=which_norm))
    mean_value_gt=np.mean(data_df["running_diff_gt"])
    mean_value_data=np.mean(data_df["running_diff"])
    
    std_value_gt=np.std(data_df["running_diff_gt"])
    std_value_data=np.std(data_df["running_diff"])

    # ax[0].set_title(f"{baseline} residual")
    # ax[0].hist(data_copy["running_diff"],bins=int(np.sqrt(len(data_copy["running_diff"]))), alpha=0.4, color="green", label="pred")
    # ax[0].hist(data_copy["running_diff_gt"],bins=int(np.sqrt(len(data_copy["running_diff_gt"]))), alpha=0.4, color="blue", label="gt")
    # ax[0].legend()
    # ax[1].set_title(f"{baseline} loss")
    # ax[1].hist(data_copy["loss"],bins=int(np.sqrt(len(data_copy["loss"]))), alpha=0.4, color="green", label="loss")
    # ax[1].legend()
    #only red lights 0.01 1
    #0.2 0.5
    #single curve 0.2 ung std_value gt 1
    #oder 0.1
    count=0
    for i in tqdm(range(7,len(data_df["pred"]))):
        image_path=data_df.iloc[i]["index"]
        prev_image_path=data_df.iloc[i-1]["index"]
        if config.img_seq_len<7:
            temporal_images=np.concatenate([np.zeros_like(Image.open(data_df.iloc[0]["index"]))]*(7-config.img_seq_len))
            temporal_images=np.concatenate([temporal_images,np.concatenate([np.array(Image.open(data_df.iloc[i-q]["index"])) for q in reversed(range(config.img_seq_len))], axis=0)])
        else:
            temporal_images=np.concatenate([np.array(Image.open(data_df.iloc[i-q]["index"])) for q in reversed(range(config.img_seq_len))], axis=0)
        pred_residual=norm(data_df["pred"][i]-data_df["pred"][i-1], ord=which_norm)
        gt_residual=norm(data_df["gt"][i]-data_df["gt"][i-1], ord=which_norm)
        if visualize_non_copycat:
            visualize_model(config=config, save_path=os.path.join(os.environ.get("WORK_DIR"),"vis",baseline), rgb=temporal_images, lidar_bev=torch.Tensor(vis_dict[image_path]["lidar_bev"]),
                                    pred_wp_prev=torch.Tensor(data_df[data_df["index"]==prev_image_path]["pred"].to_numpy()[0]),gt_bev_semantic=torch.ByteTensor(vis_dict[image_path]["bev_semantic"]), step=i, target_point=torch.Tensor(vis_dict[image_path]["target_point"]), pred_wp=torch.Tensor(data_df[data_df["index"]==image_path]["pred"].to_numpy()[0]),
                                    gt_wp=torch.Tensor(data_df[data_df["index"]==image_path]["gt"].to_numpy()[0]),pred_residual=pred_residual, gt_residual=gt_residual,copycat_count=count)
        if pred_residual<std_value_data and gt_residual>std_value_gt:
            #0.15 and 1 for the one curve only
            count+=1
            # for j in reversed(range(1,last_and_future+1)):
            #     image_path=data_df.iloc[i-j]["index"].replace("\x00", "")
                
            #     image=Image.open(image_path)

            #     visualize_model(config=config, save_path=os.path.join(os.environ.get("WORK_DIR"),"vis",baseline), rgb=torch.Tensor(np.array(image)).permute(2,0,1), lidar_bev=torch.Tensor(vis_dict[image_path]["lidar_bev"]),
            #                     gt_bev_semantic=torch.ByteTensor(vis_dict[image_path]["bev_semantic"]), step=i-j, target_point=torch.Tensor(vis_dict[image_path]["target_point"]), pred_wp=torch.Tensor(data_df[data_df["index"]==image_path]["pred"].to_numpy()[0]),
            #                     gt_wp=torch.Tensor(data_df[data_df["index"]==image_path]["gt"].to_numpy()[0]), copycat_count=count)
            
            prev_image_path=data_df.iloc[i-1]["index"]
            temporal_images=np.concatenate([np.array(Image.open(data_df.iloc[i-q]["index"])) for q in reversed(range(config.img_seq_len))], axis=0)
            if visualize_non_copycat or visualize_copycat:
                visualize_model(config=config, save_path=os.path.join(os.environ.get("WORK_DIR"),"vis",baseline), rgb=temporal_images, lidar_bev=torch.Tensor(vis_dict[image_path]["lidar_bev"]),
                                gt_bev_semantic=torch.ByteTensor(vis_dict[image_path]["bev_semantic"]), step=i, imprint=True,target_point=torch.Tensor(vis_dict[image_path]["target_point"]),
                                pred_wp=torch.Tensor(data_df[data_df["index"]==image_path]["pred"].to_numpy()[0]),
                                pred_wp_prev=torch.Tensor(data_df[data_df["index"]==prev_image_path]["pred"].to_numpy()[0]),
                                gt_wp=torch.Tensor(data_df[data_df["index"]==image_path]["gt"].to_numpy()[0]), pred_residual=pred_residual, gt_residual=gt_residual,copycat_count=count, detect=True)
    print(f"count for real copycat for baseline {baseline}: {count}")
# if __name__=="__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--training-repetition",
#         type=int,
#         default=0,
#         required=True,
#     )
#     parser.add_argument(
#         "--baseline-folder-name",
#         default=None,
#         required=True,
#     )
#     parser.add_argument(
#         "--experiment",
#         default=None,
#         required=True,
#         help="filename of experiment without .yaml",
#     )
#     parser.add_argument(
#         "--setting",
#         type=str,
#         default="all",
#         help="coil requires to be trained on Town01 only, so Town01 are train conditions and Town02 is Test Condition",
#     )
#     arguments = parser.parse_args()
#     generate_metric(arguments)