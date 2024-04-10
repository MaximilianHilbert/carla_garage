
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from coil_utils.baseline_helpers import visualize_model
import os
import pickle
from team_code.data import CARLA_Data
from torch.utils.data import Sampler
class SequentialSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


def norm(differences, ord):
    if ord==1:
        return np.sum(np.absolute(differences))
    if ord==2:
        return np.sqrt(np.sum(np.absolute(differences)**2))
    

def load_image_sequence(config,df_data,current_iteration):
    root=os.path.dirname(df_data.iloc[current_iteration]["image"])
    index_in_dataset=int(os.path.basename(df_data.iloc[current_iteration]["image"]).replace(".jpg",""))
    return np.concatenate([Image.open(os.path.join(root, "0"*(4-len(str(index_in_dataset-i)))+f"{index_in_dataset-i}.jpg"))  for i in reversed(range(config.img_seq_len))],axis=0)



def main(args):
    basename=os.path.join(os.environ.get("WORK_DIR"),
                        "_logs",
                        args.baseline_folder_name,args.experiment,
                        f"repetition_{str(args.training_repetition)}", f"{args.setting}",f"{args.baseline_folder_name}_{args.experiment}")
    with open(f"{basename}_config.pkl", 'rb') as f:
        config = pickle.load(f)
    with open(f"{basename}_wp.pkl", 'rb') as f:
        wp_dict = pickle.load(f)
    val_set = CARLA_Data(root=config.val_data, config=config, rank=0,baseline=args.baseline_folder_name)
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
    data_df = pd.DataFrame.from_dict(wp_dict, orient='index', columns=['image','pred', 'gt', 'loss'])
    data_df['running_diff'] = data_df['pred'].diff()
    data_df['running_diff']=data_df['running_diff'].apply(lambda x: norm(x,ord=args.norm))
    data_df['running_diff_gt'] = data_df['gt'].diff()
    data_df['running_diff_gt']=data_df['running_diff_gt'].apply(lambda x: norm(x,ord=args.norm))
   

    std_value_gt=np.std(data_df["running_diff_gt"])
    std_value_data=np.std(data_df["running_diff"])

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
        
        if config.img_seq_len<7:
            empties=np.concatenate([np.zeros_like(Image.open(data_df.iloc[0]["image"]))]*(7-config.img_seq_len))
            image_sequence=load_image_sequence(config,data_df, data_loader_position)
            image_sequence=np.concatenate([empties, image_sequence], axis=0)
        else:
            image_sequence=load_image_sequence(config,data_df, data_loader_position)
        pred_residual=norm(data_df.iloc[current_index]["pred"]-data_df.iloc[previous_index]["pred"], ord=args.norm)
        gt_residual=norm(data_df.iloc[current_index]["gt"]-data_df.iloc[previous_index]["gt"], ord=args.norm)
        if args.visualize_non_copycat:
            visualize_model(config=config, save_path=os.path.join(os.environ.get("WORK_DIR"),"vis",args.baseline_folder_name), rgb=image_sequence, lidar_bev=torch.Tensor(data["lidar"]),
                                    pred_wp_prev=torch.Tensor(data_df.iloc[previous_index]["pred"][0]),gt_bev_semantic=torch.ByteTensor(data["bev_semantic"]), step=current_index, target_point=torch.Tensor(data["target_point"]), pred_wp=torch.Tensor(data_df.iloc[current_index]["pred"][0]),
                                    gt_wp=torch.Tensor(data_df.iloc[current_index]["gt"][0]),pred_residual=pred_residual, gt_residual=gt_residual,copycat_count=count, frame=data_loader_position)
        if pred_residual<std_value_data*0.3 and gt_residual>std_value_gt:
            #0.15 and 1 for the one curve only
            count+=1
            if args.visualize_non_copycat or args.visualize_copycat:
                visualize_model(config=config, save_path=os.path.join(os.environ.get("WORK_DIR"),"vis",args.baseline_folder_name), rgb=image_sequence, lidar_bev=torch.Tensor(data["lidar"]),
                        pred_wp_prev=torch.Tensor(data_df.iloc[previous_index]["pred"][0]),gt_bev_semantic=torch.ByteTensor(data["bev_semantic"]), step=current_index, target_point=torch.Tensor(data["target_point"]), pred_wp=torch.Tensor(data_df.iloc[current_index]["pred"][0]),
                        gt_wp=torch.Tensor(data_df.iloc[current_index]["gt"][0]),pred_residual=pred_residual, gt_residual=gt_residual,copycat_count=count, detect=True, frame=data_loader_position)
    print(f"count for real copycat for baseline {args.baseline_folder_name}: {count}")
if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
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
        "--experiment",
        default=None,
        required=True,
        help="filename of experiment without .yaml",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="all",
        help="coil requires to be trained on Town01 only, so Town01 are train conditions and Town02 is Test Condition",
    )
    parser.add_argument(
        "--visualize-non-copycat",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--visualize-copycat",
        type=int,
        default=0,
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