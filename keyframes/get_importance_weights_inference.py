import argparse
import numpy as np
import torch
import os
from action_correlation_model import train_ape_model
from team_code.config import GlobalConfig
from team_code.data import CARLA_Data
from torch.utils.data import Dataset, DataLoader
from coil_configuration.coil_config import merge_with_yaml, g_conf
from action_correlation_model import ActionModel
import re
from torch.utils.data import SequentialSampler
from tqdm import tqdm
from get_importance_weights_training import merge_config_files

def main(args):
    merged_config_object=merge_config_files(args.baseline_folder_name,args.baseline_name)
    checkpoint_path=os.path.join(os.environ.get("WORK_DIR"), "_logs","keyframes", f"repetition_{str(args.repetition)}", "checkpoints")
    checkpoint=os.listdir(checkpoint_path)[0]
    number_previous_actions, number_current_and_future_actions, repetition, neurons=re.findall(r'\d+', checkpoint)
    action_prediction_model = ActionModel(input_dim=int(number_previous_actions)*3, output_dim=int(number_current_and_future_actions)*3, neurons=[int(neurons)])
    checkpoint=torch.load(os.path.join(checkpoint_path, checkpoint))
    action_prediction_model.load_state_dict(checkpoint["state_dict"])
    action_prediction_model=action_prediction_model.cuda()
    full_dataset=CARLA_Data(root=merged_config_object.train_data, config=merged_config_object)
    sampler=SequentialSampler(full_dataset)
    data_loader=DataLoader(full_dataset, sampler=sampler, shuffle=False, batch_size=args.batch_size, num_workers=args.number_of_workers)
    action_prediction_model.eval()
    action_predict_losses=[]
    for data in tqdm(data_loader):
        previous_actions = data["previous_actions"].cuda()
        current_actions=data["current_and_future_actions"][:3].cuda()
        

        predict_curr_action = action_prediction_model(previous_actions)
        test_loss = ((predict_curr_action - current_actions).pow(2) * torch.Tensor(
            [0.5, 0.45, 0.05]).cuda()).sum().cpu().item()
        action_predict_losses.append(test_loss)
    os.makedirs(os.path.join(os.environ.get("WORK_DIR"), "_logs","keyframes"), exist_ok=True)
    np.save(os.path.join(os.environ.get("WORK_DIR"), "_logs","keyframes", f"repetition_{str(args.repetition)}",f"bcoh_weights_prev{number_previous_actions}_curr{number_current_and_future_actions}_rep{repetition}_neurons{neurons}.npy"),
            action_predict_losses)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-folder-name', dest="baseline_folder_name", type=str, default="keyframes", help='name of the folder that gets created for the baseline')
    parser.add_argument('--baseline-name', dest="baseline_name", type=str, default="keyframes_vanilla_weights.yaml", help='name of the experiment/subfoldername that gets created for the baseline')
    parser.add_argument('--training-repetition', dest="repetition", type=int, default=0, help='repetition')
    parser.add_argument('--batch-size', dest="batch_size", type=int, default=1, help='batch')
    parser.add_argument('--number-of-workers', dest="number_of_workers", type=int, default=1, help='workers to load with')
    
    args = parser.parse_args()
    main(args)
