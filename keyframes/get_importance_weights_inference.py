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
from get_importance_weights_training import merge_config_files

#TODO implement batching if performance is too slow
    #     merged_config_object=merge_config_files(args)
    
    # layer_num_str = [str(n) for n in args.neurons]
    # checkpoint_path="action_correlation/checkpoints"
    # if os.path.exists(checkpoint_path):
    #     checkpoint_file=os.listdir(checkpoint_path)[0]
    #     action_prediction_model = ActionModel(input_dim=merged_config_object.number_previous_actions*3, output_dim=(merged_config_object.number_future_actions+1)*3, neurons=args.neurons)
    #     checkpoint=torch.load(os.path.join(checkpoint_path, checkpoint_file))
    #     action_prediction_model.load_state_dict(checkpoint["state_dict"])
    #     action_prediction_model=action_prediction_model.cuda()
    #     full_dataset=CARLA_Data(root=merged_config_object.train_data, config=merged_config_object)
    #     testloader = DataLoader(dataset=full_dataset, batch_size=merged_config_object.batch_size, num_workers=args.number_of_workers, shuffle=False)
        

    # else:
    #     action_prediction_model, dataset= train_ape_model(args, merged_config_object,if_save=True)
    
    # action_prediction_model.eval()
    # #
    # #img_path_list, measurements_list = np.load(args.data, allow_pickle=True)
    # action_predict_losses={}
    # for batch_idx, batch in enumerate(testloader):
    #     print(f"inference on {batch_idx} of {len(testloader)}")
    #     previous_actions = batch["previous_actions"]
    #     current_and_future_actions=batch["current_and_future_actions"]
        
    #     previous_actions = previous_actions.cuda()
    #     current_actions = current_and_future_actions[:,:3].cuda()

    #     predict_curr_action = action_prediction_model(previous_actions)
    #     test_losses = ((predict_curr_action - current_actions).pow(2) * torch.stack([torch.Tensor(
    #         [0.5, 0.45, 0.05]) for _ in range(merged_config_object.batch_size)]).cuda()).sum(axis=1).cpu()
    #     indices=range( (batch_idx)*merged_config_object.batch_size,((batch_idx+1)*merged_config_object.batch_size))
    #     for frame_index, loss in zip(indices, test_losses):
    #         action_predict_losses[frame_index]=loss

    # np.save(('_prev{}_curr{}_layer{}.npy'.format(merged_config_object.number_previous_actions, merged_config_object.number_future_actions+1, '-'.join(layer_num_str))),
    #         action_predict_losses)
def main(args):
    merged_config_object=merge_config_files(args)
    checkpoint_path=os.path.join(os.environ.get("WORK_DIR"), "_logs","keyframes", "checkpoints")
    # for checkpoint in os.listdir(checkpoint_path):
    #     number_previous_actions, number_current_and_future_actions, repetition, neurons=re.findall(r'\d+', checkpoint)
    #     action_prediction_model = ActionModel(input_dim=int(number_previous_actions)*3, output_dim=int(number_current_and_future_actions)*3, neurons=[int(neurons)])
    #     checkpoint=torch.load(os.path.join(checkpoint_path, checkpoint))
    #     action_prediction_model.load_state_dict(checkpoint["state_dict"])
    #     action_prediction_model=action_prediction_model.cuda()
    #     full_dataset=CARLA_Data(root=merged_config_object.train_data, config=merged_config_object)

    #     action_prediction_model.eval()
    #     action_predict_losses=[]
    #     for index in range(len(full_dataset)):
    #         if index%100==0:
    #             print(f"inference on {index} of {len(full_dataset)}")
    #         data=full_dataset.__getitem__(index)
    #         previous_actions = data["previous_actions"]
    #         current_and_future_actions=data["current_and_future_actions"]
            
    #         previous_actions = torch.FloatTensor(previous_actions).cuda().unsqueeze(0)
    #         current_actions = torch.FloatTensor(current_and_future_actions[:3]).cuda().unsqueeze(0)

    #         predict_curr_action = action_prediction_model(previous_actions)
    #         test_loss = ((predict_curr_action - current_actions).pow(2) * torch.Tensor(
    #             [0.5, 0.45, 0.05]).cuda()).sum().cpu().item()
    #         action_predict_losses.append(test_loss)
    #     os.makedirs(os.path.join(os.environ.get("WORK_DIR"), "_logs","keyframes"), exist_ok=True)
    #     np.save(os.path.join(os.environ.get("WORK_DIR"), "_logs","keyframes", f"_prev{number_previous_actions}_curr{number_current_and_future_actions}_rep{repetition}_neurons{neurons}.npy"),
    #             action_predict_losses)
    
    #     merged_config_object=merge_config_files(args)
    
    layer_num_str = [str(n) for n in args.neurons]
    checkpoint_path="action_correlation/checkpoints"
    if os.path.exists(checkpoint_path):
        checkpoint_file=os.listdir(checkpoint_path)[0]
        action_prediction_model = ActionModel(input_dim=merged_config_object.number_previous_actions*3, output_dim=(merged_config_object.number_future_actions+1)*3, neurons=args.neurons)
        checkpoint=torch.load(os.path.join(checkpoint_path, checkpoint_file))
        action_prediction_model.load_state_dict(checkpoint["state_dict"])
        action_prediction_model=action_prediction_model.cuda()
        full_dataset=CARLA_Data(root=merged_config_object.train_data, config=merged_config_object)
        testloader = DataLoader(dataset=full_dataset, batch_size=merged_config_object.batch_size, num_workers=args.number_of_workers, shuffle=False)
        

    else:
        action_prediction_model, dataset= train_ape_model(args, merged_config_object,if_save=True)
    
    action_prediction_model.eval()
    #
    #img_path_list, measurements_list = np.load(args.data, allow_pickle=True)
    action_predict_losses={}
    for batch_idx, batch in enumerate(testloader):
        print(f"inference on {batch_idx} of {len(testloader)}")
        previous_actions = batch["previous_actions"]
        current_and_future_actions=batch["current_and_future_actions"]
        
        previous_actions = previous_actions.cuda()
        current_actions = current_and_future_actions[:,:3].cuda()

        predict_curr_action = action_prediction_model(previous_actions)
        test_losses = ((predict_curr_action - current_actions).pow(2) * torch.stack([torch.Tensor(
            [0.5, 0.45, 0.05]) for _ in range(merged_config_object.batch_size)]).cuda()).sum(axis=1).cpu()
        indices=range( (batch_idx)*merged_config_object.batch_size,((batch_idx+1)*merged_config_object.batch_size))
        for frame_index, loss in zip(indices, test_losses):
            action_predict_losses[frame_index]=loss

    np.save(('_prev{}_curr{}_layer{}.npy'.format(merged_config_object.number_previous_actions, merged_config_object.number_future_actions+1, '-'.join(layer_num_str))),
            action_predict_losses)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-folder-name', dest="baseline_folder_name", type=str, default="keyframes", help='name of the folder that gets created for the baseline')
    parser.add_argument('--baseline-name', dest="baseline_name", type=str, default="keyframes_vanilla_weights", help='name of the experiment/subfoldername that gets created for the baseline')
    
    
    args = parser.parse_args()
    main(args)
