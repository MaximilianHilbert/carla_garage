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
def merge_config_files(args):
    #merge the old baseline config coil_config and the experiment dependent yaml config into one g_conf object

    merge_with_yaml(os.path.join(os.environ.get("CONFIG_ROOT"), args.baseline_folder_name, args.baseline_name + '.yaml'))
    
    # init transfuser config file, necessary for the dataloader
    shared_configuration = GlobalConfig()
    shared_configuration.initialize(root_dir=shared_configuration.root_dir)
    #translates the necessary old argument names in the yaml file of the baseline to the new transfuser config, generating one shared object configuration
    shared_configuration.number_previous_actions=g_conf.NUMBER_PREVIOUS_ACTIONS
    shared_configuration.img_seq_len=g_conf.IMAGE_SEQ_LEN 
    shared_configuration.all_frames_including_blank=g_conf.ALL_FRAMES_INCLUDING_BLANK
    shared_configuration.targets=g_conf.TARGETS
    shared_configuration.batch_size=g_conf.BATCH_SIZE
    shared_configuration.inputs=g_conf.INPUTS
    shared_configuration.optimizer=g_conf.OPTIMIZER
    shared_configuration.process_name=g_conf.PROCESS_NAME
    shared_configuration.preload_model_batch=g_conf.PRELOAD_MODEL_BATCH
    shared_configuration.preload_model_alias=g_conf.PRELOAD_MODEL_ALIAS
    shared_configuration.preload_model_checkpoint=g_conf.PRELOAD_MODEL_CHECKPOINT
    shared_configuration.augmentation=g_conf.AUGMENTATION
    shared_configuration.model_configuration=g_conf.MODEL_CONFIGURATION
    shared_configuration.model_type=g_conf.MODEL_TYPE
    shared_configuration.mem_extract_model_type=g_conf.MEM_EXTRACT_MODEL_TYPE
    shared_configuration.mem_extract_model_configuration=g_conf.MEM_EXTRACT_MODEL_CONFIGURATION
    shared_configuration.learning_rate=g_conf.LEARNING_RATE
    shared_configuration.loss_function=g_conf.LOSS_FUNCTION
    shared_configuration.blank_frames_type=g_conf.BLANK_FRAMES_TYPE
    shared_configuration.all_frames_including_blank=g_conf.ALL_FRAMES_INCLUDING_BLANK
    shared_configuration.image_seq_len=g_conf.IMAGE_SEQ_LEN
    shared_configuration.branch_loss_weight=g_conf.BRANCH_LOSS_WEIGHT
    shared_configuration.variable_weight=g_conf.VARIABLE_WEIGHT
    shared_configuration.experiment_generated_name=g_conf.EXPERIMENT_GENERATED_NAME
    shared_configuration.experiment_name=g_conf.EXPERIMENT_NAME
    shared_configuration.experiment_batch_name=g_conf.EXPERIMENT_BATCH_NAME
    shared_configuration.log_scalar_writing_frequency=g_conf.LOG_SCALAR_WRITING_FREQUENCY
    shared_configuration.log_image_writing_frequency=g_conf.LOG_IMAGE_WRITING_FREQUENCY
    shared_configuration.number_future_actions=g_conf.NUMBER_FUTURE_ACTIONS
    shared_configuration.no_speed_input=g_conf.NO_SPEED_INPUT
    shared_configuration.action_correlation_model_type=g_conf.ACTION_CORRELATION_MODEL_TYPE
    shared_configuration.lidar_seq_len=g_conf.LIDAR_SEQ_LEN
    shared_configuration.epochs=g_conf.EPOCHS
    shared_configuration.use_color_aug=g_conf.USE_COLOR_AUG
    shared_configuration.augment=g_conf.AUGMENT
    shared_configuration.correlation_weights=g_conf.CORRELATION_WEIGHTS
    return shared_configuration

def get_prev_actions(index, img_path_list, prev_action_num, measurements_list):
    previous_actions_list = []
    previous_action_index = index
    previous_action_index_buffer = index
    while len(previous_actions_list) < prev_action_num * 3:
        previous_action_index -= 3
        if (previous_action_index < 0) or (
                img_path_list[previous_action_index].split('/')[0] != img_path_list[index].split('/')[0]):
            previous_actions_list.append(measurements_list[previous_action_index_buffer]['brake'])
            previous_actions_list.append(measurements_list[previous_action_index_buffer]['throttle'])
            previous_actions_list.append(measurements_list[previous_action_index_buffer]['steer'])
        else:
            previous_actions_list.append(measurements_list[previous_action_index]['brake'])
            previous_actions_list.append(measurements_list[previous_action_index]['throttle'])
            previous_actions_list.append(measurements_list[previous_action_index]['steer'])
            previous_action_index_buffer = previous_action_index
    previous_actions_list.reverse()
    previous_action_stack = np.array(previous_actions_list).astype(np.float)
    return previous_action_stack


def get_target_actions(index, img_path_list, target_action_num, measurements_list):
    target_actions_list = []
    target_actions_index = index
    target_actions_index_buff = index
    while len(target_actions_list) < target_action_num * 3:
        if (target_actions_index >= len(img_path_list)) or (img_path_list[target_actions_index].split('/')[0] != img_path_list[index].split('/')[0]):
            target_actions_list.append(measurements_list[target_actions_index_buff]['steer'])
            target_actions_list.append(measurements_list[target_actions_index_buff]['throttle'])
            target_actions_list.append(measurements_list[target_actions_index_buff]['brake'])
        else:
            target_actions_list.append(measurements_list[target_actions_index]['steer'])
            target_actions_list.append(measurements_list[target_actions_index]['throttle'])
            target_actions_list.append(measurements_list[target_actions_index]['brake'])
            target_actions_index_buff = target_actions_index
        target_actions_index += 3
    return np.array(target_actions_list).astype(np.float)


def main(args):
    merged_config_object=merge_config_files(args)
    
    layer_num_str = [str(n) for n in args.neurons]
    checkpoint_path=os.path.join(os.environ.get("WORK_DIR"), "action_correlation", "checkpoints")
    if os.path.exists(checkpoint_path):
        checkpoint_file=os.listdir(checkpoint_path)[0]
        action_prediction_model = ActionModel(input_dim=merged_config_object.number_previous_actions*3, output_dim=(merged_config_object.number_future_actions+1)*3, neurons=args.neurons)
        checkpoint=torch.load(os.path.join(checkpoint_path, checkpoint_file))
        action_prediction_model.load_state_dict(checkpoint["state_dict"])
        action_prediction_model=action_prediction_model.cuda()
        full_dataset=CARLA_Data(root=merged_config_object.train_data, config=merged_config_object)

    else:
        action_prediction_model, full_dataset= train_ape_model(args, merged_config_object,if_save=True)
    
    action_prediction_model.eval()
    #
    #img_path_list, measurements_list = np.load(args.data, allow_pickle=True)
    action_predict_losses=[]
    for index in range(len(full_dataset)):
        print(f"inference on {index} of {len(full_dataset)}")
        data=full_dataset.__getitem__(index)
        previous_actions = data["previous_actions"]
        current_and_future_actions=data["current_and_future_actions"]
        
        previous_actions = torch.FloatTensor(previous_actions).cuda().unsqueeze(0)
        current_actions = torch.FloatTensor(current_and_future_actions[:3]).cuda().unsqueeze(0)

        predict_curr_action = action_prediction_model(previous_actions)
        test_loss = ((predict_curr_action - current_actions).pow(2) * torch.Tensor(
            [0.5, 0.45, 0.05]).cuda()).sum().cpu().item()
        action_predict_losses.append(test_loss)
    os.makedirs(os.path.join(os.environ.get("WORK_DIR"), "action_correlation"), exist_ok=True)
    np.save(os.path.join(os.environ.get("WORK_DIR"), "action_correlation", "_prev{}_curr{}_layer{}.npy".format(merged_config_object.number_previous_actions, merged_config_object.number_future_actions+1, '-'.join(layer_num_str))),
            action_predict_losses)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neurons', type=int, nargs='+', default=[300], help='the dimensions of the FC layers')
    parser.add_argument('--baseline-folder-name', dest="baseline_folder_name", type=str, default="keyframes", help='name of the folder that gets created for the baseline')
    parser.add_argument('--baseline-name', dest="baseline_name", type=str, default="keyframes_vanilla_weights", help='name of the experiment/subfoldername that gets created for the baseline')
    parser.add_argument('--use-disk-cache', dest="use_disk_cache", type=int, default=0, help='use caching on /scratch')
    parser.add_argument('--number-of-workers', dest="number_of_workers", type=int, default=12, help='dataloader workers')
    
    args = parser.parse_args()
    main(args)

