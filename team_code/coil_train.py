import os
import sys
import random
import time
import traceback
import torch
import torch.optim as optim
from diskcache import Cache
from coil_configuration.coil_config import set_type_of_process, merge_with_yaml, g_conf
from coil_network.coil_model import CoILModel
from coil_network.optimizer import adjust_learning_rate_auto
from coil_input import Augmenter, select_balancing_strategy
from coil_logger import coil_logger
from coil_utils.checkpoint_schedule import is_ready_to_save, get_latest_saved_checkpoint
from team_code.data import CARLA_Data
from team_code.config import GlobalConfig
from coil_utils.general import create_log_folder, create_exp_path, erase_logs
from coil_utils.checkpoint_schedule import is_ready_to_save, get_latest_saved_checkpoint, \
                                    check_loss_validation_stopped
import numpy as np
import matplotlib.pyplot as plt
import heapq
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def merge_config_files(baseline, experiment, training=True):
    #merge the old baseline config coil_config and the experiment dependent yaml config into one g_conf object

    merge_with_yaml(os.path.join(os.environ.get("CONFIG_ROOT"), baseline, experiment+".yaml"))
    
    # init transfuser config file, necessary for the dataloader
    shared_configuration = GlobalConfig()
    if training:
        shared_configuration.initialize(root_dir=shared_configuration.root_dir)
    #translates the necessary old argument names in the yaml file of the baseline to the new transfuser config, generating one shared object configuration
    shared_configuration.number_previous_actions=g_conf.NUMBER_PREVIOUS_ACTIONS
    shared_configuration.number_future_actions=g_conf.NUMBER_FUTURE_ACTIONS
    shared_configuration.img_seq_len=g_conf.IMAGE_SEQ_LEN 
    shared_configuration.all_frames_including_blank=g_conf.ALL_FRAMES_INCLUDING_BLANK
    shared_configuration.targets=g_conf.TARGETS
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
    shared_configuration.importance_sample_method=g_conf.IMPORTANCE_SAMPLE_METHOD
    shared_configuration.softmax_temper=g_conf.SOFTMAX_TEMPER
    shared_configuration.threshold_ratio=g_conf.THRESHOLD_RATIO
    shared_configuration.threshold_weight=g_conf.THRESHOLD_WEIGHT
    shared_configuration.speed_input=g_conf.SPEED_INPUT
    shared_configuration.train_with_actions_as_input=g_conf.TRAIN_WITH_ACTIONS_AS_INPUT
    shared_configuration.correlation_weights=g_conf.CORRELATION_WEIGHTS
    shared_configuration.baseline_folder_name=baseline
    shared_configuration.baseline_name=experiment
    shared_configuration.auto_lr=g_conf.AUTO_LR
    shared_configuration.auto_lr_step=g_conf.AUTO_LR_STEP
    return shared_configuration
from coil_network.loss_functional import compute_branches_masks
def get_predictions(controls, branches):
    controls_mask = compute_branches_masks(controls,branches[0].shape[1])
    loss_branches_vec = []
    for i in range(len(branches) -1):
        loss_branches_vec.append(branches[i]*controls_mask[i])
    return loss_branches_vec , branches[-1]

def visualize_model(iteration, current_image, current_speed, action_labels, controls, branches, loss):
    actions, speed=get_predictions(controls, branches)
    actions=[action_tuple.detach().cpu().numpy() for action_tuple in actions]
    speed=speed[0].detach().cpu().numpy()

    from PIL import Image, ImageDraw, ImageFont
    current_image=torch.squeeze(current_image)
    current_image=current_image.cpu().numpy()
    current_image=current_image*255
    # Load the image
    image=np.transpose(current_image, axes=(1,2,0)).astype(np.uint8)
    image = Image.fromarray(image)

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Define the text and its position
    speed_text = f"current_speed_label: {current_speed}" 
    speed_text_pos = (10, 10)
    loss_text = f"current_loss: {loss}" 
    loss_test_pos = (500, 20)

    controls_text = f"current_control: {controls}" 
    controls_test_pos = (500, 10)
    speed_pred_text = f"speed_prediction: {speed}" 
    speed_pred_pos = (10, 20)

    actions_labels_text = f"action_labels: {action_labels}" 
    actions_labels_pos = (10, 40)

    # Choose a font and size
    actions_pred_text=f"action_predictions: {np.array2string(np.array(actions))}" 
    actions_pred_pos = (10, 50)
    # Render the text onto the image
    draw.text(controls_test_pos, controls_text, fill="white")
    draw.text(loss_test_pos, loss_text, fill="white")
    draw.text(speed_text_pos, speed_text, fill="white")
    draw.text(speed_pred_pos, speed_pred_text, fill="white")

    draw.text(actions_labels_pos, actions_labels_text, fill="white")
    draw.text(actions_pred_pos, actions_pred_text, fill="white")

    # Save the image with the text
    image.save(f"/home/maximilian/Master/carla_garage/vis/{iteration}.jpg")

def main(args, suppress_output=False):
    merged_config_object=merge_config_files(args.baseline_folder_name, args.baseline_name.replace(".yaml", ""))
    create_log_folder(f'{os.environ.get("WORK_DIR")}/_logs',merged_config_object.baseline_folder_name)
    create_exp_path(f'{os.environ.get("WORK_DIR")}/_logs',merged_config_object.baseline_folder_name,merged_config_object.baseline_name, repetition=args.training_repetition)
    
    """
        The main training function. This functions loads the latest checkpoint
        for a given, exp_batch (folder) and exp_alias (experiment configuration).
        With this checkpoint it starts from the beginning or continue some training.
    Args:
        gpu: The GPU number
        exp_batch: the folder with the experiments
        exp_alias: the alias, experiment name
        suppress_output: if the output are going to be saved on a file
        number_of_workers: the number of threads used for data loading

    Returns:
        None

    """
    try:
        # We set the visible cuda devices to select the GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        # At this point the log file with the correct naming is created.
        # You merge the yaml file with the global configuration structure.
       
        set_type_of_process('train', merged_config_object,args, args.training_repetition)
        # Set the process into loading status.
        #coil_logger.add_message('Loading', {'GPU': args.gpu})

        set_seed(args.seed)

        # Put the output to a separate file if it is the case
        if suppress_output:
            if not os.path.exists('_output_logs'):
                os.mkdir('_output_logs')
            sys.stdout = open(
                            os.path.join(
                                '_output_logs', args.baseline_name + '_' 
                                + merged_config_object.process_name + '_' + str(os.getpid()) + ".out"
                            ), 
                            "a", buffering=1
                        )
            sys.stderr = open(
                            os.path.join(
                                '_output_logs', args.baseline_name 
                                + '_err_' + merged_config_object.process_name + '_' + str(os.getpid()) + ".out"
                            ),
                            "a", buffering=1
                        )

        if coil_logger.check_finish('train'):
            coil_logger.add_message('Finished', {})
            return

        # Get the latest checkpoint to be loaded
        # returns none if there are no checkpoints saved for this model
        checkpoint_file = get_latest_saved_checkpoint(merged_config_object,repetition=args.training_repetition)
        if checkpoint_file is not None:
            checkpoint = torch.load(
                                    os.path.join(
                                        os.environ.get("WORK_DIR"), '_logs', merged_config_object.baseline_folder_name, merged_config_object.baseline_name,f"repetition_{str(args.training_repetition)}",
                                        'checkpoints', get_latest_saved_checkpoint(merged_config_object,repetition=args.training_repetition)
                                    )
                                )
            iteration = checkpoint['iteration']
            best_loss = checkpoint['best_loss']
            best_loss_iter = checkpoint['best_loss_iter']
        else:
            iteration = 0
            best_loss = 10000.0
            best_loss_iter = 0

       
        # By instantiating the augmenter we get a callable that augment images and transform them
        # into tensors.
        augmenter = Augmenter(merged_config_object.augmentation)
        if bool(args.use_disk_cache):
        # NOTE: This is specific to our cluster setup where the data is stored on slow storage.
        # During training, we cache the dataset on the fast storage of the local compute nodes.
        # Adapt to your cluster setup as needed. Important initialize the parallel threads from torch run to the
        # same folder (so they can share the cache).
            tmp_folder = str(os.environ.get('SCRATCH', '/tmp'))
            print('Tmp folder for dataset cache: ', tmp_folder)
            tmp_folder = tmp_folder + '/dataset_cache'
            shared_dict = Cache(directory=tmp_folder, size_limit=int(768 * 1024**3))
        else:
            shared_dict = None
        #introduce new dataset from the Paper TransFuser++
        dataset=CARLA_Data(root=merged_config_object.train_data, config=merged_config_object, shared_dict=shared_dict)
        if "keyframes" in args.baseline_name:
            #load the correlation weights and reshape them, that the last 3 elements that do not fit into the batch size dimension get dropped, because the dataloader of Carla_Dataset does the same, it should fit
            list_of_files_path=os.path.join(os.environ.get("WORK_DIR"), "_logs", merged_config_object.baseline_folder_name, f"repetition_{str(args.training_repetition)}")
            for file in os.listdir(list_of_files_path):
                full_filename=os.path.join(list_of_files_path, file)
                if f"rep{str(args.training_repetition)}" in file:
                    dataset.set_correlation_weights(path=full_filename)
            action_predict_threshold=get_action_predict_loss_threshold(dataset.get_correlation_weights(),merged_config_object.threshold_ratio)
        print("Loaded dataset")
        data_loader = select_balancing_strategy(args,dataset, iteration)
        if "arp" in args.baseline_name:
            policy = CoILModel(merged_config_object.model_type, merged_config_object.model_configuration)
            policy.cuda()

            mem_extract = CoILModel(merged_config_object.mem_extract_model_type, merged_config_object.mem_extract_model_configuration)
            mem_extract.cuda()
        else:
            model=CoILModel(merged_config_object.model_type, merged_config_object.model_configuration)
            model.cuda()
        if merged_config_object.optimizer == 'Adam':
            if "arp" in args.baseline_name:
                policy_optimizer = optim.Adam(policy.parameters(), lr=merged_config_object.learning_rate)
                mem_extract_optimizer = optim.Adam(mem_extract.parameters(), lr=merged_config_object.learning_rate)
            else:
                optimizer= optim.Adam(model.parameters(), lr=merged_config_object.learning_rate)
        elif merged_config_object.optimizer == 'SGD':
            if "arp" in args.baseline_name:
                policy_optimizer = optim.SGD(policy.parameters(), lr=merged_config_object.learning_rate, momentum=0.9)
                mem_extract_optimizer = optim.SGD(mem_extract.parameters(), lr=merged_config_object.learning_rate, momentum=0.9)
            else:
                optimizer = optim.SGD(model.parameters(), lr=merged_config_object.learning_rate, momentum=0.9)
        else:
            raise ValueError

        if checkpoint_file is not None or merged_config_object.preload_model_alias is not None:
            accumulated_time = checkpoint['total_time']
            if "arp" in args.baseline_name:

                policy.load_state_dict(checkpoint['policy_state_dict'])
                policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
                policy_loss_window = coil_logger.recover_loss_window('policy_train', iteration)
                
                mem_extract.load_state_dict(checkpoint['mem_extract_state_dict'])
                mem_extract_optimizer.load_state_dict(checkpoint['mem_extract_optimizer'])
                mem_extract_loss_window = coil_logger.recover_loss_window('mem_extract_train', iteration)
            else:
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                accumulated_time = checkpoint['total_time']
                loss_window = coil_logger.recover_loss_window('train', iteration)
        else:  # We accumulate iteration time and keep the average speed
            accumulated_time = 0
            if "arp" in args.baseline_name:
                policy_loss_window = []
                mem_extract_loss_window = []
            else:
                loss_window = []

        print("Before the loss")
        if "keyframes" in args.baseline_name:
            from coil_network.keyframes_loss import Loss
        else:
            from coil_network.loss import Loss
        criterion = Loss(merged_config_object.loss_function)
        if merged_config_object.auto_lr:
            from torch.optim.lr_scheduler import StepLR
            scheduler = StepLR(optimizer, step_size=merged_config_object.auto_lr_step, gamma=0.5)
        for data in data_loader:

            #data=next(islice(iter(data_loader), 1))
            """
            ####################################
                Main optimization loop
            ####################################
            """
            # if g_conf.FINISH_ON_VALIDATION_STALE is not None and \
            #         check_loss_validation_stopped(iteration, g_conf.FINISH_ON_VALIDATION_STALE):
            #     break
            capture_time = time.time()
            controls = get_controls_from_data(data)
            iteration += 1
            if "arp" in args.baseline_name:
                if iteration % 1000 == 0:
                    adjust_learning_rate_auto(policy_optimizer, policy_loss_window)
                    adjust_learning_rate_auto(mem_extract_optimizer, mem_extract_loss_window)
                obs_history = data['temporal_rgb'].cuda()
                obs_history=obs_history/255.
                current_obs=data['rgb'].cuda()
                current_obs=current_obs/255.
                obs_history=obs_history.reshape(args.batch_size, -1, merged_config_object.camera_height, merged_config_object.camera_width)
                current_obs=current_obs.reshape(args.batch_size, -1, merged_config_object.camera_height, merged_config_object.camera_width)
                current_speed =torch.zeros_like(dataset.extract_inputs(data, merged_config_object)).reshape(args.batch_size, 1).to(torch.float32).cuda()

                mem_extract.zero_grad()
                mem_extract_branches, memory = mem_extract(obs_history)
                previous_action=data["previous_actions"].to(torch.float32)
                steer=torch.unsqueeze(data["steer"], 1)
                throttle=torch.unsqueeze(data["throttle"], 1)
                brake=torch.unsqueeze(data["brake"] , 1)
                current_targets=torch.concat([steer, throttle, brake], dim=1).to(torch.float32)
                mem_extract_targets=current_targets-previous_action
                loss_function_params = {
                'branches': mem_extract_branches,
                'targets': mem_extract_targets.cuda(),
                'controls': controls,
                'inputs': data["speed"].reshape(args.batch_size, -1).to(torch.float32).cuda(),
                'branch_weights': merged_config_object.branch_loss_weight,
                'variable_weights': merged_config_object.variable_weight
                }

                mem_extract_loss, _ = criterion(loss_function_params)
                mem_extract_loss.backward()
                mem_extract_optimizer.step()
                policy.zero_grad()
                policy_branches = policy(current_obs, current_speed, memory)
                loss_function_params = {
                    'branches': policy_branches,
                    'targets':current_targets.cuda(),
                    'controls': controls,
                    'inputs': data["speed"].reshape(args.batch_size, -1).to(torch.float32).cuda(),
                    'branch_weights': merged_config_object.branch_loss_weight,
                    'variable_weights': merged_config_object.variable_weight
                }
                policy_loss, _ = criterion(loss_function_params)
                policy_loss.backward()
                policy_optimizer.step()
                if is_ready_to_save(iteration):
                    state = {
                        'iteration': iteration,
                        'policy_state_dict': policy.state_dict(),
                        'mem_extract_state_dict': mem_extract.state_dict(),
                        'best_loss': best_loss,
                        'total_time': accumulated_time,
                        'policy_optimizer': policy_optimizer.state_dict(),
                        'mem_extract_optimizer': mem_extract_optimizer.state_dict(),
                        'best_loss_iter': best_loss_iter
                    }
                    torch.save(
                        state, 
                        os.path.join(
                            os.environ.get("WORK_DIR"), "_logs", merged_config_object.baseline_folder_name, merged_config_object.baseline_name, f"repetition_{str(args.training_repetition)}",
                                'checkpoints', str(iteration) + '.pth'
                        )
                    )
                coil_logger.add_scalar('Policy_Loss', policy_loss.data, iteration)
                coil_logger.add_scalar('Mem_Extract_Loss', mem_extract_loss.data, iteration)

                if policy_loss.data < best_loss:
                    best_loss = policy_loss.data.tolist()
                    best_loss_iter = iteration
                accumulated_time += time.time() - capture_time
                policy_loss_window.append(policy_loss.data.tolist())
                mem_extract_loss_window.append(mem_extract_loss.data.tolist())
                coil_logger.write_on_error_csv(os.path.join(
                            os.environ.get("WORK_DIR"), "_logs", merged_config_object.baseline_folder_name, merged_config_object.baseline_name, f"repetition_{str(args.training_repetition)}",'policy_train'), policy_loss.data)
                coil_logger.write_on_error_csv(os.path.join(
                            os.environ.get("WORK_DIR"), "_logs", merged_config_object.baseline_folder_name, merged_config_object.baseline_name, f"repetition_{str(args.training_repetition)}",'mem_extract_train'), mem_extract_loss.data)
                #visualize_model(iteration=iteration, action_labels=current_targets, branches=policy_branches, controls=controls,current_image=current_obs,current_speed=data["speed"].to(torch.float32), loss=policy_loss)
                if iteration%100==0:
                    print("Iteration: %d  Policy_Loss: %f" % (iteration, policy_loss.data))
                    print("Iteration: %d  Mem_Extract_Loss: %f" % (iteration, mem_extract_loss.data))

            else:
                if not merged_config_object.auto_lr:
                    if iteration % 2500 == 0:
                        adjust_learning_rate_auto(optimizer,loss_window)
                model.zero_grad()
                optimizer.zero_grad()
                single_frame_input=torch.squeeze(data['rgb'].to(torch.float32).cuda())
                single_frame_input=single_frame_input/255.
                single_frame_input=single_frame_input.to(torch.float32).reshape(args.batch_size, -1, merged_config_object.camera_height ,merged_config_object.camera_width).cuda()
                
                
                current_speed =data["speed"].reshape(args.batch_size, 1).to(torch.float32).cuda()
                #TODO WHY ARE THE PREVIOUS ACTIONS INPUT TO THE BCOH BASELINE??????!!!!#######################################################
                if "bcso" in args.baseline_name:
                    if merged_config_object.train_with_actions_as_input:
                        branches = model(single_frame_input,
                                current_speed,
                                data['previous_actions'].reshape(args.batch_size, -1).to(torch.float32).cuda())
                    else:
                        branches = model(single_frame_input,
                                    current_speed)
                else:
                    multi_frame_input=torch.cat([data['temporal_rgb'].cuda(), data['rgb'].cuda()], dim=1)/255.
                    multi_frame_input=multi_frame_input.reshape(args.batch_size, -1, merged_config_object.camera_height ,merged_config_object.camera_width)
                    if merged_config_object.train_with_actions_as_input:
                        branches = model(multi_frame_input,
                                current_speed,
                                data['previous_actions'].reshape(args.batch_size, -1).to(torch.float32).cuda())
                    else:
                        branches = model(multi_frame_input,
                                    current_speed)
                    ########################################################introduce importance weight adding to the temporal images/lidars and the current one################
                if "keyframes" in args.baseline_name:
                    reweight_params = {'importance_sampling_softmax_temper': merged_config_object.softmax_temper,
                                    'importance_sampling_threshold': action_predict_threshold,
                                    'importance_sampling_method': merged_config_object.importance_sample_method,
                                    'importance_sampling_threshold_weight': merged_config_object.threshold_weight,
                                    'action_predict_loss': data["correlation_weight"].squeeze().cuda()}
                else:
                    reweight_params={}
                steer=torch.unsqueeze(data["steer"], 1)
                throttle=torch.unsqueeze(data["throttle"], 1)
                brake=torch.unsqueeze(data["brake"] , 1)
                targets=torch.concat([steer, throttle, brake], dim=1).to(torch.float32)
                loss_function_params = {
                'branches': branches,
                'targets': targets.cuda(),
                **reweight_params,
                'controls': controls,
                'inputs': data["speed"].to(torch.float32).cuda(),
                'branch_weights': merged_config_object.branch_loss_weight,
                'variable_weights': merged_config_object.variable_weight
                }
                if "keyframes" in args.baseline_name:
                    loss, loss_info, _ = criterion(loss_function_params)
                else:
                    loss, _ = criterion(loss_function_params)
                
                loss.backward()
                optimizer.step()
                if is_ready_to_save(iteration):

                    state = {
                        'iteration': iteration,
                        'state_dict': model.state_dict(),
                        'best_loss': best_loss,
                        'total_time': accumulated_time,
                        'optimizer': optimizer.state_dict(),
                        'best_loss_iter': best_loss_iter
                    }
            
                    torch.save(
                        state, 
                        os.path.join(
                            os.environ.get("WORK_DIR"), "_logs", merged_config_object.baseline_folder_name, merged_config_object.baseline_name, f"repetition_{str(args.training_repetition)}",
                                'checkpoints', str(iteration) + '.pth'
                        )
                    )

                coil_logger.add_scalar('Loss', loss.data, iteration)
                if "keyframes" in args.baseline_name:
                    for loss_name, loss_value in loss_info.items():
                        if loss_value.shape[0] > 0:
                            average_loss_value = (torch.sum(loss_value) / loss_value.shape[0]).data.item()
                        else:
                            average_loss_value = 0
                        coil_logger.add_scalar(loss_name, average_loss_value, iteration)

                if loss.data < best_loss:
                    best_loss = loss.data.tolist()
                    best_loss_iter = iteration
                accumulated_time += time.time() - capture_time
                loss_window.append(loss.data.tolist())
                #coil_logger.write_on_error_csv('train', loss.data)
                if merged_config_object.auto_lr:
                    scheduler.step()
                
                #visualize_model(iteration=iteration, action_labels=targets, branches=branches, controls=controls,current_image=single_frame_input,current_speed=data["speed"].to(torch.float32), loss=loss)
                print(optimizer.param_groups[0]['lr'])
                print("Iteration: %d  Loss: %f" % (iteration, loss.data))
            torch.cuda.empty_cache()
    
        
        #coil_logger.add_message('Finished', {})

    except RuntimeError as e:
        traceback.print_exc()
        #coil_logger.add_message('Error', {'Message': str(e)})

    except:
        traceback.print_exc()
        #coil_logger.add_message('Error', {'Message': 'Something Happened'})

def get_controls_from_data(data):
    one_hot_tensor=data["command"]
    indices = torch.argmax(one_hot_tensor, dim=1).numpy()
    controls=indices.reshape(args.batch_size, 1)
    return controls
def get_action_predict_loss_threshold(correlation_weights, ratio):
    _action_predict_loss_threshold = {}
    if ratio in _action_predict_loss_threshold:
        return _action_predict_loss_threshold[ratio]
    else:
        action_predict_losses = correlation_weights
        threshold = heapq.nlargest(int(len(action_predict_losses) * ratio), action_predict_losses)[-1]
        _action_predict_loss_threshold[ratio] = threshold
        return threshold
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',  dest='seed',required=True, type=int, default=345345)
    parser.add_argument('--training_repetition', dest="training_repetition", type=int, default=0, required=True)
    parser.add_argument('--gpu', dest="gpu", default=0, required=True)
    parser.add_argument('--baseline_folder_name', dest="baseline_folder_name", default=None, required=True)
    parser.add_argument('--baseline_name', dest="baseline_name", default=None, required=True)
    parser.add_argument('--number_of_workers', dest="number_of_workers", default=12, type=int, required=True)
    parser.add_argument('--use-disk-cache', dest="use_disk_cache", type=int, default=0)
    parser.add_argument('--batch-size', dest="batch_size", type=int, default=10)
    args = parser.parse_args()
    main(args)