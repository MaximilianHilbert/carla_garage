import os
import sys
import time
import traceback
import torch
from tqdm import tqdm
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
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import heapq
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def merge_config_files(args,training=True):
    #merge the old baseline config coil_config and the experiment dependent yaml config into one g_conf object

    merge_with_yaml(os.path.join(os.environ.get("CONFIG_ROOT"), args.baseline_folder_name, args.baseline_name+".yaml"))
    
    # init transfuser config file, necessary for the dataloader
    shared_configuration = GlobalConfig()
    if training:
        shared_configuration.initialize(root_dir=shared_configuration.root_dir, setting=args.setting)
    #translates the necessary old argument names in the yaml file of the baseline to the new transfuser config, generating one shared object configuration
    shared_configuration.number_previous_actions=g_conf.NUMBER_PREVIOUS_ACTIONS
    shared_configuration.epochs=g_conf.EPOCHS
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
    shared_configuration.baseline_folder_name=args.baseline_folder_name
    shared_configuration.baseline_name=args.baseline_name
    shared_configuration.auto_lr=g_conf.AUTO_LR
    shared_configuration.every_epoch=g_conf.EVERY_EPOCH
    shared_configuration.auto_lr_step=g_conf.AUTO_LR_STEP
    shared_configuration.num_repetitions=args.dataset_repetition
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

class Logger():
    def __init__(self, baseline_name, experiment, repetition):
        self.dir_name = os.path.join(os.environ.get("WORK_DIR"),'_logs',baseline_name, experiment, f"repetition_{str(repetition)}")
        self.full_name = os.path.join(self.dir_name, "tensorboard")
    
    def add_scalar(self, name, scalar, step):
        self.writer.add_scalar(name, scalar,step)

    def create_tensorboard_logs(self):
        self.writer = SummaryWriter(log_dir=self.full_name)
    def create_checkpoint_logs(self,):
        os.makedirs(os.path.join(self.dir_name, "checkpoints"), exist_ok=True)
def get_free_training_port():
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    return sock.getsockname()[1]


def main(args):
    
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['LOCAL_RANK'])
    print(f"World-size {world_size}, Rank {rank}")
    dist.init_process_group(backend="nccl",
                                    init_method='env://',
                                    world_size=world_size,
                                    rank=rank)
    if rank==0:
        print("Backend initialized")
    device_id = torch.device(f'cuda:{rank}')
    

    merged_config_object=merge_config_files(args)
    logger=Logger(merged_config_object.baseline_folder_name, merged_config_object.baseline_name, args.training_repetition)
    if rank==0:
        logger.create_tensorboard_logs()
        print(f"Start of Training {args.baseline_folder_name}, {args.baseline_name}, {args.training_repetition}")
    logger.create_checkpoint_logs()
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
        
        set_seed(args.seed)

       
        checkpoint_file = get_latest_saved_checkpoint(merged_config_object,repetition=args.training_repetition)
        if checkpoint_file is not None:
            checkpoint = torch.load(
                                    os.path.join(
                                        os.environ.get("WORK_DIR"), '_logs', merged_config_object.baseline_folder_name, merged_config_object.baseline_name,f"repetition_{str(args.training_repetition)}",
                                        'checkpoints', get_latest_saved_checkpoint(merged_config_object,repetition=args.training_repetition)
                                    )
                                )
            epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            best_loss_epoch = checkpoint['best_loss_epoch']
        else:
            epoch = 0
            best_loss = 10000.0
            best_loss_epoch = 0
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
        dataset=CARLA_Data(root=merged_config_object.train_data, config=merged_config_object, shared_dict=shared_dict, rank=rank)
        if "keyframes" in args.baseline_name:
            #load the correlation weights and reshape them, that the last 3 elements that do not fit into the batch size dimension get dropped, because the dataloader of Carla_Dataset does the same, it should fit
            list_of_files_path=os.path.join(os.environ.get("WORK_DIR"), "_logs", merged_config_object.baseline_folder_name, f"repetition_{str(args.training_repetition)}")
            for file in os.listdir(list_of_files_path):
                full_filename=os.path.join(list_of_files_path, file)
                if f"rep{str(args.training_repetition)}" in file:
                    dataset.set_correlation_weights(path=full_filename)
            action_predict_threshold=get_action_predict_loss_threshold(dataset.get_correlation_weights(),merged_config_object.threshold_ratio)
        print("Loaded dataset")
        sampler=DistributedSampler(dataset)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                              num_workers=args.number_of_workers,
                                              pin_memory=True,
                                              shuffle=False, #because of DDP
                                              drop_last=True,
                                              sampler=sampler
                                              )
        if "arp" in args.baseline_name:
            policy = CoILModel(merged_config_object.model_type, merged_config_object.model_configuration)
            policy.to(device_id)
            policy=DDP(policy, device_ids=[device_id])
            mem_extract = CoILModel(merged_config_object.mem_extract_model_type, merged_config_object.mem_extract_model_configuration)
            mem_extract.to(device_id)
            mem_extract=DDP(mem_extract, device_ids=[device_id])
        else:
            model=CoILModel(merged_config_object.model_type, merged_config_object.model_configuration)
            model.to(device_id)
            model=DDP(model, device_ids=[device_id])
        if merged_config_object.optimizer == 'Adam':
            if "arp" in args.baseline_name:
                policy_optimizer = optim.Adam(policy.parameters(), lr=merged_config_object.learning_rate)
                mem_extract_optimizer = optim.Adam(mem_extract.parameters(), lr=merged_config_object.learning_rate)
                mem_extract_scheduler=MultiStepLR(mem_extract_optimizer, milestones=args.adapt_lr_milestones, gamma=0.1)
                policy_scheduler=MultiStepLR(policy_optimizer, milestones=args.adapt_lr_milestones, gamma=0.1)
            else:
                optimizer= optim.Adam(model.parameters(), lr=merged_config_object.learning_rate)
                scheduler=MultiStepLR(optimizer, milestones=args.adapt_lr_milestones, gamma=0.1)
        elif merged_config_object.optimizer == 'SGD':
            if "arp" in args.baseline_name:
                policy_optimizer = optim.SGD(policy.parameters(), lr=merged_config_object.learning_rate, momentum=0.9)
                mem_extract_optimizer = optim.SGD(mem_extract.parameters(), lr=merged_config_object.learning_rate, momentum=0.9)
                mem_extract_scheduler=MultiStepLR(mem_extract_optimizer, milestones=args.adapt_lr_milestones, gamma=0.1)
                policy_scheduler=MultiStepLR(policy_optimizer, milestones=args.adapt_lr_milestones, gamma=0.1)
            else:
                optimizer = optim.SGD(model.parameters(), lr=merged_config_object.learning_rate, momentum=0.9)
                scheduler=MultiStepLR(optimizer, milestones=args.adapt_lr_milestones, gamma=0.1)
        else:
            raise ValueError

        if checkpoint_file is not None or merged_config_object.preload_model_alias is not None:
            accumulated_time = checkpoint['total_time']
            already_trained_epochs=checkpoint["epoch"]
            if "arp" in args.baseline_name:
                policy.load_state_dict(checkpoint['policy_state_dict'])
                policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
                mem_extract.load_state_dict(checkpoint['mem_extract_state_dict'])
                mem_extract_optimizer.load_state_dict(checkpoint['mem_extract_optimizer'])
            else:
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                accumulated_time = checkpoint['total_time']
            
        else:  # We accumulate iteration time and keep the average speed
            accumulated_time = 0
            already_trained_epochs=0
        print("Before the loss")
        if "keyframes" in args.baseline_name:
            from coil_network.keyframes_loss import Loss
        else:
            from coil_network.loss import Loss
        criterion = Loss(merged_config_object.loss_function)
        for epoch in tqdm(range(1+already_trained_epochs, merged_config_object.epochs+1), disable=rank!=0):
            for iteration, data in enumerate(tqdm(data_loader, disable=rank!=0), start=1):
                # if g_conf.FINISH_ON_VALIDATION_STALE is not None and \
                #         check_loss_validation_stopped(iteration, g_conf.FINISH_ON_VALIDATION_STALE):
                #     break
                capture_time = time.time()
                controls = get_controls_from_data(data,args.batch_size,device_id)
                current_image=torch.reshape(data['rgb'].to(device_id).to(torch.float32)/255., (args.batch_size, -1, merged_config_object.camera_height, merged_config_object.camera_width))
                current_speed =data["speed"].to(device_id).reshape(args.batch_size, 1)
                targets=torch.concat([data["steer"].to(device_id).reshape(args.batch_size,1), data["throttle"].to(device_id).reshape(args.batch_size,1), data["brake"].to(device_id).reshape(args.batch_size,1)], dim=1).reshape(args.batch_size,3)
                if "arp" in args.baseline_name or "bcoh" in args.baseline_name or "keyframes" in args.baseline_name:
                    temporal_images=data['temporal_rgb'].to(device_id)/255.
                    previous_action=data["previous_actions"].to(device_id)
                if "arp" in args.baseline_name:
                    current_speed_zero_speed =torch.zeros_like(current_speed)
                    mem_extract.zero_grad()
                    mem_extract_branches, memory = mem_extract(temporal_images)
                    
                    mem_extract_targets=targets-previous_action
                    loss_function_params_memory = {
                    'branches': mem_extract_branches,
                    'targets': mem_extract_targets,
                    'controls': controls,
                    'inputs': current_speed,
                    'branch_weights': merged_config_object.branch_loss_weight,
                    'variable_weights': merged_config_object.variable_weight
                    }

                    mem_extract_loss, _ = criterion(loss_function_params_memory)
                    mem_extract_loss.backward()
                    mem_extract_optimizer.step()
                    policy.zero_grad()
                    policy_branches = policy(current_image, current_speed_zero_speed, memory)
                    loss_function_params_policy = {
                        'branches': policy_branches,
                        'targets':targets,
                        'controls': controls,
                        'inputs': current_speed,
                        'branch_weights': merged_config_object.branch_loss_weight,
                        'variable_weights': merged_config_object.variable_weight
                    }
                    policy_loss, _ = criterion(loss_function_params_policy)
                    policy_loss.backward()
                    policy_optimizer.step()
                    if is_ready_to_save(epoch, iteration, data_loader, merged_config_object) and rank ==0:
                        state = {
                            'epoch': epoch,
                            'policy_state_dict': policy.state_dict(),
                            'mem_extract_state_dict': mem_extract.state_dict(),
                            'best_loss': best_loss,
                            'total_time': accumulated_time,
                            'policy_optimizer': policy_optimizer.state_dict(),
                            'mem_extract_optimizer': mem_extract_optimizer.state_dict(),
                            'best_loss_epoch': best_loss_epoch
                        }
                        torch.save(
                            state, 
                            os.path.join(
                                os.environ.get("WORK_DIR"), "_logs", merged_config_object.baseline_folder_name, merged_config_object.baseline_name, f"repetition_{str(args.training_repetition)}",
                                    'checkpoints', str(epoch) + '.pth'
                            )
                        )
                    if rank==0:
                        logger.add_scalar('Policy_Loss_Iterations', policy_loss.data, (epoch-1)*len(data_loader)+iteration)
                        logger.add_scalar('Policy_Loss_Epochs', policy_loss.data, (epoch-1))
                        logger.add_scalar('Mem_Extract_Loss_Iterations', mem_extract_loss.data, (epoch-1)*len(data_loader)+iteration)
                        logger.add_scalar('Mem_Extract_Loss_Epochs', mem_extract_loss.data, (epoch-1))
                    if policy_loss.data < best_loss:
                        best_loss = policy_loss.data.tolist()
                        best_loss_epoch = epoch
                    accumulated_time += time.time() - capture_time
                    if iteration%args.printing_step==0 and rank==0:
                        print(f"Epoch: {epoch} // Iteration: {iteration} // Policy_Loss: {policy_loss.data}")
                        print(f"Epoch: {epoch} // Iteration: {iteration} // Mem_Extract_Loss: {mem_extract_loss.data}")
                    policy_scheduler.step()
                    mem_extract_scheduler.step()
                else:
                    model.zero_grad()
                    optimizer.zero_grad()
                    
                #TODO WHY ARE THE PREVIOUS ACTIONS INPUT TO THE BCOH BASELINE??????!!!!#######################################################
                if "bcoh" in args.baseline_name or "keyframes" in args.baseline_name:
                    temporal_and_current_images=torch.cat([temporal_images, current_image], axis=1)
                    if merged_config_object.train_with_actions_as_input:
                        branches = model(temporal_and_current_images,
                                current_speed,
                                previous_action)
                    else:
                        branches = model(temporal_and_current_images,
                                current_speed)
                if "bcso" in args.baseline_name:
                    branches = model(current_image,
                                current_speed)

                if "keyframes" in args.baseline_name:
                    reweight_params = {'importance_sampling_softmax_temper': merged_config_object.softmax_temper,
                                    'importance_sampling_threshold': action_predict_threshold,
                                    'importance_sampling_method': merged_config_object.importance_sample_method,
                                    'importance_sampling_threshold_weight': merged_config_object.threshold_weight,
                                    'action_predict_loss': data["correlation_weight"].squeeze().to(device_id)}
                else:
                    reweight_params={}
                if "arp" not in args.baseline_name:
                    loss_function_params = {
                    'branches': branches,
                    'targets': targets,
                    **reweight_params,
                    'controls': controls,
                    'inputs':current_speed,
                    'branch_weights': merged_config_object.branch_loss_weight,
                    'variable_weights': merged_config_object.variable_weight
                    }
                    if "keyframes" in args.baseline_name:
                        loss, loss_info, _ = criterion(loss_function_params)
                    else:
                        loss, _ = criterion(loss_function_params)
            
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    if is_ready_to_save(epoch, iteration, data_loader, merged_config_object) and rank ==0:
                        state = {
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'best_loss': best_loss,
                            'total_time': accumulated_time,
                            'optimizer': optimizer.state_dict(),
                            'best_loss_epoch': best_loss_epoch
                        }
                
                        torch.save(
                            state, 
                            os.path.join(
                                os.environ.get("WORK_DIR"), "_logs", merged_config_object.baseline_folder_name, merged_config_object.baseline_name, f"repetition_{str(args.training_repetition)}",
                                    'checkpoints', str(epoch) + '.pth'
                            )
                        )
                    
                    if loss.data < best_loss:
                        best_loss = loss.data.tolist()
                        best_loss_epoch = epoch
                    accumulated_time += time.time() - capture_time
                    if rank==0:
                        if iteration%args.printing_step==0:
                            print(f"Epoch: {epoch} // Iteration: {iteration} // Loss:{loss.data}")
                        logger.add_scalar(f'{merged_config_object.baseline_name}_loss', loss.data, (epoch-1)*len(data_loader)+iteration)
                        logger.add_scalar(f'{merged_config_object.baseline_name}_loss_Epochs', loss.data, (epoch-1))
            torch.cuda.empty_cache()
        dist.destroy_process_group()

    except RuntimeError as e:
        traceback.print_exc()

    except:
        traceback.print_exc()

def get_controls_from_data(data,batch_size, device_id):
    one_hot_tensor=data["command"].to(device_id)
    indices = torch.argmax(one_hot_tensor, dim=1)
    controls=indices.reshape(batch_size, 1)
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
    parser.add_argument('--baseline_folder_name', dest="baseline_folder_name", default=None, required=True)
    parser.add_argument('--baseline_name', dest="baseline_name", default=None, required=True)
    parser.add_argument('--number_of_workers', dest="number_of_workers", default=12, type=int, required=True)
    parser.add_argument('--use-disk-cache', dest="use_disk_cache", type=int, default=0)
    parser.add_argument('--batch-size', dest="batch_size", type=int, default=30)
    parser.add_argument('--printing-step', dest="printing_step", type=int, default=10000)
    parser.add_argument('--adapt-lr-milestones', dest="adapt_lr_milestones", nargs="+",type=int, default=[30])
    parser.add_argument('--setting',type=str, default="all", help="coil requires to be trained on Town01 only, so Town01 are train conditions and Town02 is Test Condition")
    parser.add_argument(
        '--dataset-repetition',
        type=int,
        default=1
    )

    arguments = parser.parse_args()
    main(arguments)