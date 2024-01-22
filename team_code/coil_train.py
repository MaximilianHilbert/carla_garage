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
from coil_network.loss import Loss
from coil_network.optimizer import adjust_learning_rate_auto
from coil_input import Augmenter, select_balancing_strategy
from coil_logger import coil_logger
from coil_utils.checkpoint_schedule import is_ready_to_save, get_latest_saved_checkpoint
from team_code.data import CARLA_Data
from team_code.config import GlobalConfig
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def merge_config_files():
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
    return shared_configuration

def main(args, suppress_output=False):
    merged_config_object=merge_config_files()
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
        coil_logger.add_message('Loading', {'GPU': args.gpu})

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

        # Preload option
        if merged_config_object.preload_model_alias is not None:
            checkpoint = torch.load(
                                os.path.join(
                                    '_logs', merged_config_object.preload_model_batch, merged_config_object.preload_model_alias,
                                    'checkpoints', str(merged_config_object.preload_model_checkpoint) + '.pth'
                                )
                            )

        # Get the latest checkpoint to be loaded
        # returns none if there are no checkpoints saved for this model
        checkpoint_file = get_latest_saved_checkpoint(repetition=args.training_repetition)
        if checkpoint_file is not None:
            checkpoint = torch.load(
                                    os.path.join(
                                        '_logs', args.baseline_folder_name, args.baseline_name,str(args.training_repetition),
                                        'checkpoints', get_latest_saved_checkpoint(repetition=args.training_repetition)
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
       
        print("Loaded dataset")

        data_loader = select_balancing_strategy(dataset, iteration, args.number_of_workers)
        policy = CoILModel(merged_config_object.model_type, merged_config_object.model_configuration)
        policy.cuda()

        mem_extract = CoILModel(merged_config_object.mem_extract_model_type, merged_config_object.mem_extract_model_configuration)
        mem_extract.cuda()

        if merged_config_object.optimizer == 'Adam':
            policy_optimizer = optim.Adam(policy.parameters(), lr=merged_config_object.learning_rate)
            mem_extract_optimizer = optim.Adam(mem_extract.parameters(), lr=merged_config_object.learning_rate)
        elif merged_config_object.optimizer == 'SGD':
            policy_optimizer = optim.SGD(policy.parameters(), lr=merged_config_object.learning_rate, momentum=0.9)
            mem_extract_optimizer = optim.SGD(mem_extract.parameters(), lr=merged_config_object.learning_rate, momentum=0.9)
        else:
            raise ValueError

        if checkpoint_file is not None or merged_config_object.preload_model_alias is not None:
            accumulated_time = checkpoint['total_time']
        
            policy.load_state_dict(checkpoint['policy_state_dict'])
            policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
            policy_loss_window = coil_logger.recover_loss_window('policy_train', iteration)
            
            mem_extract.load_state_dict(checkpoint['mem_extract_state_dict'])
            mem_extract_optimizer.load_state_dict(checkpoint['mem_extract_optimizer'])
            mem_extract_loss_window = coil_logger.recover_loss_window('mem_extract_train', iteration)
        else:  # We accumulate iteration time and keep the average speed
            accumulated_time = 0
            policy_loss_window = []
            mem_extract_loss_window = []

        print("Before the loss")

        criterion = Loss(merged_config_object.loss_function)

        for data in data_loader:
            """
            ####################################
                Main optimization loop
            ####################################
            """

            iteration += 1
            if iteration % 1000 == 0:
                adjust_learning_rate_auto(policy_optimizer, policy_loss_window)
                adjust_learning_rate_auto(mem_extract_optimizer, mem_extract_loss_window)


            capture_time = time.time()
            one_hot_tensor=data["next_command"]
            indices = torch.argmax(one_hot_tensor, dim=1).numpy()
            controls=torch.cuda.FloatTensor(indices).reshape(merged_config_object.batch_size, 1)
            if merged_config_object.blank_frames_type == 'black':
                blank_images_tensor = torch.cat([torch.zeros_like(data['rgb']) for _ in range(merged_config_object.all_frames_including_blank - merged_config_object.img_seq_len)], dim=1).cuda()

            obs_history = torch.cat([blank_images_tensor,data['temporal_rgb'].cuda(), data['rgb'].cuda()], dim=1).to(torch.float32).cuda()
            obs_history=obs_history/255.
            obs_history=obs_history.view(merged_config_object.batch_size, 30, merged_config_object.camera_height, merged_config_object.camera_width)

            current_obs = torch.zeros_like(obs_history).cuda()
            current_obs[:, -3:] = obs_history[:, -3:]
            
                
            current_speed =torch.zeros_like(dataset.extract_inputs(data, merged_config_object)).reshape(merged_config_object.batch_size, 1).to(torch.float32).cuda()
                

            
            mem_extract.zero_grad()
            mem_extract_branches, memory = mem_extract(obs_history)
            previous_action=torch.cat([data["previous_steers"][0], data["previous_throttles"][0], torch.where(data["previous_brakes"][0], torch.tensor(1.0), torch.tensor(0.0))]).cuda()
            #TODO watch out with implementation of previous action tensor
            loss_function_params = {
                'branches': mem_extract_branches,
                #we get only bool values from our expert, so we have to convert them back to floats, so that the baseline can work with it
                'targets': (dataset.extract_targets(data, merged_config_object).cuda() -previous_action).reshape(merged_config_object.batch_size, -1),
                'controls': controls.cuda(),
                'inputs': data["speed"].cuda(),
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
                'targets': dataset.extract_targets(data, merged_config_object).reshape(merged_config_object.batch_size, -1).cuda(),
                'controls': controls.cuda(),
                'inputs': dataset.extract_inputs(data, merged_config_object).cuda(),
                'branch_weights': merged_config_object.branch_loss_weight,
                'variable_weights': merged_config_object.variable_weight
            }
            policy_loss, _ = criterion(loss_function_params)
            policy_loss.backward()
            policy_optimizer.step()
                      
            """
            ####################################
                Saving the model if necessary
            ####################################
            """

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
                        os.environ.get("WORK_DIR"), args.baseline_folder_name, args.baseline_name, str(args.training_repetition),
                         'checkpoints', str(iteration) + '.pth'
                    )
                )

            """
            ################################################
                   Adding tensorboard logs.
                   Making calculations for logging purposes.
                   These logs are monitored by the printer module.
            #################################################
            """
            coil_logger.add_scalar('Policy_Loss', policy_loss.data, iteration)
            coil_logger.add_scalar('Mem_Extract_Loss', mem_extract_loss.data, iteration)
            #coil_logger.add_image('Image', torch.squeeze(data['rgb']), iteration)
            if policy_loss.data < best_loss:
                best_loss = policy_loss.data.tolist()
                best_loss_iter = iteration

            # Log a random position
            position = random.randint(0, len(data) - 1)

            output = policy.extract_branch(torch.stack(policy_branches[0:4]), controls)
            error = torch.abs(output - dataset.extract_targets(data, merged_config_object).reshape(merged_config_object.batch_size, -1).cuda())

            accumulated_time += time.time() - capture_time

            policy_loss_window.append(policy_loss.data.tolist())
            mem_extract_loss_window.append(mem_extract_loss.data.tolist())
            coil_logger.write_on_error_csv('policy_train', policy_loss.data)
            coil_logger.write_on_error_csv('mem_extract_train', mem_extract_loss.data)
            print("Iteration: %d  Policy_Loss: %f" % (iteration, policy_loss.data))
            print("Iteration: %d  Mem_Extract_Loss: %f" % (iteration, mem_extract_loss.data))

        coil_logger.add_message('Finished', {})

    except KeyboardInterrupt:
        coil_logger.add_message('Error', {'Message': 'Killed By User'})

    except RuntimeError as e:
        traceback.print_exc()
        coil_logger.add_message('Error', {'Message': str(e)})

    except:
        traceback.print_exc()
        coil_logger.add_message('Error', {'Message': 'Something Happened'})

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',  dest='seed',required=True, type=int, default=345345)
    parser.add_argument('--training_repetition', dest="training_repetition", type=int, default=0, required=True)
    parser.add_argument('--gpu', dest="gpu", default=0, required=True)
    parser.add_argument('--baseline_folder_name', dest="baseline_folder_name", default=None, required=True)
    parser.add_argument('--baseline_name', dest="baseline_name", default=None, required=True)
    parser.add_argument('--number_of_workers', dest="number_of_workers", default=12, type=int, required=True)
    parser.add_argument('--use-disk-cache', dest="use_disk_cache", type=bool, default=0)
    args = parser.parse_args()
    main(args)