import os
import sys
import random
import time
import traceback
import torch
import torch.optim as optim
import numpy as np
from coil_config.coil_config import g_conf, set_type_of_process, merge_with_yaml
from coil_network import CoILModel, Loss, adjust_learning_rate_auto
from coil_input import CoILDataset, Augmenter, select_balancing_strategy
from coil_logger import coil_logger
from coil_utils.checkpoint_schedule import is_ready_to_save, get_latest_saved_checkpoint, check_loss_validation_stopped
from team_code.data import CARLA_Data

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main(args,seed=235345,training_repetition=0, merged_config_object=None, suppress_output=False):
    global g_conf
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
        g_conf.immutable(False)
        set_type_of_process('train', args, training_repetition)
        # Set the process into loading status.
        coil_logger.add_message('Loading', {'GPU': args.gpu})

        set_seed(seed)

        # Put the output to a separate file if it is the case
        if suppress_output:
            if not os.path.exists('_output_logs'):
                os.mkdir('_output_logs')
            sys.stdout = open(
                            os.path.join(
                                '_output_logs', args.baseline_name + '_' 
                                + g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out"
                            ), 
                            "a", buffering=1
                        )
            sys.stderr = open(
                            os.path.join(
                                '_output_logs', args.baseline_name 
                                + '_err_' + g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out"
                            ),
                            "a", buffering=1
                        )

        if coil_logger.check_finish('train'):
            coil_logger.add_message('Finished', {})
            return

        # Preload option
        if g_conf.PRELOAD_MODEL_ALIAS is not None:
            checkpoint = torch.load(
                                os.path.join(
                                    '_logs', g_conf.PRELOAD_MODEL_BATCH, g_conf.PRELOAD_MODEL_ALIAS,
                                    'checkpoints', str(g_conf.PRELOAD_MODEL_CHECKPOINT) + '.pth'
                                )
                            )

        # Get the latest checkpoint to be loaded
        # returns none if there are no checkpoints saved for this model
        checkpoint_file = get_latest_saved_checkpoint(repetition=training_repetition)
        if checkpoint_file is not None:
            checkpoint = torch.load(
                                    os.path.join(
                                        '_logs', args.baseline_folder_name, args.baseline_name,str(training_repetition),
                                        'checkpoints', get_latest_saved_checkpoint(repetition=training_repetition)
                                    )
                                )
            iteration = checkpoint['iteration']
            best_loss = checkpoint['best_loss']
            best_loss_iter = checkpoint['best_loss_iter']
        else:
            iteration = 0
            best_loss = 10000.0
            best_loss_iter = 0

        # Define the dataset. This structure is has the __get_item__ redefined in a way
        # that you can access the positions from the root directory as a in a vector.
        full_dataset = os.path.join(args.dataset_root)

        # By instantiating the augmenter we get a callable that augment images and transform them
        # into tensors.
        augmenter = Augmenter(g_conf.AUGMENTATION)

        # Instantiate the class used to read a dataset. The coil dataset generator
        # can be found

        dataset=CARLA_Data(root=merged_config_object.train_data, config=merged_config_object)
        # dataset = CoILDataset(
        #                 full_dataset,
        #                 transform = augmenter,
        #                 preload_name = str(g_conf.NUMBER_OF_HOURS) + 'hours_' + g_conf.TRAIN_DATASET_NAME
        #             )
        
        print("Loaded dataset")

        data_loader = select_balancing_strategy(dataset, iteration, args.number_of_workers)
        policy = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
        policy.cuda()

        mem_extract = CoILModel(g_conf.MEM_EXTRACT_MODEL_TYPE, g_conf.MEM_EXTRACT_MODEL_CONFIGURATION)
        mem_extract.cuda()

        if g_conf.OPTIMIZER == 'Adam':
            policy_optimizer = optim.Adam(policy.parameters(), lr=g_conf.LEARNING_RATE)
            mem_extract_optimizer = optim.Adam(mem_extract.parameters(), lr=g_conf.LEARNING_RATE)
        elif g_conf.OPTIMIZER == 'SGD':
            policy_optimizer = optim.SGD(policy.parameters(), lr=g_conf.LEARNING_RATE, momentum=0.9)
            mem_extract_optimizer = optim.SGD(mem_extract.parameters(), lr=g_conf.LEARNING_RATE, momentum=0.9)
        else:
            raise ValueError

        if checkpoint_file is not None or g_conf.PRELOAD_MODEL_ALIAS is not None:
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

        criterion = Loss(g_conf.LOSS_FUNCTION)

        for data in data_loader:
            #old values for commands and what they mean
            # REACH_GOAL = 0.0
            # GO_STRAIGHT = 5.0
            # TURN_RIGHT = 4.0
            # TURN_LEFT = 3.0
            # LANE_FOLLOW = 2.0
            #new_command cala 0.9: old_command carla 0.8
            #TODO what does 5 mean? Guessed it means finished route
            command_translation_dict={
                5: 0.0,
                1: 3.0,
                2:4.0,
                3:5.0,
                4:2.0
            }

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
            controls=torch.cuda.FloatTensor([command_translation_dict[index] for index in indices]).reshape(g_conf.BATCH_SIZE, 1)
            if g_conf.BLANK_FRAMES_TYPE == 'black':
                blank_images_tensor = torch.cat([torch.zeros_like(data['rgb']) for _ in range(g_conf.ALL_FRAMES_INCLUDING_BLANK - g_conf.IMAGE_SEQ_LEN)], dim=1).cuda()

            obs_history = torch.cat([blank_images_tensor,data['temporal_rgb'].cuda(), data['rgb'].cuda()], dim=1).to(torch.float32).cuda()
            obs_history=obs_history/255.
            obs_history=obs_history.view(g_conf.BATCH_SIZE, 30, merged_config_object.camera_height, merged_config_object.camera_width)

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
                'branch_weights': g_conf.BRANCH_LOSS_WEIGHT,
                'variable_weights': g_conf.VARIABLE_WEIGHT
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
                'branch_weights': g_conf.BRANCH_LOSS_WEIGHT,
                'variable_weights': g_conf.VARIABLE_WEIGHT
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
                        os.environ.get("WORK_DIR"), args.baseline_folder_name, args.baseline_name, str(training_repetition),
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

            # coil_logger.add_message(
            #     'Iterating',
            #     {
            #         'Iteration': iteration,
            #         'Loss': policy_loss.data.tolist(),
            #         'Images/s': (iteration * g_conf.BATCH_SIZE) / accumulated_time,
            #         'BestLoss': best_loss, 
            #         'BestLossIteration': best_loss_iter,
            #         'Output': output[position].data.tolist(),
            #         'GroundTruth': dataset.extract_targets(data, merged_config_object)[position].data.tolist(),
            #         'Error': error[position].data.tolist(),
            #         'Inputs': dataset.extract_inputs(data,merged_config_object)[position].data.tolist()
            #     },
            #     iteration
            # )
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
    main()