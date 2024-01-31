from team_code.data import CARLA_Data
from team_code.config import GlobalConfig
import argparse
import os
import torch
from torch.utils.data import DataLoader
from coil_configuration.coil_config import g_conf, merge_with_yaml
import timeit
def main(args):
    merge_with_yaml(os.path.join('coil_configuration', args.baseline_folder_name, args.baseline_name + '.yaml'))
    shared_configuration = GlobalConfig()
    shared_configuration.initialize(root_dir=shared_configuration.root_dir)
    #only set config_transfuser args to yaml/coiltraine args that matter for dataset generation
    shared_configuration.number_previous_actions=g_conf.NUMBER_PREVIOUS_ACTIONS
    shared_configuration.number_future_actions=g_conf.NUMBER_FUTURE_ACTIONS
    shared_configuration.img_seq_len=g_conf.IMAGE_SEQ_LEN 
    shared_configuration.all_frames_including_blank=g_conf.ALL_FRAMES_INCLUDING_BLANK
    shared_configuration.targets=g_conf.TARGETS
    shared_configuration.batch_size=1
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
    torch.manual_seed(123)
    data=CARLA_Data(root=shared_configuration.train_data,config=shared_configuration)
    data_loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=args.number_of_workers)
    def test(data_Loader):
        le=range(10)
        for idx, data in zip(le, data_loader):
          continue
    timer=timeit.Timer(lambda: test(data_loader))
    execution_time=timer.timeit(100)
    print("final execution time")
    print(execution_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_name", default="arp_vanilla", dest='baseline_name',help="")
    parser.add_argument("--baseline_folder_name", default="ARP", dest='baseline_folder_name',help="")
    parser.add_argument("--number-of-workers", dest='number_of_workers',help="")
    
    args = parser.parse_args()
    
    main(args)