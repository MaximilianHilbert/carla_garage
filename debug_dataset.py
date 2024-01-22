from team_code.data import CARLA_Data
from team_code.config import GlobalConfig
import argparse
import os
import torch
from torch.utils.data import DataLoader
from coil_configuration.coil_config import g_conf, merge_with_yaml
from diskcache import Cache
import timeit
def main(args):
    merge_with_yaml(os.path.join('coil_configuration', args.baseline_folder_name, args.baseline_name + '.yaml'))
    config_tf = GlobalConfig()
    config_tf.initialize(root_dir=config_tf.root_dir)
    #only set config_transfuser args to yaml/coiltraine args that matter for dataset generation
    config_tf.number_previous_actions=g_conf.NUMBER_PREVIOUS_ACTIONS
    config_tf.img_seq_len=g_conf.IMAGE_SEQ_LEN 
    config_tf.all_frames_including_blank=g_conf.ALL_FRAMES_INCLUDING_BLANK
    torch.manual_seed(123)
    for cache_value in [True, False]:
        if bool(cache_value):
        # NOTE: This is specific to our cluster setup where the data is stored on slow storage.
        # During training, we cache the dataset on the fast storage of the local compute nodes.
        # Adapt to your cluster setup as needed. Important initialize the parallel threads from torch run to the
        # same folder (so they can share the cache).
            tmp_folder = "/home/maximilian/SCRATCH/tmp"
            print('Tmp folder for dataset cache: ', tmp_folder)
            tmp_folder = tmp_folder + '/dataset_cache'
            shared_dict = Cache(directory=tmp_folder, size_limit=int(768 * 1024**3))
        else:
            shared_dict = None
            
        data=CARLA_Data(root=config_tf.train_data,config=config_tf, shared_dict=shared_dict)
        data_loader = DataLoader(data, batch_size=1, shuffle=False)
        # sampler_train = torch.utils.data.SequentialSampler(data)
        # dataloader_train = DataLoader(data,
        #                         sampler=sampler_train,
        #                         batch_size=1,

        #                         num_workers=1)
        
        callable_function=lambda: data.__getitem__(0)
        timer=timeit.Timer(callable_function)
        execution_time=timer.timeit(number=100)
        print(cache_value)
        print(execution_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_name", default="arp_vanilla", dest='baseline_name',help="")
    parser.add_argument("--baseline_folder_name", default="ARP", dest='baseline_folder_name',help="")
    args = parser.parse_args()
    
    main(args)