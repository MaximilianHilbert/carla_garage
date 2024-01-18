from team_code.data import CARLA_Data
from team_code.config import GlobalConfig
import argparse
import os
from coil_config.coil_config import g_conf, merge_with_yaml

def main(args):
    merge_with_yaml(os.path.join('coil_config', args.baseline_folder_name, args.baseline_name + '.yaml'))
    config_tf = GlobalConfig()
    config_tf.initialize(root_dir=config_tf.root_dir)
    #only set config_transfuser args to yaml/coiltraine args that matter for dataset generation
    config_tf.number_previous_actions=g_conf.NUMBER_PREVIOUS_ACTIONS
    config_tf.img_seq_len=g_conf.NUMBER_FRAMES_FUSION 
    config_tf.all_frames_including_blank=g_conf.ALL_FRAMES_INCLUDING_BLANK

    dataset=CARLA_Data(root=config_tf.train_data, config=config_tf)
    datapoint=dataset.__getitem__(0)
    print("d")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--baseline-config', dest="baseline_config", default='experiments/config_nocrash.yaml')
    parser.add_argument("--baseline-name", default="arp_vanilla", dest='baseline_name',help="")
    parser.add_argument("--baseline-folder-name", default="ARP", dest='baseline_folder_name',help="")
    args = parser.parse_args()
 
    main(args)