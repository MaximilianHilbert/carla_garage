import os
from team_code.coil_train import main as training_loop
from coil_utils.general import create_log_folder, create_exp_path, erase_logs,\
                          erase_wrong_plotting_summaries, erase_validations
from coil_config.coil_config import g_conf, merge_with_yaml
def main(args):
    port = args.port
    tm_port = port + 2
    g_conf.VARIABLE_WEIGHT = {}
    merge_with_yaml(os.path.join('coil_config', args.baseline_folder_name, args.baseline_name + '.yaml'))
    if g_conf.MAGICAL_SEEDS is None:
        raise "Set MAGICAL SEEDS in config files as a list of magical seeds being used for training the baselines"
    else:
        seeds=g_conf.MAGICAL_SEEDS
        for training_repetition, seed in enumerate(seeds):
            create_log_folder(args.baseline_folder_name)
            erase_logs(args.baseline_folder_name)
            create_exp_path(args.baseline_folder_name,args.baseline_name, repetition=training_repetition)
            training_loop(args, seed, training_repetition)
    
    # p = multiprocessing.Process(target=train.execute,
    #                             args=(gpu, exp_batch, exp_alias, suppress_output, number_of_workers))
    # p.start()
    

#TODO training repetitions for yaml files with different seeds
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    
    # Agent configs

    parser.add_argument('--dataset-root',  dest='dataset_root',required=True)
    
    parser.add_argument('--gpu', default=0, required=True)
    parser.add_argument('--agent', default='autoagents/image_agent')
    parser.add_argument('--baseline-config', dest="baseline_config", default='experiments/config_nocrash.yaml')
    

    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')

    parser.add_argument('--timeout', default="60.0",
                        help='Set the CARLA client timeout value in seconds')
                        
    parser.add_argument('--port', type=int, default=2000)

    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of dataset repetitions.')
    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument("--resume",type=bool, default=False)
    parser.add_argument("--baseline-name", default="arp_vanilla", dest='baseline_name',help="")
    parser.add_argument("--baseline-folder-name", default="ARP", dest='baseline_folder_name',help="")
    parser.add_argument(
        '-vd',
        '--val-datasets',
        dest='validation_datasets',
        nargs='+',
        default=[]
    )

    parser.add_argument(
        '-nw',
        '--number-of-workers',
        dest='number_of_workers',
        type=int,
        default=1
    )
    parser.add_argument(
        '-si', '--save-images',
        action='store_true',
        dest='save_images',
        help='Set to save the images'
    )
    parser.add_argument(
        '-nsv', '--not-save-videos',
        action='store_true',
        dest='not_save_videos',
        help='Set to not save the videos'
    )
    parser.add_argument(
        '-spv', '--save-processed-videos',
        action='store_true',
        dest='save_processed_videos',
        help='Set to save the processed image'
    )
    parser.add_argument(
        '-pr', '--policy-roll-out',
        action='store_true',
        dest='policy_roll_out',
        help='Set to save the policy roll out'
    )
    
    args = parser.parse_args()
    
    main(args)
