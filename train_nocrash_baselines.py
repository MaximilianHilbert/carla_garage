import os
from coil_utils.general import create_log_folder, create_exp_path, erase_logs
import subprocess
def set_environment_variables():
    command="""
        export WORK_DIR=/home/maximilian/Master/carla_garage
        export CONFIG_ROOT=${WORK_DIR}/coil_config
        export CARLA_ROOT=${WORK_DIR}/carla
        export COIL_UTILS=${WORK_DIR}/coil_utils
        export DATASET_ROOT=/media/maximilian/HDD/training_data_split
        export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
        export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
        export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
        export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
        export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
        export PYTHONPATH=$PYTHONPATH:${CONFIG_ROOT}
        export PYTHONPATH=$PYTHONPATH:${COIL_UTILS}
        
        """
    subprocess.run(command, shell=True)
def generate_and_place_batch_script(args,seed, repetition):
    # template=f"""
    # #!/bin/bash
    # #SBATCH --job-name={args.baseline_folder_name,args.baseline_name,seed, repetition}
    # #SBATCH --output=/mnt/qb/work/geiger/gwb629/slurmlogs/%j.out
    # #SBATCH --error=/mnt/qb/work/geiger/gwb629/slurmlogs/%j.err
    # #SBATCH --nodes=1
    # #SBATCH --tasks-per-node=1
    # #SBATCH --gres=gpu:1
    # #SBATCH --mem=32G
    # ##SBATCH --cpus-per-task=32

    # export WORK_DIR=/mnt/qb/work/geiger/gwb629/carla_garage
    # export CONFIG_ROOT=$WORK_DIR/coil_config
    # export CARLA_ROOT=$WORK_DIR/carla
    # export DATASET_ROOT=/mnt/qb/work2/geiger0/bjaeger25/datasets/hb_dataset_v08_2023_05_10
    # export LD_LIBRARY_PATH="/mnt/qb/work/geiger/gwb629/conda/garage/lib":$LD_LIBRARY_PATH
    # export CARLA_SERVER=$CARLA_ROOT/CarlaUE4.sh
    # export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI
    # export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
    # export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
    # export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla/":$PYTHONPATH
    # export PYTHONPATH=$PYTHONPATH:$CONFIG_ROOT
    # """
    # template=template+"conda activate /mnt/qb/work/geiger/gwb629/conda/garage/bin/python"
    template=f"conda run -n garage python3 $WORK_DIR/team_code/coil_train.py --gpu {args.gpu} --seed {seed} --training_repetition {repetition} --baseline_folder_name {args.baseline_folder_name} --baseline_name {args.baseline_name} --number_of_workers {args.number_of_workers}"
    subprocess.run(template, shell=True)
def main(args):
    set_environment_variables()
    for training_repetition, seed in enumerate(args.seeds):
            create_log_folder(f'{os.environ.get("WORK_DIR")}/_logs',args.baseline_folder_name)
            erase_logs(f'{os.environ.get("WORK_DIR")}/_logs',args.baseline_folder_name)
            create_exp_path(f'{os.environ.get("WORK_DIR")}/_logs',args.baseline_folder_name,args.baseline_name, repetition=training_repetition)
            generate_and_place_batch_script(args,seed, training_repetition)
    
    # p = multiprocessing.Process(target=train.execute,
    #                             args=(gpu, exp_batch, exp_alias, suppress_output, number_of_workers))
    # p.start()
    

#TODO training repetitions for yaml files with different seeds
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    
    # Agent configs

    parser.add_argument('--training_seeds', default=1, required=False)
    parser.add_argument('--gpu', default=0, required=True)
    parser.add_argument('--agent', default='autoagents/image_agent')
    parser.add_argument('--baseline_config', dest="baseline_config", default='experiments/config_nocrash.yaml')
    parser.add_argument("--baseline_name", default="arp_vanilla", dest='baseline_name',help="")
    parser.add_argument("--baseline_folder_name", default="ARP", dest='baseline_folder_name',help="")
    parser.add_argument('--seeds', nargs='+', type=int, help='List of seed values')
    

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
    parser.add_argument(
        '-vd',
        '--val-datasets',
        dest='validation_datasets',
        nargs='+',
        default=[]
    )
    
    parser.add_argument(
        '-nw',
        '--number_of_workers',
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
