import os
from coil_utils.general import create_log_folder, create_exp_path, erase_logs
import subprocess
def generate_and_place_batch_script(args, seed, repetition):
    job_path=os.path.join(os.environ.get("WORK_DIR"), "job_files")
    os.makedirs(job_path, exist_ok=True)
    job_full_path=os.path.join(job_path, f"{args.baseline_folder_name}_{args.baseline_name}_{str(repetition)}.sh")
    with open(job_full_path, 'w', encoding='utf-8') as f:
        command=f"""#!/bin/bash
#SBATCH --job-name=reproduce_{args.baseline_folder_name}_{args.baseline_name}
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00-01:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={args.number_of_workers}
#SBATCH --output=/mnt/qb/work/geiger/gwb629/slurmlogs/carla_garage/%j.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/qb/work/geiger/gwb629/slurmlogs/carla_garage/%j.err   # File to which STDERR will be written

export WORK_DIR=/mnt/qb/work/geiger/gwb629/carla_garage
export CONFIG_ROOT=$WORK_DIR/coil_config
export CARLA_ROOT=$WORK_DIR/carla
export DATASET_ROOT=/mnt/qb/work2/geiger0/bjaeger25/datasets/hb_dataset_v08_2023_05_10
export LD_LIBRARY_PATH="/mnt/qb/work/geiger/gwb629/conda/garage/lib":$LD_LIBRARY_PATH
export COIL_NETWORK=$WORK_DIR/coil_network
export TEAM_CODE=$WORK_DIR/team_code


export CARLA_SERVER=$CARLA_ROOT/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla/":$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$CONFIG_ROOT
export PYTHONPATH=$PYTHONPATH:$COIL_NETWORK
export PYTHONPATH=$PYTHONPATH:$TEAM_CODE
export PYTHONPATH=$PYTHONPATH:$WORK_DIR


conda run -n garage python3 $WORK_DIR/team_code/coil_train.py --gpu {args.gpu} --seed {seed} --training_repetition {repetition} --baseline_folder_name {args.baseline_folder_name} --baseline_name {args.baseline_name} --number_of_workers {args.number_of_workers}
        """
        f.write(command)
    subprocess.Popen(f'chmod u+x {job_full_path}', shell=True)
    subprocess.Popen(["sbatch", job_full_path], shell=True)

def main(args):
    for training_repetition, seed in enumerate(args.seeds):
            create_log_folder(f'{os.environ.get("WORK_DIR")}/_logs',args.baseline_folder_name)
            erase_logs(f'{os.environ.get("WORK_DIR")}/_logs',args.baseline_folder_name)
            create_exp_path(f'{os.environ.get("WORK_DIR")}/_logs',args.baseline_folder_name,args.baseline_name, repetition=training_repetition)
            generate_and_place_batch_script(args,seed, training_repetition)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_seeds', default=1, required=False)
    parser.add_argument('--gpu', default=0, required=True)
    parser.add_argument("--baseline_name", default="arp_vanilla", dest='baseline_name',help="")
    parser.add_argument("--baseline_folder_name", default="ARP", dest='baseline_folder_name',help="")
    parser.add_argument('--seeds', nargs='+', type=int, help='List of seed values')
    parser.add_argument(
         '--repetitions',
        type=int,
        default=1,
        help='Number of dataset repetitions.')
    parser.add_argument(
        '-nw',
        '--number_of_workers',
        dest='number_of_workers',
        type=int,
        default=1
    )
    args = parser.parse_args()
    main(args)
