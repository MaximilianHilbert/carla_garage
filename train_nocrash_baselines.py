import os
import subprocess
def generate_and_place_batch_script(workers, seed, training_repetition, baseline_folder_name, baseline_name):
    job_path=os.path.join(os.environ.get("WORK_DIR"), "job_files")
    os.makedirs(job_path, exist_ok=True)
    job_full_path=os.path.join(job_path, f"{baseline_folder_name}_{baseline_name}_{str(training_repetition)}.sh")
    with open(job_full_path, 'w', encoding='utf-8') as f:
        command=f"""#!/bin/bash
#SBATCH --job-name=reproduce_{baseline_folder_name}_{baseline_name}_{training_repetition}
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=03-00:00
#SBATCH --partition=week
#SBATCH --gres=gpu:A4000:1
#SBATCH --cpus-per-task={workers}
#SBATCH --output=/home/hilbert/slurmlogs/%j.out  # File to which STDOUT will be written
#SBATCH --error=/home/hilbert/slurmlogs/%j.err   # File to which STDERR will be written

#export WORK_DIR=/home/hilbert/carla_garage
#export CONFIG_ROOT=$WORK_DIR/coil_configuration
#export TEAM_CODE=$WORK_DIR/team_code
#export CARLA_ROOT=$WORK_DIR/carla
#export DATASET_ROOT=/home/hilbert/dataset_v08
#export LD_LIBRARY_PATH="/home/hilbert/miniconda3/envs/garage/lib":$LD_LIBRARY_PATH


export CARLA_SERVER=$CARLA_ROOT/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla/":$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$CONFIG_ROOT
export PYTHONPATH=$PYTHONPATH:$COIL_NETWORK
export PYTHONPATH=$PYTHONPATH:$TEAM_CODE
export PYTHONPATH=$PYTHONPATH:$WORK_DIR

source /home/hilbert/.bashrc
eval "$(conda shell.bash hook)"
conda activate garage
python3 $WORK_DIR/team_code/coil_train.py --gpu {args.gpu} --seed {seed} --training_repetition {training_repetition} --use-disk-cache {args.use_disk_cache} --baseline_folder_name {baseline_folder_name} --baseline_name {baseline_name} --number_of_workers {workers}
        """
        f.write(command)
    out=subprocess.check_output(f'chmod u+x {job_full_path}', shell=True)
    out=subprocess.check_output(f"sbatch {job_full_path}", shell=True)
    print(out)
def main(args):
    for training_repetition, seed in enumerate(args.seeds):
        for baseline_folder_name in args.baseline_folder_names:
            for baseline_name in args.baseline_names:
                generate_and_place_batch_script(args.number_of_workers,seed, training_repetition, baseline_folder_name, baseline_name)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, required=True)
    parser.add_argument("--baseline_names", nargs='+', type=str, dest='baseline_names',help="")
    parser.add_argument("--baseline_folder_names", nargs='+', type=str, dest='baseline_folder_names',help="")
    parser.add_argument('--seeds', nargs='+', type=int, help='List of seed values')
    parser.add_argument('--use-disk-cache', dest="use_disk_cache", type=bool, default=0)
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
