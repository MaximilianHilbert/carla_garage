import os
import subprocess
def generate_batch_script(args, seed, training_repetition, baseline_folder_name, baseline_name, batch_size, walltime):
    if args.train_local:
        subprocess.check_output(f"torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d $TEAM_CODE/coil_train.py --seed {seed} --training_repetition {training_repetition} --use-disk-cache {args.use_disk_cache} --baseline_folder_name {baseline_folder_name} --baseline_name {baseline_name} --number_of_workers {int(args.number_of_cpus/8)} --batch-size {batch_size}", shell=True)
    else:
        job_path=os.path.join(os.environ.get("WORK_DIR"), "job_files")
        os.makedirs(job_path, exist_ok=True)
        job_full_path=os.path.join(job_path, f"{baseline_folder_name}_{baseline_name.replace('.yaml', '')}_{str(training_repetition)}.sh")
        with open(job_full_path, 'w', encoding='utf-8') as f:
            command=f"""#!/bin/sh
#SBATCH --job-name=reproduce_{baseline_folder_name}_{baseline_name}_{training_repetition}
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=0-{walltime}:00
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu-2080ti
#SBATCH --cpus-per-task={args.number_of_cpus}
#SBATCH --mem=350G
#SBATCH --output=/mnt/qb/work/geiger/gwb629/slurmlogs/%j.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/qb/work/geiger/gwb629/slurmlogs/%j.err   # File to which STDERR will be written

#export WORK_DIR=/home/hilbert/carla_garage
#export CONFIG_ROOT=$WORK_DIR/coil_configuration
#export TEAM_CODE=$WORK_DIR/team_code
#export CARLA_ROOT=$WORK_DIR/carla
#export DATASET_ROOT=/home/hilbert/dataset_v08
#export LD_LIBRARY_PATH="/home/hilbert/miniconda3/envs/garage/lib":$LD_LIBRARY_PATH

export WORK_DIR=/mnt/qb/work/geiger/gwb629/carla_garage
export CONFIG_ROOT=$WORK_DIR/coil_configuration
export CARLA_ROOT=$WORK_DIR/carla
export DATASET_ROOT=/mnt/qb/work2/geiger0/bjaeger25/datasets/hb_dataset_v08_2023_05_10
export LD_LIBRARY_PATH="/mnt/qb/work/geiger/gwb629/conda/garage/lib":$LD_LIBRARY_PATH

export CARLA_SERVER=$CARLA_ROOT/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla/":$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$CONFIG_ROOT
export PYTHONPATH=$PYTHONPATH:$COIL_NETWORK
export PYTHONPATH=$PYTHONPATH:$TEAM_CODE
export PYTHONPATH=$PYTHONPATH:$WORK_DIR

# source /home/hilbert/.bashrc
# eval "$(conda shell.bash hook)"
# conda activate garage
source ~/.bashrc
conda activate /mnt/qb/work/geiger/gwb629/conda/garage
export OMP_NUM_THREADS={args.number_of_cpus}  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d $TEAM_CODE/coil_train.py --seed {seed} --training_repetition {training_repetition} --use-disk-cache {args.use_disk_cache} --baseline_folder_name {baseline_folder_name} --baseline_name {baseline_name} --number_of_workers {int(args.number_of_cpus/8)} --batch-size {batch_size}
        """
            f.write(command)
def place_batch_scripts():
    root=os.path.join(os.environ.get("WORK_DIR"), "job_files")
    for file in os.listdir(root):
        full_path=os.path.join(root, file)
        out=subprocess.check_output(f'chmod u+x {full_path}', shell=True)
        out=subprocess.check_output(f"sbatch {full_path}", shell=True)
        print(out)
def main(args):
    for training_repetition, seed in enumerate(args.seeds):
        for baseline_folder_name, batch_size, walltime in zip(args.baseline_folder_names,args.batch_sizes, args.walltimes):
            for baseline_name in os.listdir(os.path.join(os.environ.get("CONFIG_ROOT"), baseline_folder_name)):
                generate_batch_script(args,seed, training_repetition, baseline_folder_name, baseline_name, batch_size, walltime)
    if not args.train_local:
        place_batch_scripts()
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_folder_names", nargs='+', type=str, dest='baseline_folder_names',help="")
    parser.add_argument('--seeds', nargs='+', type=int, help='List of seed values')
    parser.add_argument('--use-disk-cache', dest="use_disk_cache", type=int, default=0)
    parser.add_argument(
         '--repetitions',
        type=int,
        default=1,
        help='Number of dataset repetitions.')
    parser.add_argument(
        '-nw',
        '--number_of_cpus',
        dest='number_of_cpus',
        type=int,
        default=1
    )
    parser.add_argument(
        '--batch-sizes',
        nargs="+",
        type=int,
    )
    parser.add_argument(
        '--train-local',
        dest='train_local',
        type=int,
        default=0
    )
    parser.add_argument(
        '--walltimes',
        nargs="+",
        type=int,
    )
    args = parser.parse_args()
    main(args)
