import os
import subprocess
import numpy as np
from coil_utils.baseline_helpers import get_ablations_dict
import pickle
from coil_utils.baseline_helpers import get_latest_saved_checkpoint, generate_experiment_name
max_img_seq_len_baselines=6
def generate_batch_script(
    args,
    seed,
    training_repetition,
    baseline_folder_name,
    ablations_dict,
    batch_size,
    walltime,
    complete_string
):
    if args.train_local:
        subprocess.check_output(
            f"torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d $TEAM_CODE/coil_train.py --seed {seed} --training-repetition {training_repetition} --use-disk-cache {args.use_disk_cache} --baseline-folder-name {baseline_folder_name} --experiment {experiment} --number-of-workers {int(args.number_of_cpus/8)} --batch-size {batch_size}",
            shell=True,
        )
    else:
        job_path = os.path.join(os.environ.get("WORK_DIR"), "job_files")
        os.makedirs(job_path, exist_ok=True)
        job_full_path = os.path.join(
            job_path,
            f"{complete_string}.sh")
        with open(job_full_path, "w", encoding="utf-8") as f:
            if args.cluster=="tcml":
                 command = f"""#!/bin/sh
#SBATCH --job-name={complete_string}
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=0-{walltime}:00
#SBATCH --gres=gpu:4
#SBATCH --partition=week
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=9G
#SBATCH --output="/home/hilbert/slurmlogs/{complete_string}.out"
#SBATCH --error="/home/hilbert/slurmlogs/{complete_string}.err"


export WORK_DIR=/home/hilbert/carla_garage
export CONFIG_ROOT=$WORK_DIR/coil_configuration
export TEAM_CODE=$WORK_DIR/team_code
export CARLA_ROOT=$WORK_DIR/carla
export DATASET_ROOT=$DATASET_ROOT
export LD_LIBRARY_PATH="/home/hilbert/miniconda3/envs/garage/lib":$LD_LIBRARY_PATH


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
export OMP_NUM_THREADS=24  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d $TEAM_CODE/coil_train.py --use-disk-cache {args.use_disk_cache} --setting {args.setting} --lossweights {' '.join(map(str, args.lossweights))} --experiment-id {args.experiment_id} --seed {seed} --batch-size {batch_size} --number-of-workers 4 --baseline-folder-name {baseline_folder_name} --training-repetition {training_repetition} \
"""+"--"+" --".join([f'{ablation} {" ".join(map(str,value)) if isinstance(value, list) else value}' for ablation, value in ablations_dict.items()])
            else:


                command = f"""#!/bin/sh
#SBATCH --job-name={complete_string}
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=0-{walltime}:00
#SBATCH --gres=gpu:8
#SBATCH --partition=2080-galvani
#SBATCH --cpus-per-task=64
#SBATCH --output="/mnt/lustre/work/geiger/gwb629/slurmlogs/{complete_string}.out"
#SBATCH --error="/mnt/lustre/work/geiger/gwb629/slurmlogs/{complete_string}.err"


export WORK_DIR=/mnt/lustre/work/geiger/gwb629/carla_garage
export CONFIG_ROOT=$WORK_DIR/coil_configuration
export CARLA_ROOT=$WORK_DIR/carla
export DATASET_ROOT=$DATASET_ROOT
export LD_LIBRARY_PATH="/mnt/lustre/work/geiger/gwb629/conda/garage/lib":$LD_LIBRARY_PATH
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

source ~/.bashrc
conda activate /mnt/lustre/work/geiger/gwb629/conda/garage
export OMP_NUM_THREADS=64  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d $TEAM_CODE/coil_train.py --use-disk-cache {args.use_disk_cache} --setting {args.setting} --lossweights {' '.join(map(str, args.lossweights))} --experiment-id {args.experiment_id} --seed {seed} --batch-size {batch_size} --number-of-workers 8 --baseline-folder-name {baseline_folder_name} --training-repetition {training_repetition} \
"""+"--"+" --".join([f'{ablation} {" ".join(map(str,value)) if isinstance(value, list) else value}' for ablation, value in ablations_dict.items()])
            f.write(command)


def place_batch_scripts():
    root = os.path.join(os.environ.get("WORK_DIR"), "job_files")
    if os.path.isdir(root):
        for file in os.listdir(root):
            full_path = os.path.join(root, file)
            out = subprocess.check_output(f'chmod u+x {full_path}', shell=True)
            out = subprocess.check_output(f'sbatch {full_path}', shell=True)
            print(out)
    else:
        print("nothing to submit")

def main(args):
    for training_repetition, seed in enumerate(args.seeds):
        for baseline_folder_name, batch_size, walltime in zip(
            args.baseline_folder_names, args.batch_sizes, args.walltimes
        ):
            if args.single_process:
                for experiment in args.single_process:
                    generate_batch_script(
                    args,
                    seed,
                    training_repetition,
                    baseline_folder_name,
                    batch_size,
                    walltime,

                )
            else:
                experiment_string,ablations_dict=generate_experiment_name(args, baseline_folder_name)
                # in case we want to ablate prevnum we set it to the past 6 waypoints, because our baselines consider 6 previous frames
                final_log_dir=os.path.join(os.environ.get("WORK_DIR"), "_logs", baseline_folder_name, experiment_string,f"repetition_{training_repetition}", args.setting)
                if os.path.isdir(final_log_dir):
                    checkpoint=get_latest_saved_checkpoint(final_log_dir)
                    if checkpoint==30:
                        continue
                experiment_string=f"{experiment_string}_tr_{training_repetition}"
                generate_batch_script(
                    args,
                    seed,
                    training_repetition,
                    baseline_folder_name,
                    ablations_dict,
                    batch_size,
                    walltime,
                    experiment_string
                )
    if not args.train_local:
        place_batch_scripts()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-folder-names",
        nargs="+",
        type=str,
        help="",
    )
    parser.add_argument(
        "--single-process",
        nargs="+",
        type=str,
        help="",
    )
    
    parser.add_argument("--seeds", nargs="+", type=int, help="List of seed values")
    parser.add_argument("--use-disk-cache", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1, help="Number of dataset repetitions.")
    parser.add_argument("--number-of-cpus", type=int, default=1)
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
    )

    parser.add_argument("--train-local", type=int, default=0)
    parser.add_argument("--datarep", type=int, default=1)
    parser.add_argument(
        "--walltimes",
        nargs="+",
        type=int,
    )
    parser.add_argument(
        "--speed",
        type=int,
        choices=[0,1],
        default=0,
    )
    parser.add_argument(
        "--prevnum",
        type=int,
        choices=[0,1],
        default=0,
        help="n-1 is considered"
    )
    parser.add_argument(
        "--framehandling",
        type=str,
        choices=["stacking","unrolling"],
        default="unrolling",

    )
    parser.add_argument(
        "--bev",
        type=int,
        choices=[0,1],
        default=0

    )
    parser.add_argument(
        "--init",
        type=int,
        choices=[0,1],
        default=0

    )
    parser.add_argument(
        "--detectboxes",
        type=int,
        choices=[0,1],
        default=0

    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=0

    )
    parser.add_argument(
        "--tf_pp_rep",
        type=int,
        default=0

    )
    parser.add_argument(
        "--predict_vectors",
        type=int,
        default=0,
        choices=[0,1]

    )
    parser.add_argument(
        "--lossweights",
        nargs="+",
        type=float,
        default=[0.33, 0.33, 0.33]

    )
    parser.add_argument(
        "--subsampling",
        type=int,
        choices=[0,1],
        default=0
    )
    parser.add_argument(
        "--setting",
        default="02_withheld",
        type=str,
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="tcml"
    )
    parser.add_argument(
        "--augment",
        type=int,
        default=0
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet"
    )
    parser.add_argument(
        "--pretrained",
        type=int,
        default=1
    )
    parser.add_argument(
        "--velocity-brake-prediction",
        type=int,
        default=0
    )
    parser.add_argument(
        "--ego-velocity-prediction",
        type=int,
        default=0
    )
    parser.add_argument(
        "--experiment-id",
        required=True
    )
    parser.add_argument(
        "--rear-cam",
        type=int,
        default=0
    )
    args = parser.parse_args()
    main(args)
