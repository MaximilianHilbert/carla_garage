from coil_utils.baseline_helpers import generate_experiment_name
import subprocess
import time
from pathlib import Path
import os
import fnmatch
import xml.etree.ElementTree as ET
import ujson
from tqdm import tqdm
import sys
import numpy as np
import pandas as pd

# Our centOS is missing some c libraries.
# Usually miniconda has them, so we tell the linker to look there as well.
# newlib = '/mnt/lustre/work/geiger/gwb629/conda/garage/lib'
# if not newlib in os.environ['LD_LIBRARY_PATH']:
#   os.environ['LD_LIBRARY_PATH'] += ':' + newlib


def get_num_jobs(code_root,job_name, username):
    len_usrn = len(username)
    num_running_jobs = int(
        subprocess.check_output(
            f"SQUEUE_FORMAT2='username:{len_usrn},name:130' squeue --sort V | grep {username} | grep {job_name} | wc -l",
            shell=True,
        )
        .decode("utf-8")
        .replace("\n", "")
    )
    with open(os.path.join(code_root,"max_num_jobs.txt"), "r", encoding="utf-8") as f:
        max_num_parallel_jobs = int(f.read())

    return num_running_jobs, max_num_parallel_jobs


def main(args):

    if args.cluster=="tcml":
        partition = "day"
        username = "hilbert"
        code_root="/home/hilbert/carla_garage"
    if args.cluster=="galvani":
        partition = "2080-galvani"
        username = "gwb629"
        code_root="/mnt/lustre/work/geiger/gwb629/carla_garage"
    else:
        partition = "2080-galvani"
        username = "gwb629"
        code_root="/home/maximilian/Master/carla_garage"
    epochs = ["31"]
    seeds = [10214, 43534, 53543]
    dataset_root=os.environ.get("DATASET_ROOT")
    log_dir = os.path.join(code_root, "_logs")
    setting="02_withheld"
    job_nr = 0
    already_placed_files = {}
    meta_jobs = {}
    experiment_name_stem = "training"
    for baseline, batch_size in zip(args.baseline_folder_names, args.batch_sizes):
        _,ablations_dict=generate_experiment_name(args, baseline)
        for repetition, seed in enumerate(seeds):
            model_dir=os.path.join(log_dir, baseline, args.experiment_id, f"repetition_{repetition}",setting,"checkpoints", "31.pth")
            if os.path.exists(model_dir):
                print(f"Training already finished for {model_dir}")
                continue
            
            train_filename = (baseline+"_"+
            experiment_name_stem
            + f"_e-{args.experiment_id}")
            
            bash_save_dir = Path(
                os.path.join(
                    code_root,
                    "training",
                    baseline,
                    args.experiment_id,
                    f"repetition_{repetition}",
                    setting,
                    "run_bashs",
                )
            )

            logs_save_dir = Path(
                os.path.join(
                    code_root,
                    "training",
                    baseline,
                    args.experiment_id,
                    f"repetition_{repetition}",
                    setting,
                    "logs",
                )
            )
            bash_save_dir.mkdir(parents=True, exist_ok=True)
            logs_save_dir.mkdir(parents=True, exist_ok=True)
            
            if os.path.exists(model_dir):
                need_to_resubmit=False
            else:
                need_to_resubmit=True

            # Finds a free port
            if args.cluster=="tcml":
                command = f"""#!/bin/sh
#SBATCH --job-name={baseline}_{repetition}_{setting}_{args.experiment_id}
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=0-72:00
#SBATCH --gres=gpu:4
#SBATCH --partition=week
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=9G
#SBATCH --output={os.path.join(logs_save_dir,job_nr, ".out")}
#SBATCH --error={os.path.join(logs_save_dir,job_nr, ".err")}


export WORK_DIR=/home/hilbert/carla_garage
export CONFIG_ROOT=$WORK_DIR/coil_configuration
export TEAM_CODE=$WORK_DIR/team_code
export CARLA_ROOT=$WORK_DIR/carla
export DATASET_ROOT={dataset_root}
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
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d $TEAM_CODE/coil_train.py --setting {args.setting} --lossweights {' '.join(map(str, args.lossweights))} --experiment-id {args.experiment_id} --seed {seed} --batch-size {batch_size} --number-of-workers 4 --baseline-folder-name {baseline} --training-repetition {repetition} \
"""+"--"+" ".join([f' --{key} {value}' for key, value in ablations_dict.items()])
        if args.cluster in ["galvani", "debug"]:


                command = f"""#!/bin/sh
#SBATCH --job-name={baseline}_{repetition}_{setting}_{args.experiment_id}
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=0-72:00
#SBATCH --gres=gpu:8
#SBATCH --partition=2080-galvani
#SBATCH --cpus-per-task=64
#SBATCH --output={os.path.join(logs_save_dir,f"{job_nr}.out")}
#SBATCH --error={os.path.join(logs_save_dir,f"{job_nr}.err")}


export WORK_DIR=/mnt/lustre/work/geiger/gwb629/carla_garage
export CONFIG_ROOT=$WORK_DIR/coil_configuration
export CARLA_ROOT=$WORK_DIR/carla
export DATASET_ROOT={dataset_root}
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
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d $TEAM_CODE/coil_train.py --setting {args.setting} --lossweights {' '.join(map(str, args.lossweights))} --experiment-id {args.experiment_id} --seed {seed} --batch-size {batch_size} --number-of-workers 8 --baseline-folder-name {baseline} --training-repetition {repetition} \
"""+"--"+" ".join([f' --{key} {value}' for key, value in ablations_dict.items()])
        job_file=f"{bash_save_dir}/train_{train_filename}.sh"
        with open(job_file, "w", encoding="utf-8") as rsh:
            rsh.write(command)

        # Wait until submitting new jobs that the #jobs are at below max
        (
            num_running_jobs,
            max_num_parallel_jobs,
        ) = get_num_jobs(
            code_root=code_root,
            job_name=experiment_name_stem,
            username=username,
        )
        print(f"{num_running_jobs}/{max_num_parallel_jobs} jobs are running...")
        while num_running_jobs >= max_num_parallel_jobs:
            (
                num_running_jobs,
                max_num_parallel_jobs,
            ) = get_num_jobs(code_root=code_root,
                job_name=experiment_name_stem,
                username=username,
            )
        time.sleep(0.05)
        if train_filename not in already_placed_files.keys():
            print(f"Submitting job {job_nr}: {job_file}")
            _=subprocess.check_output(
                    f"chmod u+x {job_file}",
                    shell=True,
                )
            jobid = (
                subprocess.check_output(
                    f"sbatch {job_file}",
                    shell=True,
                )
                .decode("utf-8")
                .strip()
                .rsplit(" ", maxsplit=1)[-1]
            )
            meta_jobs[jobid] = (
                False,
                job_file,
                0,
            )
            already_placed_files[train_filename] = job_file
            job_nr += 1

   
    training_finished = False
    while not training_finished:
        num_running_jobs, max_num_parallel_jobs = get_num_jobs(code_root=code_root,job_name=experiment_name_stem, username=username)
        print(f"{num_running_jobs} jobs are running...")
        time.sleep(1)

        # resubmit unfinished jobs
        for k in list(meta_jobs.keys()):
            (
                job_finished,
                job_file,
                resubmitted,
            ) = meta_jobs[k]
            need_to_resubmit = False
            if not job_finished and resubmitted < 50:
                # check whether job is running
                if int(subprocess.check_output(f"squeue | grep {k} | wc -l", shell=True).decode("utf-8").strip()) == 0:
                    # check whether result file is finished?
                    if os.path.exists(model_dir):
                        print(f"Training finished for {model_dir}")
                    else:        
                        need_to_resubmit = True
                    if not need_to_resubmit:
                        # delete old job
                        print(f"Finished job {job_file}")
                        meta_jobs[k] = (True, None, 0)
                    else:
                        need_to_resubmit = True

            if need_to_resubmit:
                # print("Remove file: ", result_file)
                # Path(result_file).unlink()
                print(f"resubmit sbatch {job_file}")
                jobid = (
                    subprocess.check_output(f"sbatch {job_file}", shell=True)
                    .decode("utf-8")
                    .strip()
                    .rsplit(" ", maxsplit=1)[-1]
                )
                meta_jobs[jobid] = (
                    False,
                    job_file,
                    resubmitted + 1,
                )
                num_running_jobs += 1
            time.sleep(1)

            if num_running_jobs == 0:
                training_finished = True
               

    print("Training finished")


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
    
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
    )

    parser.add_argument("--train-local", type=int, default=0)
    parser.add_argument("--datarep", type=int, default=1)

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