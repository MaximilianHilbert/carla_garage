#!/bin/sh
#SBATCH --job-name=bcso_test
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=2080-galvani
#SBATCH --time=00-72:00
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --output=/mnt/lustre/work/geiger/gwb629/slurmlogs/bcso_test.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/lustre/work/geiger/gwb629/slurmlogs/bcso_test.err   # File to which STDERR will be written

#local
# export WORK_DIR=/home/maximilian/Master/carla_garage
# export CONFIG_ROOT=${WORK_DIR}/coil_configuration
# export TEAM_CODE=$WORK_DIR/team_code
# export CARLA_ROOT=${WORK_DIR}/carla
# export COIL_NETWORK=${WORK_DIR}/coil_network
# export DATASET_ROOT=/home/maximilian/training_data_split
#cluster tcml
# export WORK_DIR=/home/hilbert/carla_garage
# export CONFIG_ROOT=${WORK_DIR}/coil_configuration
# export TEAM_CODE=$WORK_DIR/team_code
# export CARLA_ROOT=${WORK_DIR}/carla
# export DATASET_ROOT=/home/hilbert/dataset_v08
# export LD_LIBRARY_PATH="/home/hilbert/miniconda3/envs/garage/lib":$LD_LIBRARY_PATH

#uni pc
# export WORK_DIR=/home/maximilian-hilbert/carla_garage
# export CONFIG_ROOT=${WORK_DIR}/coil_configuration
# export CARLA_ROOT=${WORK_DIR}/carla
# export DATASET_ROOT=/home/maximilian-hilbert/datasets/tf_dataset
# export LD_LIBRARY_PATH="/mnt/lustre/work/geiger/gwb629/conda/garage/lib":$LD_LIBRARY_PATH
# export TEAM_CODE=$WORK_DIR/team_code
# export COIL_NETWORK=${WORK_DIR}/coil_network
#mlcloud
export WORK_DIR=/mnt/lustre/work/geiger/gwb629/carla_garage
export CONFIG_ROOT=${WORK_DIR}/coil_configuration
export CARLA_ROOT=${WORK_DIR}/carla
export DATASET_ROOT=/mnt/lustre/work/geiger/bjaeger25/old_repos/datasets/hb_dataset_v08_2023_05_10
export LD_LIBRARY_PATH="/mnt/lustre/work/geiger/gwb629/conda/garage/lib":$LD_LIBRARY_PATH
export TEAM_CODE=$WORK_DIR/team_code
export COIL_NETWORK=${WORK_DIR}/coil_network

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export PYTHONPATH=$PYTHONPATH:$CONFIG_ROOT
export PYTHONPATH=$PYTHONPATH:${COIL_NETWORK}
export PYTHONPATH=$PYTHONPATH:${TEAM_CODE}
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}

#mlcloud
source ~/.bashrc
conda activate /mnt/lustre/work/geiger/gwb629/conda/garage
#tcml
# source /home/hilbert/.bashrc
# eval "$(conda shell.bash hook)"
# conda activate garage
export OMP_NUM_THREADS=64  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d $TEAM_CODE/coil_train.py --seed 10214 --baseline-folder-name bcso --number-of-workers 8 --batch-size 10 --setting 02_withheld --bev 1 --detectboxes 1 --freeze 0 --training-repetition 0 --swin 1
