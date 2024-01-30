#!/bin/sh
#SBATCH --job-name=reproduce_ARP_arp_vanilla_single
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=day
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:A4000:1
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=12
#SBATCH --output=/home/hilbert/slurmlogs/%j.out  # File to which STDOUT will be written
#SBATCH --error=/home/hilbert/slurmlogs/%j.err   # File to which STDERR will be written

#local
# export WORK_DIR=/home/maximilian/Master/carla_garage
# export CONFIG_ROOT=${WORK_DIR}/coil_configuration
# export TEAM_CODE=$WORK_DIR/team_code
# export CARLA_ROOT=${WORK_DIR}/carla
# export COIL_NETWORK=${WORK_DIR}/coil_network
# export DATASET_ROOT=/home/maximilian/training_data_split
#cluster tcml
export WORK_DIR=/home/hilbert/carla_garage
export CONFIG_ROOT=${WORK_DIR}/coil_configuration
export TEAM_CODE=$WORK_DIR/team_code
export CARLA_ROOT=${WORK_DIR}/carla
export DATASET_ROOT=/home/hilbert/dataset_v08
export LD_LIBRARY_PATH="/home/hilbert/miniconda3/envs/garage/lib":$LD_LIBRARY_PATH
#cluster
#export WORK_DIR=/mnt/qb/work/geiger/gwb629/carla_garage
#export CONFIG_ROOT=${WORK_DIR}/coil_configuration
#export TEAM_CODE=$WORK_DIR/team_code
#export CARLA_ROOT=${WORK_DIR}/carla
#export DATASET_ROOT=/mnt/qb/work/geiger/gwb629/datasets/hb_dataset_v08_2023_05_10
#export LD_LIBRARY_PATH="/mnt/qb/work/geiger/gwb629/conda/garage/lib":$LD_LIBRARY_PATH

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
# source ~/.bashrc
# conda activate /mnt/qb/work/geiger/gwb629/conda/garage
#tcml
source /home/hilbert/.bashrc
eval "$(conda shell.bash hook)"
conda activate garage

python $TEAM_CODE/coil_train.py --seed 1 --gpu 0 --baseline_folder_name ARP --baseline_name arp_vanilla --number_of_workers 12 --training_repetition 0 --use-disk-cache 0