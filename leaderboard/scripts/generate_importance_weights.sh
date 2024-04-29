#!/bin/bash
#SBATCH --job-name=gen_importance_weights
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=week
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=9G
#SBATCH --output=/home/hilbert/slurmlogs/%j.out  # File to which STDOUT will be written
#SBATCH --error=/home/hilbert/slurmlogs/%j.err   # File to which STDERR will be written
#uni pc


# export WORK_DIR=/home/maximilian-hilbert/carla_garage
# export CONFIG_ROOT=${WORK_DIR}/coil_configuration
# export TEAM_CODE=$WORK_DIR/team_code
# export CARLA_ROOT=${WORK_DIR}/carla
# export COIL_NETWORK=${WORK_DIR}/coil_network
# export DATASET_ROOT=/home/maximilian-hilbert/datasets/transfuser_v08
#local
# export WORK_DIR=/home/maximilian/Master/carla_garage
# export CONFIG_ROOT=${WORK_DIR}/coil_configuration
# export TEAM_CODE=$WORK_DIR/team_code
# export CARLA_ROOT=${WORK_DIR}/carla
# export COIL_NETWORK=${WORK_DIR}/coil_network
# export DATASET_ROOT=/home/maximilian/test

export WORK_DIR=/mnt/qb/work/geiger/gwb629/carla_garage
export CONFIG_ROOT=$WORK_DIR/coil_configuration
export CARLA_ROOT=$WORK_DIR/carla
export DATASET_ROOT=/mnt/qb/work2/geiger0/bjaeger25/datasets/hb_dataset_v08_2023_05_10
export LD_LIBRARY_PATH="/mnt/qb/work/geiger/gwb629/conda/garage/lib":$LD_LIBRARY_PATH
export TEAM_CODE=$WORK_DIR/team_code
export COIL_NETWORK=${WORK_DIR}/coil_network

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

# source ~/.bashrc
# conda activate /mnt/qb/work/geiger/gwb629/conda/garage

python $WORK_DIR/keyframes/get_importance_weights_training.py --seeds 111 --training-repetition 0 --baseline-folder-name keyframes --experiment keyframes_weights --number-of-workers 25 --neurons 300 --batch-size 512 --setting 02_withheld
#python $WORK_DIR/keyframes/get_importance_weights_inference.py --training-repetition 0 --baseline-folder-name keyframes --experiment keyframes_weights --number-of-workers 25 --setting 02_withheld
