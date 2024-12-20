#!/bin/bash
#SBATCH --job-name=gen_importance_weights
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=week
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=9G
#SBATCH --output=/home/hilbert/slurmlogs/gen_importance_weights.out  # File to which STDOUT will be written
#SBATCH --error=/home/hilbert/slurmlogs/gen_importance_weights.err   # File to which STDERR will be written
#uni pc


# export WORK_DIR=/home/maximilian-hilbert/carla_garage
# export CONFIG_ROOT=${WORK_DIR}/coil_configuration
# export TEAM_CODE=$WORK_DIR/team_code
# export CARLA_ROOT=${WORK_DIR}/carla
# export COIL_NETWORK=${WORK_DIR}/coil_network
# export DATASET_ROOT=/home/maximilian-hilbert/datasets/tf_dataset
#local
# export WORK_DIR=/home/maximilian/Master/carla_garage
# export CONFIG_ROOT=${WORK_DIR}/coil_configuration
# export TEAM_CODE=$WORK_DIR/team_code
# export CARLA_ROOT=${WORK_DIR}/carla
# export COIL_NETWORK=${WORK_DIR}/coil_network
# export DATASET_ROOT=/home/maximilian/test
#mlcloud
# export WORK_DIR=/mnt/qb/work/geiger/gwb629/carla_garage
# export CONFIG_ROOT=$WORK_DIR/coil_configuration
# export CARLA_ROOT=$WORK_DIR/carla
# export DATASET_ROOT=/mnt/qb/work2/geiger0/bjaeger25/datasets/hb_dataset_v08_2023_05_10
# export LD_LIBRARY_PATH="/mnt/qb/work/geiger/gwb629/conda/garage/lib":$LD_LIBRARY_PATH
# export TEAM_CODE=$WORK_DIR/team_code
# export COIL_NETWORK=${WORK_DIR}/coil_network
#tcml
export WORK_DIR=/home/hilbert/carla_garage
export CONFIG_ROOT=$WORK_DIR/coil_configuration
export CARLA_ROOT=$WORK_DIR/carla
export DATASET_ROOT=/home/hilbert/datasets/triangular_augmentation_intensity_25_augmented_rear_camera2024_09_16
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
export OMP_NUM_THREADS=12  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
python $WORK_DIR/keyframes/get_importance_weights_inference.py --use-case training --training-repetition 0 --number-of-workers 12 --setting 02_withheld
