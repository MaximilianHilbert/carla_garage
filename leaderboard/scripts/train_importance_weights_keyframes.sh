#!/bin/bash
#SBATCH --job-name=reproduce_keyframes
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=0-12:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-2080ti,gpu-v100
#SBATCH --cpus-per-task=25
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

python $WORK_DIR/keyframes/get_importance_weights_training.py --seeds 122 --baseline-folder-name keyframes --baseline-name keyframes_vanilla_weights --number-of-workers 25 --use-disk-cache 0 --neurons 300
