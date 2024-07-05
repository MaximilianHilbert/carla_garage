# export WORK_DIR=/home/maximilian/Master/carla_garage
# export CONFIG_ROOT=${WORK_DIR}/coil_configuration
# export TEAM_CODE=$WORK_DIR/team_code
# export CARLA_ROOT=${WORK_DIR}/carla
# export COIL_NETWORK=${WORK_DIR}/coil_network
# export DATASET_ROOT=/home/maximilian/training_data_split
#mlcloud
export WORK_DIR=/mnt/lustre/work/geiger/gwb629/carla_garage
export CONFIG_ROOT=${WORK_DIR}/coil_configuration
export CARLA_ROOT=${WORK_DIR}/carla
export DATASET_ROOT=/mnt/lustre/work/geiger/bjaeger25/old_repos/datasets/hb_dataset_v08_2023_05_10
export LD_LIBRARY_PATH="/mnt/lustre/work/geiger/gwb629/conda/garage/lib":$LD_LIBRARY_PATH
export TEAM_CODE=$WORK_DIR/team_code
export CARLA_ROOT=${WORK_DIR}/carla
export COIL_NETWORK=${WORK_DIR}/coil_network
#tcml
# export WORK_DIR=/home/hilbert/carla_garage
# export CONFIG_ROOT=${WORK_DIR}/coil_configuration
# export TEAM_CODE=$WORK_DIR/team_code
# export CARLA_ROOT=${WORK_DIR}/carla
# export DATASET_ROOT=/home/hilbert/dataset_v08
# export LD_LIBRARY_PATH="/home/hilbert/miniconda3/envs/garage/lib":$LD_LIBRARY_PATH

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export PYTHONPATH=$PYTHONPATH:$CONFIG_ROOT
export PYTHONPATH=$PYTHONPATH:${COIL_NETWORK}
export PYTHONPATH=$PYTHONPATH:${TEAM_CODE}
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}
#seeds must match repetition number
python $WORK_DIR/train_nocrash_baselines.py --repetitions 3 \
 --seeds 10214 43534 53543 --bev 1 --detectboxes 1 --backbone videoresnet \
 --baseline-folder-names bcso bcoh keyframes arp --batch-sizes 10 3 3 3 \
 --walltimes 30 30 30 30 --pretrained 1 --train-local 0 \
 --setting 02_withheld --cluster galvani
