#local
export WORK_DIR=/home/maximilian/Master/carla_garage
export CONFIG_ROOT=${WORK_DIR}/coil_configuration
export TEAM_CODE=$WORK_DIR/team_code
export CARLA_ROOT=${WORK_DIR}/carla
export COIL_NETWORK=${WORK_DIR}/coil_network
export DATASET_ROOT=/home/maximilian/training_data_split
#cluster
# export WORK_DIR=/mnt/qb/work/geiger/gwb629/carla_garage
# export CONFIG_ROOT=${WORK_DIR}/coil_config
# export CARLA_ROOT=${WORK_DIR}/carla
# export DATASET_ROOT=/mnt/qb/work2/geiger0/bjaeger25/datasets/hb_dataset_v08_2023_05_10
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

python $TEAM_CODE/coil_train.py --seed 1 --gpu 0 --baseline_folder_name ARP --baseline_name arp_vanilla --number_of_workers 12 --training_repetition 0