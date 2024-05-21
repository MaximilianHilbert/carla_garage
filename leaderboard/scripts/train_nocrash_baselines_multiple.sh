# export WORK_DIR=/home/maximilian/Master/carla_garage
# export CONFIG_ROOT=${WORK_DIR}/coil_configuration
# export TEAM_CODE=$WORK_DIR/team_code
# export CARLA_ROOT=${WORK_DIR}/carla
# export COIL_NETWORK=${WORK_DIR}/coil_network
# export DATASET_ROOT=/home/maximilian/training_data_split
#mlcloud
# export WORK_DIR=/mnt/qb/work/geiger/gwb629/carla_garage
# export CONFIG_ROOT=${WORK_DIR}/coil_configuration
# export CARLA_ROOT=${WORK_DIR}/carla
# export DATASET_ROOT=/mnt/qb/work2/geiger0/bjaeger25/datasets/hb_dataset_v08_2023_05_10
# export LD_LIBRARY_PATH="/mnt/qb/work/geiger/gwb629/conda/garage/lib":$LD_LIBRARY_PATH
# export TEAM_CODE=$WORK_DIR/team_code
# export CARLA_ROOT=${WORK_DIR}/carla
# export COIL_NETWORK=${WORK_DIR}/coil_network
#tcml
export WORK_DIR=/home/hilbert/carla_garage
export CONFIG_ROOT=${WORK_DIR}/coil_configuration
export TEAM_CODE=$WORK_DIR/team_code
export CARLA_ROOT=${WORK_DIR}/carla
export DATASET_ROOT=/home/hilbert/dataset_v08
export LD_LIBRARY_PATH="/home/hilbert/miniconda3/envs/garage/lib":$LD_LIBRARY_PATH

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
#72 cores means, 9 cores per dataloader, per GPU
#when single process is set to one or multiple yaml files within a given baseline folder name, only one baseline folder name is allowed, if single-process is not set multiple folders are allowed and all will be executed in batch mode
python $WORK_DIR/train_nocrash_baselines.py --ablations speed backbone td prevnum --repetitions 3 --seeds 10214 43534 53543 --baseline-folder-names arp bcso bcoh keyframes --use-disk-cache 0 --batch-sizes 6 25 16 16 --walltimes 100 100 100 100 --train-local 0 --dataset-repetition 3 --setting 02_withheld
