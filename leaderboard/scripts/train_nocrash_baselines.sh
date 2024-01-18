#local
# export WORK_DIR=/home/maximilian/Master/carla_garage
# export CONFIG_ROOT=${WORK_DIR}/coil_config
# export CARLA_ROOT=${WORK_DIR}/carla
# export DATASET_ROOT=/home/maximilian/carla100
#cluster
export WORK_DIR=/mnt/qb/work/geiger/gwb629/carla_garage
export CONFIG_ROOT=${WORK_DIR}/coil_config
export CARLA_ROOT=${WORK_DIR}/carla
export DATASET_ROOT=/mnt/qb/work2/geiger0/bjaeger25/datasets/hb_dataset_v08_2023_05_10
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/mnt/qb/work/geiger/gwb629/conda/garage/lib"
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export PYTHONPATH=$PYTHONPATH:${CONFIG_ROOT}



python3 ${WORK_DIR}/train_nocrash_baselines.py --dataset-root ${DATASET_ROOT} --gpu 0 --agent ${WORK_DIR}/team_code/coil_agent.py --baseline-config ${WORK_DIR}/coil_config 
