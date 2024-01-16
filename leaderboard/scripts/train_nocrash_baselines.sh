#local
export WORK_DIR=/home/maximilian/Master/carla_garage
export DATASET_ROOT=/home/maximilian/carla100

python3 ${WORK_DIR}/train_nocrash_baselines.py --dataset-root ${DATASET_ROOT} --gpu 0 --agent ${WORK_DIR}/team_code/coil_agent.py --baseline-config ${WORK_DIR}/coil_config 
