export CARLA_ROOT=${1:-/home/maximilian/Master/carla_garage/carla}
export WORK_DIR=${2:-/home/maximilian/Master/carla_garage}

export CONFIG_ROOT=${WORK_DIR}/coil_configuration
export TEAM_CODE=$WORK_DIR/team_code
export COIL_NETWORK=$WORK_DIR/coil_network
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla/:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH}:${WORK_DIR}:${CONFIG_ROOT}:${TEAM_CODE}:${COIL_NETWORK}

export SCENARIOS=${WORK_DIR}/leaderboard/data/scenarios/eval_scenarios.json
export ROUTES=${WORK_DIR}/leaderboard/data/longest6.xml
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=SENSORS
export CHECKPOINT_ENDPOINT=${WORK_DIR}/results/arp.json
export TEAM_AGENT=${WORK_DIR}/team_code/coil_agent.py
export TEAM_CONFIG=${WORK_DIR}/coil_config/coil_config.py
export DEBUG_CHALLENGE=0
export RESUME=1
export DATAGEN=0
export SAVE_PATH=${WORK_DIR}/results
export DIRECT=0
export BENCHMARK=longest6
export COIL_MODEL_CHECKPOINT=/home/maximilian/Master/carla_garage/pretrained_models/arp/checkpoints/100000.pth

python3 ${WORK_DIR}/evaluate_nocrash_baselines.py \
--repetitions=${REPETITIONS} \
--coil_checkpoint=${COIL_MODEL_CHECKPOINT} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--resume=${RESUME} \
--timeout=600 \
--resume=true \
--eval_id=id_000 \
--town=Town01 \
--weather=train \
--log_path=${WORK_DIR}/logs

