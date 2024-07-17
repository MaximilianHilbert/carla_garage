export CARLA_ROOT=${1:-/home/maximilian/Master/carla_garage/carla}
export WORK_DIR=${2:-/home/maximilian/Master/carla_garage}
export TEAM_CODE=$WORK_DIR/team_code
export COIL_UTILS=$WORK_DIR/coil_utils
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}:${COIL_UTILS}:${TEAM_CODE}:${WORK_DIR}

export SCENARIOS=${WORK_DIR}/leaderboard/data/scenarios/eval_scenarios.json
export ROUTES=${WORK_DIR}/leaderboard/data/longest6.xml
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=SENSORS
export TEAM_AGENT=${WORK_DIR}/team_code/coil_agent.py
export TEAM_CONFIG=${WORK_DIR}/_logs/bcso/id_001/repetition_0/02_withheld
export DEBUG_CHALLENGE=0
export RESUME=1
export DATAGEN=0
export SAVE_PATH=${WORK_DIR}/results
export DIRECT=0
export BENCHMARK=longest6


python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=1 \
--resume=${RESUME} \
--timeout=600 \
--visualize-combined=1 \
--experiment-id id_000 \
--trafficManagerSeed 252534