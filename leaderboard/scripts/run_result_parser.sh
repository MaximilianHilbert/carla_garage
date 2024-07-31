export WORK_DIR=/home/hilbert/carla_garage
export CONFIG_ROOT=$WORK_DIR/coil_configuration
export TEAM_CODE=$WORK_DIR/team_code
export COIL_NETWORK=$WORK_DIR/coil_network


export CARLA_ROOT=/home/hilbert/carla_garage/carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=$WORK_DIR/scenario_runner
export LEADERBOARD_ROOT=$WORK_DIR/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}
export PYTHONPATH=$PYTHONPATH:$CONFIG_ROOT
export PYTHONPATH=$PYTHONPATH:$COIL_NETWORK
export PYTHONPATH=$PYTHONPATH:$TEAM_CODE
export PYTHONPATH=$PYTHONPATH:$WORK_DIR

python $WORK_DIR/tools/result_parser.py --xml $WORK_DIR/leaderboard/data/longest6.xml \
 --results $WORK_DIR/evaluation/ --log_dir $WORK_DIR/evaluation/ \
 --town_maps $WORK_DIR/leaderboard/data/town_maps_xodr --map_dir $WORK_DIR/leaderboard/data/town_maps_tga \
 --device cpu --map_data_folder $WORK_DIR/tools/proxy_simulator/map_data \
 --subsample 1 --strict --visualize_infractions
