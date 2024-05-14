#!/bin/sh
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

python $WORK_DIR/tools/closed_loop_copycat_tools.py