#uni pc
export WORK_DIR=/home/maximilian-hilbert/carla_garage
export CONFIG_ROOT=${WORK_DIR}/coil_configuration
export CARLA_ROOT=${WORK_DIR}/carla
export DATASET_ROOT=/home/maximilian-hilbert/datasets/tf_dataset
export LD_LIBRARY_PATH="/mnt/lustre/work/geiger/gwb629/conda/garage/lib":$LD_LIBRARY_PATH
export TEAM_CODE=$WORK_DIR/team_code
export COIL_NETWORK=${WORK_DIR}/coil_network

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export PYTHONPATH=$PYTHONPATH:$CONFIG_ROOT
export PYTHONPATH=$PYTHONPATH:${COIL_NETWORK}
export PYTHONPATH=$PYTHONPATH:${TEAM_CODE}
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}

export OMP_NUM_THREADS=12  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=100 \
 --rdzv_backend=c10d $WORK_DIR/measure_velocity_accel_stats.py --seed 10214 \
 --baseline-folder-name bcoh --number-of-workers 12 \
 --batch-size 1 \
 --bev 1 --detectboxes 1 --predict_vectors 1 \
 --training-repetition 0 --backbone swin \
 --experiment-id id_000
