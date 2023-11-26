#!/bin/bash

export CARLA_ROOT=/home/maximilian/Master/carla_garage/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/maximilian/anaconda3/lib

export OMP_NUM_THREADS=20  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
export CUDA_VISIBLE_DEVICES=0
torchrun --nnodes=1 --nproc_per_node=1 --max_restarts=1 \
--rdzv_id=42353467 --rdzv_backend=c10d train.py --id train_id_000 \
--batch_size 2 --setting 02_05_withheld --root_dir /home/maximilian/Master/train_debugging \
--logdir /home/maximilian/Master/log_debugging --use_controller_input_prediction 1 \
--use_wp_gru 0 --use_discrete_command 1 --use_tp 1 --continue_epoch 1 --cpu_cores 4 \
--num_repetitions 1
