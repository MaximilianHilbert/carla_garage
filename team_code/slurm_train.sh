#!/bin/bash
#SBATCH --job-name=reproduce_transfuser
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00-48:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --output=/mnt/qb/work/geiger/gwb629/transfuser_vanilla/slurmlogs/%j.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/qb/work/geiger/gwb629/transfuser_vanilla/slurmlogs/%j.err   # File to which STDERR will be written
#SBATCH --partition=gpu-2080ti,gpu-v100

# print info about current job
scontrol show job $SLURM_JOB_ID

pwd
#git clone https://github.com/autonomousvision/carla_garage.git
#cd carla_garage
#chmod +x setup_carla.sh
#./setup_carla.sh
eval "$(conda shell.bash hook)"
#conda env create -f environment.yml
conda activate garage
#conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
#conda install tqdm tensorboard
#conda install -c conda-forge libjpeg-turbo 
export CARLA_ROOT=/mnt/qb/work/geiger/gwb629/transfuser_vanilla/carla_garage/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/geiger/gwb629/.conda/envs/garage/lib/

export OMP_NUM_THREADS=32  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
torchrun --nnodes=1 --nproc_per_node=8 --max_restarts=0 --rdzv_id=$SLURM_JOB_ID \
--rdzv_backend=c10d /mnt/qb/work/geiger/gwb629/transfuser_vanilla/carla_garage/team_code/train.py \
--id train_id_000 --batch_size 8 --setting 02_05_withheld \
--root_dir /mnt/qb/work2/geiger0/bjaeger25/datasets/hb_dataset_v08_2023_05_10 \
--logdir /mnt/qb/work/geiger/gwb629/transfuser_vanilla/model_logs \
--use_controller_input_prediction 1 --use_wp_gru 0 --use_discrete_command 1 \
--use_tp 1 --continue_epoch 1 --cpu_cores 32 --num_repetitions 1 --use_disk_cache 1
