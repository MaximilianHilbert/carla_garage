"""
Evaluates a driving model on a set of CARLA routes wherein each route is evaluated on a separate machine in parallel.
This script generates the necessary shell files to run this on a SLURM cluster.
It also monitors the evaluation and resubmits crashed routes.
At the end all results files are aggregated and parsed.
Best run inside a tmux terminal.
"""

import subprocess
import time
from pathlib import Path
import os
import fnmatch
import xml.etree.ElementTree as ET
import ujson
from tqdm import tqdm
import sys
import numpy as np
import pandas as pd

# Our centOS is missing some c libraries.
# Usually miniconda has them, so we tell the linker to look there as well.
# newlib = '/mnt/lustre/work/geiger/gwb629/conda/garage/lib'
# if not newlib in os.environ['LD_LIBRARY_PATH']:
#   os.environ['LD_LIBRARY_PATH'] += ':' + newlib


def create_run_eval_bash(
    work_dir,
    route,
    town,
    weather,
    setting,
    seed,
    eval_filename,
    baseline,
    model_dir,
    bash_save_dir,
    results_save_dir,
    carla_tm_port_start,
    carla_root,
    eval_rep,
    benchmark,
    experiment
):
    Path(f"{results_save_dir}").mkdir(parents=True, exist_ok=True)
    with open(f"{bash_save_dir}/eval_{eval_filename}.sh", "w", encoding="utf-8") as rsh:
        rsh.write(
f"""
export WORK_DIR={work_dir}
export CONFIG_ROOT=$WORK_DIR/coil_configuration
export TEAM_CODE=$WORK_DIR/team_code
export COIL_NETWORK=$WORK_DIR/coil_network


export CARLA_ROOT={carla_root}
export CARLA_SERVER=${{CARLA_ROOT}}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${{CARLA_ROOT}}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${{CARLA_ROOT}}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${{CARLA_ROOT}}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=$WORK_DIR/scenario_runner
export LEADERBOARD_ROOT=$WORK_DIR/leaderboard
export PYTHONPATH="${{CARLA_ROOT}}/PythonAPI/carla/":"${{SCENARIO_RUNNER_ROOT}}":"${{LEADERBOARD_ROOT}}":${{PYTHONPATH}}
export PYTHONPATH=$PYTHONPATH:$CONFIG_ROOT
export PYTHONPATH=$PYTHONPATH:$COIL_NETWORK
export PYTHONPATH=$PYTHONPATH:$TEAM_CODE
export PYTHONPATH=$PYTHONPATH:$WORK_DIR
"""
        )
        rsh.write(
f"""
export PORT=$1
echo 'World Port:' $PORT
export TM_PORT=`comm -23 <(seq {carla_tm_port_start} {carla_tm_port_start+49} | sort) <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
echo 'TM Port:' $TM_PORT
export SCENARIOS=leaderboard/data/scenarios/eval_scenarios.json
export BASELINE={baseline}
export COIL_CHECKPOINT={model_dir}
export EVAL_ID={eval_filename}
export ADDITIONAL_LOG_PATH={results_save_dir}
export TOWN={town}
export WEATHER={weather}
export SEED={seed}
export CHALLENGE_TRACK_CODENAME=SENSORS
export REPETITION={eval_rep}
export RESUME=1
export SETTING={setting}
export ROUTE={route}
export BENCHMARK={benchmark}
""")
        if args.cluster=="tcml":
            rsh.write(
f"""
source /home/hilbert/.bashrc
eval "$(conda shell.bash hook)"
conda activate garage
""")
        else:
            rsh.write(
f""" 
source ~/.bashrc
conda activate /mnt/lustre/work/geiger/gwb629/conda/garage
""")
        rsh.write(
"""
python3 ${WORK_DIR}/evaluate_nocrash_baselines.py \
--baseline=${BASELINE} \
--coil_checkpoint=${COIL_CHECKPOINT} \
--eval_id=${EVAL_ID} \
--log_path=${ADDITIONAL_LOG_PATH} \
--town=${TOWN} \
--weather=${WEATHER} \
--trafficManagerSeed=${SEED} \
--port=${PORT} \
--tm_port=${TM_PORT} \
--timeout=600 \
--resume=${RESUME} \
--eval_rep=${REPETITION} \
--setting=${SETTING} \
--route=${ROUTE} \
--visualize-combined=1
""" if benchmark=="nocrash" else f"""python3 $WORK_DIR/leaderboard/leaderboard/leaderboard_evaluator_local.py \
--track=SENSORS \
--agent=$WORK_DIR/team_code/coil_agent.py \
--visualize-without-rgb=0 \
--visualize-combined=1 \
--experiment-id={experiment} \
--agent-config={os.path.dirname(os.path.dirname(model_dir))} \
--routes={route} \
--scenarios=$WORK_DIR/leaderboard/data/scenarios/eval_scenarios.json \
--resume=true \
--timeout=60 \
--trafficManagerPort={carla_tm_port_start} \
--trafficManagerSeed={seed}
""")


def make_jobsub_file(root,commands, exp_name, exp_root_name, filename, partition):
    os.makedirs(os.path.join(root, 'evaluation', exp_root_name,exp_name,'run_files','logs'), exist_ok=True)
    os.makedirs(os.path.join(root, 'evaluation', exp_root_name,exp_name,'run_files','job_files'), exist_ok=True)
    job_file = os.path.join(root, "evaluation",exp_root_name,exp_name,"run_files", "job_files",f"{filename}.sh")
    qsub_template = f"""#!/bin/bash
#SBATCH --job-name={filename}
#SBATCH --partition={partition}
#SBATCH -o {os.path.join(root, "evaluation", exp_root_name,exp_name,"run_files","logs",f"qsub_out_{filename}.log")}
#SBATCH -e {os.path.join(root, "evaluation",exp_root_name,exp_name,"run_files","logs",f"qsub_err_{filename}.log")}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40gb
#SBATCH --time=00-10:00
#SBATCH --gres=gpu:1
"""
    for cmd in commands:
        qsub_template = (
            qsub_template
            + f"""
{cmd}

"""
        )

    with open(job_file, "w", encoding="utf-8") as f:
        f.write(qsub_template)
    return job_file


def get_num_jobs(job_name, username):
    len_usrn = len(username)
    num_running_jobs = int(
        subprocess.check_output(
            f"SQUEUE_FORMAT2='username:{len_usrn},name:130' squeue --sort V | grep {username} | grep {job_name} | wc -l",
            shell=True,
        )
        .decode("utf-8")
        .replace("\n", "")
    )
    with open("max_num_jobs.txt", "r", encoding="utf-8") as f:
        max_num_parallel_jobs = int(f.read())

    return num_running_jobs, max_num_parallel_jobs


def main(args):
    single_test = False
    settings_to_be_tested = ["02_withheld"]  # only set when single test is True
    training_reps_to_be_tested = ["repetition_0"]  # only set when single test is True

    towns = ["Town01", "Town02"]

    #weathers = {"train": [1, 6, 10, 14], "test": [3,8]}
    weathers = {"train": [14], "test": [8]}
    weather_conditions=["train"]
    traffics_len = 1
    if args.cluster=="tcml":
        partition = "day"
        username = "hilbert"
        code_root="/home/hilbert/carla_garage"
    else:
        partition = "2080-galvani"
        username = "gwb629"
        code_root="/home/maximilian/Master/carla_garage"
    epochs = ["30"]
    seeds = [234213, 252534, 290246]
    num_repetitions = 3
    benchmark = args.benchmark
    model_dir = os.path.join(code_root, "_logs")
    carla_root = os.path.join(code_root, "carla")

    job_nr = 0
    already_placed_files = {}
    meta_jobs = {}
    experiment_name_stem = f"{benchmark}"
    for baseline in os.listdir(model_dir):
        if baseline=="waypoint_weight_generation":
            continue
        for experiment in tqdm(os.listdir(os.path.join(model_dir, baseline))):
            if not os.path.isdir(os.path.join(model_dir, baseline, experiment)):
                continue
            for repetition in (
                tqdm(os.listdir(os.path.join(model_dir, baseline, experiment)))
                if not single_test
                else training_reps_to_be_tested
            ):  # training repetition
                for setting in (
                    os.listdir(os.path.join(model_dir, baseline, experiment, repetition))
                    if not single_test
                    else settings_to_be_tested
                ):
                    checkpoints = os.listdir(
                        os.path.join(
                            model_dir,
                            baseline,
                            experiment,
                            repetition,
                            setting,
                            "checkpoints",
                        )
                    )
                    for epoch in epochs:
                        checkpoint_file = f"{epoch}.pth"
                        if checkpoint_file not in checkpoints:
                            print(f"Not finished checkpoint file for experiment {experiment}")
                            continue
                        else:
                            for weather in weather_conditions:
                              for town in towns:
                                  for evaluation_repetition, seed in zip(
                                      range(1, num_repetitions + 1), seeds
                                  ):  # evaluation repetition
                                      expected_result_length = 0
                                      
                                      exp_names_tmp = []
                                      exp_names_tmp.append(experiment_name_stem + f"_e{evaluation_repetition}")
                                      if benchmark=="nocrash":  
                                        route_path = f"{code_root}/leaderboard/data/{benchmark}_split/{town}"
                                      else:
                                        route_path = f"{code_root}/leaderboard/data/{benchmark}_split/"
                                      if benchmark=="nocrash":
                                        route_pattern = "*.txt"
                                      else:
                                        route_pattern = "*.xml"

                                      carla_world_port_start = 10000
                                      carla_streaming_port_start = 20000
                                      carla_tm_port_start = 30000

                                      # Root folder in which each of the evaluation seeds will be stored
                                      experiment_name_root = experiment_name_stem + "_" + epoch
                                      exp_names = []
                                      for name in exp_names_tmp:
                                          exp_names.append(name + "_" + epoch)

                                      checkpoint = experiment
                                      checkpoint_new_name = checkpoint + "_" + epoch

                                      # Links the model file into team_code
                                      copy_model = False

                                      if copy_model:
                                          # copy checkpoint to my folder
                                          cmd = f"mkdir team_code/checkpoints/{checkpoint_new_name}"
                                          print(cmd)
                                          os.system(cmd)
                                          cmd = f"cp {model_dir}/{checkpoint}/config.pickle team_code/checkpoints/{checkpoint_new_name}/"
                                          print(cmd)
                                          os.system(cmd)
                                          cmd = f"ln -sf {model_dir}/{checkpoint}/{epoch}.pth team_code/checkpoints/{checkpoint_new_name}/model.pth"
                                          print(cmd)
                                          os.system(cmd)

                                      route_files = []
                                      expected_result_lengths=[]
                                      for root, _, files in os.walk(route_path):
                                        for name in files:
                                            if fnmatch.fnmatch(name, route_pattern):
                                                if benchmark=="nocrash":
                                                    with open(os.path.join(root, name)) as route_split_file:
                                                        expected_result_lengths.append(len(route_split_file.readlines())*len(weathers[weather]) * traffics_len)
                                                else:
                                                    expected_result_lengths.append(len(ET.parse(os.path.join(root, name)).findall('.//waypoint')))
                                                    route_files.append(os.path.join(root, name))

                                      for exp_name in exp_names:
                                        for route in route_files:
                                          bash_save_dir = Path(
                                              os.path.join(
                                                  code_root,
                                                  "evaluation",
                                                  experiment_name_root,
                                                  exp_name,
                                                  "run_bashs",
                                              )
                                          )
                                          results_save_dir = Path(
                                              os.path.join(
                                                  code_root,
                                                  "evaluation",
                                                  experiment_name_root,
                                                  exp_name,
                                                  "results",
                                              )
                                          )
                                          logs_save_dir = Path(
                                              os.path.join(
                                                  code_root,
                                                  "evaluation",
                                                  experiment_name_root,
                                                  exp_name,
                                                  "logs",
                                              )
                                          )
                                          bash_save_dir.mkdir(parents=True, exist_ok=True)
                                          results_save_dir.mkdir(parents=True, exist_ok=True)
                                          logs_save_dir.mkdir(parents=True, exist_ok=True)

                                      for exp_name in exp_names:
                                        for route,expected_result_length in zip(route_files,expected_result_lengths):
                                            route_stem = Path(route).stem
                                          
                                              
                                            eval_filename = (baseline+"_"+
                                            experiment_name_stem
                                          + f"_e-{experiment}_w-{weather}_t-{town}_r-{evaluation_repetition}_t-{repetition}_s-{setting}_o-{route_stem}")
                                          
                                            bash_save_dir = Path(
                                                os.path.join(
                                                    code_root,
                                                    "evaluation",
                                                    experiment_name_root,
                                                    exp_name,
                                                    "run_bashs",
                                                )
                                            )
                                            results_save_dir = Path(
                                                os.path.join(
                                                    code_root,
                                                    "evaluation",
                                                    experiment_name_root,
                                                    exp_name,
                                                    "results",
                                                )
                                            )
                                            logs_save_dir = Path(
                                                os.path.join(
                                                    code_root,
                                                    "evaluation",
                                                    experiment_name_root,
                                                    exp_name,
                                                    "logs",
                                                )
                                            )
                                          
                                            result_file = f"{results_save_dir}/{eval_filename}.csv"
                                            if benchmark=="nocrash":
                                                if os.path.exists(result_file):
                                                    with open(result_file) as file:
                                                        length=len(file.readlines()[1:])
                                                    if length == expected_result_length:
                                                        print("Found existing finished resultfile, skipping...")
                                                        continue
                                            commands = []

                                            # Finds a free port
                                            commands.append(
                                                f"FREE_WORLD_PORT=`comm -23 <(seq {carla_world_port_start} {carla_world_port_start + 49} | sort) "
                                                f"<(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`"
                                            )
                                            commands.append("echo 'World Port:' $FREE_WORLD_PORT")
                                            commands.append(
                                                f"FREE_STREAMING_PORT=`comm -23 <(seq {carla_streaming_port_start} {carla_streaming_port_start + 49} "
                                                f"| sort) <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`"
                                            )
                                            commands.append("echo 'Streaming Port:' $FREE_STREAMING_PORT")
                                            commands.append(
                                                f"SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 {carla_root}/CarlaUE4.sh "
                                                f"-carla-rpc-port=${{FREE_WORLD_PORT}} -nosound -carla-streaming-port=${{FREE_STREAMING_PORT}} -opengl &"
                                            )
                                            commands.append("sleep 180")  # Waits for CARLA to finish starting
                                            current_model = os.path.join(
                                                model_dir,
                                                baseline,
                                                experiment,
                                                repetition,
                                                setting,
                                                "checkpoints",
                                                checkpoint_file,
                                            )
                                            create_run_eval_bash(
                                                code_root,
                                                route,
                                                town,
                                                weather,
                                                setting,
                                                seed,
                                                eval_filename,
                                                baseline,
                                                current_model,
                                                bash_save_dir,
                                                results_save_dir,
                                                carla_tm_port_start,
                                                carla_root,
                                                evaluation_repetition,
                                                benchmark,
                                                experiment
                                            )
                                            commands.append(f"chmod u+x {bash_save_dir}/eval_{eval_filename}.sh")
                                            commands.append(
                                                f"{bash_save_dir}/eval_{eval_filename}.sh $FREE_WORLD_PORT"
                                            )
                                            commands.append("sleep 2")

                                            carla_world_port_start += 50
                                            carla_streaming_port_start += 50
                                            carla_tm_port_start += 50

                                            job_file = make_jobsub_file(
                                                root=code_root,
                                                commands=commands,
                                                exp_name=experiment_name_stem,
                                                exp_root_name=experiment_name_root,
                                                filename=eval_filename,
                                                partition=partition,
                                            )
                                            

                                            # Wait until submitting new jobs that the #jobs are at below max
                                            (
                                                num_running_jobs,
                                                max_num_parallel_jobs,
                                            ) = get_num_jobs(
                                                job_name=experiment_name_stem,
                                                username=username,
                                            )
                                            print(f"{num_running_jobs}/{max_num_parallel_jobs} jobs are running...")
                                            while num_running_jobs >= max_num_parallel_jobs:
                                                (
                                                    num_running_jobs,
                                                    max_num_parallel_jobs,
                                                ) = get_num_jobs(
                                                    job_name=experiment_name_stem,
                                                    username=username,
                                                )
                                            time.sleep(0.05)
                                            if eval_filename not in already_placed_files.keys():
                                                print(f"Submitting job {job_nr}: {job_file}")

                                                jobid = (
                                                    subprocess.check_output(
                                                        f"sbatch {job_file}",
                                                        shell=True,
                                                    )
                                                    .decode("utf-8")
                                                    .strip()
                                                    .rsplit(" ", maxsplit=1)[-1]
                                                )
                                                meta_jobs[jobid] = (
                                                    False,
                                                    job_file,
                                                    expected_result_length,
                                                    result_file,
                                                    0,
                                                )
                                                already_placed_files[eval_filename] = job_file
                                                job_nr += 1

    if benchmark=="nocrash":
        training_finished = False
        while not training_finished:
            num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=experiment_name_stem, username=username)
            print(f"{num_running_jobs} jobs are running...")
            time.sleep(10)

            # resubmit unfinished jobs
            for k in list(meta_jobs.keys()):
                (
                    job_finished,
                    job_file,
                    expected_result_length,
                    result_file,
                    resubmitted,
                ) = meta_jobs[k]
                need_to_resubmit = False
                if not job_finished and resubmitted < 5:
                    # check whether job is running
                    if int(subprocess.check_output(f"squeue | grep {k} | wc -l", shell=True).decode("utf-8").strip()) == 0:
                        # check whether result file is finished?
                        if os.path.exists(result_file):
                            print("file exists")
                            with open(result_file, "r", encoding="utf-8") as f_result:
                                evaluation_data_lines = len(f_result.readlines()[1:])
                                if evaluation_data_lines != expected_result_length:
                                    need_to_resubmit = True
                            if not need_to_resubmit:
                                # delete old job
                                print(f"Finished job {job_file}")
                                meta_jobs[k] = (True, None, None, None, 0)
                        else:
                            need_to_resubmit = True

                if need_to_resubmit:
                    # print("Remove file: ", result_file)
                    # Path(result_file).unlink()
                    print(f"resubmit sbatch {job_file}")
                    jobid = (
                        subprocess.check_output(f"sbatch {job_file}", shell=True)
                        .decode("utf-8")
                        .strip()
                        .rsplit(" ", maxsplit=1)[-1]
                    )
                    meta_jobs[jobid] = (
                        False,
                        job_file,
                        expected_result_length,
                        result_file,
                        resubmitted + 1,
                    )
                    meta_jobs[k] = (True, None, None, None, 0)
                    num_running_jobs += 1
            time.sleep(10)

            if num_running_jobs == 0:
                training_finished = True
    else:
        training_finished = False
        while not training_finished:
            num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=experiment_name_stem, username=username)
            print(f"{num_running_jobs} jobs are running...")
            time.sleep(10)

            # resubmit unfinished jobs
            for k in list(meta_jobs.keys()):
                job_finished, job_file, result_file, resubmitted = meta_jobs[k]
                need_to_resubmit = False
                if not job_finished and resubmitted < 5:
                    # check whether job is running
                    if int(subprocess.check_output(f"squeue | grep {k} | wc -l", shell=True).decode("utf-8").strip()) == 0:
                        # check whether result file is finished?
                        if os.path.exists(result_file):
                            with open(result_file, "r", encoding="utf-8") as f_result:
                                evaluation_data = ujson.load(f_result)
                            progress = evaluation_data["_checkpoint"]["progress"]

                            if len(progress) < 2 or progress[0] < progress[1]:
                                need_to_resubmit = True
                            else:
                                for record in evaluation_data["_checkpoint"]["records"]:
                                    if record["status"] == "Failed - Agent couldn't be set up":
                                        need_to_resubmit = True
                                        print("Resubmit - Agent not setup")
                                    elif record["status"] == "Failed":
                                        need_to_resubmit = True
                                    elif record["status"] == "Failed - Simulation crashed":
                                        need_to_resubmit = True
                                    elif record["status"] == "Failed - Agent crashed":
                                        need_to_resubmit = True

                            if not need_to_resubmit:
                                # delete old job
                                print(f"Finished job {job_file}")
                                meta_jobs[k] = (True, None, None, 0)
                        else:
                            need_to_resubmit = True

                if need_to_resubmit:
                    # Remove crashed results file
                    if os.path.exists(result_file):
                        print("Remove file: ", result_file)
                        Path(result_file).unlink()
                    print(f"resubmit sbatch {job_file}")
                    jobid = (
                        subprocess.check_output(f"sbatch {job_file}", shell=True)
                        .decode("utf-8")
                        .strip()
                        .rsplit(" ", maxsplit=1)[-1]
                    )
                    meta_jobs[jobid] = (False, job_file, result_file, resubmitted + 1)
                    meta_jobs[k] = (True, None, None, 0)

            time.sleep(10)
    print("Evaluation finished. Start parsing results.")
    # eval_root = f'{code_root}/evaluation'
    # subprocess.check_call(
    #     f'python {code_root}/tools/result_parser.py --xml {code_root}/leaderboard/data/{benchmark}.xml '
    #     f'--results {eval_root} --log_dir {eval_root} --town_maps {code_root}/leaderboard/data/town_maps_xodr '
    #     f'--map_dir {code_root}/leaderboard/data/town_maps_tga --device cpu '
    #     f'--map_data_folder {code_root}/tools/proxy_simulator/map_data --subsample 1 --strict --visualize_infractions',
    #     stdout=sys.stdout,
    #     stderr=sys.stderr,
    #     shell=True)


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster",
        type=str,
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="longest6"
    )
    args = parser.parse_args()
    main(args)