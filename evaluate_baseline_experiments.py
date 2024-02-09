import os
import subprocess
def traffic_seeds():
    return iter([
        28756, 98102, 54329, 87541, 12345, 67890, 45678, 87654, 23451, 76543, 90321, 34567, 87612, 54378, 98765, 87654, 12345, 54321, 87654, 23456
    ])
def get_latest_checkpoint(dir):
    list_of_checkpoints=os.listdir(dir)
    numbers=[int(filename.replace(".pth", "")) for filename in list_of_checkpoints]
    return str(max(numbers))+".pth"
    

def parse_models():
    checkpoint_path_dict={}
    logs_root=os.path.join(os.environ.get("WORK_DIR"), "_logs")
    for baseline in os.listdir(logs_root):
        baseline_root=os.path.join(logs_root, baseline)
        checkpoint_path_dict[baseline]={}
        for experiment in os.listdir(baseline_root):
            experiment_root=os.path.join(baseline_root, experiment)
            checkpoint_path_dict[baseline][experiment]={}
            for repetition in os.listdir(experiment_root):
                repetition_root=os.path.join(experiment_root, repetition)
                checkpoint=get_latest_checkpoint(os.path.join(repetition_root, "checkpoints"))
                checkpoint_root=os.path.join(repetition_root, "checkpoints", checkpoint)
                checkpoint_path_dict[baseline][experiment][repetition]=checkpoint_root
    return checkpoint_path_dict

def main():
    checkpoint_path_dict=parse_models()
    all_evaluation_arguments=[]
    for baseline, experiment_dict in checkpoint_path_dict.items():
        for experiment, repetition_dict in experiment_dict.items():
            for repetition, checkpoint in repetition_dict.items():
                for weather in ["train", "test"]:
                    for town in ["Town01", "Town02"]:
                        print(baseline, experiment, repetition, checkpoint)
                        arguments=[
                            "--agent-yaml",f'{os.path.join(os.environ.get("CONFIG_ROOT"), baseline, experiment, ".yaml")}',
                            "--baseline-name", experiment,
                            "--baseline-folder-name", baseline,
                            "--coil_checkpoint", checkpoint,
                            "--eval_id", f"{baseline}_{experiment}_{repetition}_{weather}_{town}",
                            "--log_path",f"{os.path.join(os.path.dirname(os.path.dirname(checkpoint)), 'evaluation')}",
                            "--town", town,
                            "--weather", weather,
                            "--trafficManagerSeed", str(next(traffic_seeds()))
                        ]
                        all_evaluation_arguments.append(arguments)
    print(f"Number of runs to be evaluated: {len(all_evaluation_arguments)}")
    for run_args in all_evaluation_arguments:
        env = os.environ.copy()
        process=subprocess.Popen(
            f'conda run -n garage python {os.path.join(os.environ.get("WORK_DIR"), "evaluate_nocrash_baselines.py")} {" ".join(run_args)}',stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=env
        )
        process.wait()
        stdout, sterr=process.communicate()
        print(stdout.decode())
if __name__=="__main__":

    main()