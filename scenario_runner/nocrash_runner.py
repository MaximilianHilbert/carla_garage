import os
import csv
from pathlib import Path
from copy import deepcopy
from leaderboard.nocrash_evaluator import NoCrashEvaluator
from coil_utils.baseline_helpers import get_ablations_dict

class NoCrashEvalRunner:
    def __init__(self, args, config,town, weather, port=1000, tm_port=1002, debug=False):
        args = deepcopy(args)

        # Inject args
        args.scenario_class = "nocrash_eval_scenario"
        args.port = port
        args.trafficManagerPort = tm_port
        args.debug = debug
        args.record = ""

        args.town = town
        args.weather = weather

        self.runner = NoCrashEvaluator(args, StatisticsManager(args, config))
        self.args = args

    def run(self):
        return self.runner.run(self.args)


class StatisticsManager:
    def __init__(self, args, config):
        self.finished_tasks = {"Town01": {}, "Town02": {}}
        ablations_dict=get_ablations_dict()
        self.ablations_dict={}
        for ablation in ablations_dict.keys():
            try:
                self.ablations_dict.update({ablation:getattr(config, ablation)})
            except AttributeError:
                self.ablations_dict.update({ablation:ablations_dict[ablation]})
        self.headers = [
        *ablations_dict,
        "eval_rep",
        "training_rep",
        "experiment",
        "town",
        "baseline",
        "traffic",
        "weather",
        "start",
        "target",
        "route_completion",
        "outside_route",
        "stops_ran",
        "inroute",
        "lights_ran",
        "collision",
        "duration",
        "timeout_blocked",
]
        logdir = Path(args.log_path)
        filename = f"{config.baseline_folder_name}_{args.eval_id}.csv"
        self.path_to_file = os.path.join(logdir, filename)
        if args.resume and os.path.exists(self.path_to_file):
            self.load(self.path_to_file)
            self.csv_file = open(self.path_to_file, "a")
        else:
            logdir.mkdir(exist_ok=True)
            with open(self.path_to_file, "w") as csv_file:
                self.csv_file = csv_file
                csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.headers)
                csv_writer.writeheader()

    def load(self, logdir):
        with open(logdir, "r") as file:
            log = csv.DictReader(file)
            for row in log:
                self.finished_tasks[row["town"]][
                    (
                        str(row["baseline"]),
                        int(row["bev"]),
                        int(row["detectboxes"]),
                        int(row["speed"]),
                        int(row["prevnum"]),
                        str(row["backbone"]),
                        str(row["lossweights"]),
                        int(row["traffic"]),
                        int(row["eval_rep"]),
                        int(row["training_rep"]),
                        str(row["experiment"]),
                        int(row["weather"]),
                        int(row["start"]),
                        int(row["target"]),
                    )
                ] = [
                    float(row["route_completion"]),
                    float(row["outside_route"]),
                    int(row["stops_ran"]),
                    float(row["inroute"]),
                    int(row["lights_ran"]),
                    int(row["collision"]),
                    float(row["duration"]),
                    float(row["timeout_blocked"])
                ]

    def log(
        self,
        town,
        baseline,
        eval_rep,
        training_rep,
        experiment,
        traffic,
        weather,
        start,
        target,
        route_completion,
        outside_route,
        stops_ran,
        inroute,
        lights_ran,
        collision,
        duration,
        timeout_blocked
    ):
        with open(self.path_to_file, "a") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=self.headers)
            csv_writer.writerow(
                {
                    "baseline": baseline,
                    **self.ablations_dict,
                    "eval_rep": eval_rep,
                    "training_rep": training_rep,
                    "experiment": experiment,
                    "town": town,
                    "traffic": traffic,
                    "weather": weather,
                    "start": start,
                    "target": target,
                    "route_completion": route_completion,
                    "outside_route": outside_route,
                    "stops_ran": stops_ran,
                    "inroute": inroute,
                    "lights_ran": lights_ran,
                    "collision": collision,
                    "duration": duration,
                    "timeout_blocked": timeout_blocked,
                }
            )

            csv_file.flush()

    def is_finished(self, args, route,weather, traffic):
        start, target = route
        key = (str(args.baseline_folder_name), str(args.eval_id), int(traffic), int(weather), int(start), int(target))
        return key in self.finished_tasks[args.town]
