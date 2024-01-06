import os
import csv
from pathlib import Path
from copy import deepcopy
from leaderboard.nocrash_evaluator import NoCrashEvaluator

class NoCrashEvalRunner():
    def __init__(self, args, town, weather, port=1000, tm_port=1002, debug=False):
        args = deepcopy(args)

        # Inject args
        args.scenario_class = 'nocrash_eval_scenario'
        args.port = port
        args.trafficManagerPort = tm_port
        args.debug = debug
        args.record = ''
        
        args.town = town
        args.weather = weather

        self.runner = NoCrashEvaluator(args, StatisticsManager(args))
        self.args = args

    def run(self):
        return self.runner.run(self.args)


class StatisticsManager:
    
    headers = [
        'town',
        'traffic',
        'weather',
        'start',
        'target',
        'route_completion',
        'lights_ran',
        'duration',
        'timeout',
        'collision'
    ]
    def __init__(self, args):
        
        self.finished_tasks = {
            'Town01': {},
            'Town02': {}
        }
        
        logdir = Path(args.log_path)
        filename=f"{args.eval_id}.csv"
        self.path_to_file=os.path.join(logdir, filename)
        if args.resume and os.path.exists(self.path_to_file):
            self.load(self.path_to_file)
            self.csv_file = open(self.path_to_file, 'a')
        else:
            logdir.mkdir(exist_ok=True)
            with open(self.path_to_file, 'w') as csv_file:
                self.csv_file=csv_file
                csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.headers)
                csv_writer.writeheader()

    def load(self, logdir):
        with open(logdir, 'r') as file:
            log = csv.DictReader(file)
            for row in log:
                self.finished_tasks[row['town']][(
                    int(row['traffic']),
                    int(row['weather']),
                    int(row['start']),
                    int(row['target']),
                )] = [
                    float(row['route_completion']),
                    int(row['lights_ran']),
                    float(row['duration']),
                ]
    
    def log(self, town, traffic, weather, start, target, route_completion, lights_ran, duration, timeout, collision):
        with open(self.path_to_file, "a") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=self.headers)
            csv_writer.writerow({
                'town'            : town,
                'traffic'         : traffic,
                'weather'         : weather,
                'start'           : start,
                'target'          : target,
                'route_completion': route_completion,
                'lights_ran'      : lights_ran,
                'duration'        : duration,
                'timeout': timeout,
                'collision': collision
            })

            csv_file.flush()
        
    def is_finished(self, town, route, weather, traffic):
        start, target = route
        key = (int(traffic), int(weather), int(start), int(target))
        return key in self.finished_tasks[town]
