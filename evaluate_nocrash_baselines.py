from nocrash_runner import NoCrashEvalRunner
from coil_utils.baseline_helpers import merge_with_command_line_args
import pickle
def main(args):
    town = args.town
    weather = args.weather
    debug=args.debug
    port = args.port
    config_path=os.path.join(os.path.dirname(os.path.dirname(args.coil_checkpoint)), "config_training.pkl")
    with open(os.path.join(config_path), 'rb') as f:
        config = pickle.load(f)
    merge_with_command_line_args(config, args)
    runner = NoCrashEvalRunner(args, config,town, weather, port=port, tm_port=args.tm_port, debug=debug)
    runner.run()


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()

    # Agent configs
    parser.add_argument(
        "--agent",
        default=f"{os.path.join(os.environ.get('TEAM_CODE'), 'coil_agent.py')}",
    )
    parser.add_argument("--visualize-combined",type=int,default=0)
    parser.add_argument("--visualize-without-rgb",type=int,default=0)
    parser.add_argument("--norm",default=2, type=int)
    

    # Benchmark configs
    parser.add_argument("--town", required=True, choices=["Town01", "Town02"])
    parser.add_argument("--weather", required=True, choices=["train", "test"])

    parser.add_argument("--host", default="localhost", help="IP of the host server (default: localhost)")
    parser.add_argument(
        "--trafficManagerSeed",
        default="0",
        help="Seed used by the TrafficManager (default: 0)",
    )
    parser.add_argument(
        "--timeout",
        default="600.0",
        help="Set the CARLA client timeout value in seconds",
    )

    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--route", type=str, help="full path to route file on disk")
    parser.add_argument("--tm_port", type=int, default=2002)
    parser.add_argument("--repetitions", type=int, default=1, help="Number of repetitions per route.")
    parser.add_argument("--eval_rep", type=int, default=1, help="Number of repetitions per route.")
    parser.add_argument("--track", type=str, default="SENSORS", help="Participation track: SENSORS, MAP")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./simulation_results.json",
        help="Path to checkpoint used for saving statistics and resuming",
    )
    parser.add_argument(
        "--eval_id",
        default="id_000",
        help="Set this to be your evaluation id for the name of the log file",
    )
    parser.add_argument(
        "--log_path",
        required=True,
        help="Set this to be your path to the log file for evaluation",
    )
    parser.add_argument(
        "--override_seq_len",
        type=int,
        )
    parser.add_argument(
        "--coil_checkpoint",
        help="Set this to be your path to the previously recorded checkpoint file of the coiltraine framework",
    )
    parser.add_argument("--baseline-folder-name", help="either arp bcoh bcso keyframes")
    parser.add_argument("--setting", help="coil or all")
    parser.add_argument("--debug", type=int, help="Run with debug output", default=0)

    args = parser.parse_args()

    main(args)
