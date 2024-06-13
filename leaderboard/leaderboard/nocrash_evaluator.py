#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""
from __future__ import print_function

import traceback
import argparse
import pickle
import itertools
from argparse import RawTextHelpFormatter
from datetime import datetime
from distutils.version import LooseVersion
import importlib
from coil_utils.baseline_helpers import merge_with_command_line_args
import os
import numpy as np
import pkg_resources
import pickle
import sys
import cv2
from coil_utils.baseline_helpers import visualize_model
import carla
import signal
from coil_utils.baseline_helpers import find_free_port
import torch
from coil_utils.baseline_helpers import merge_config
from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.scenarios.train_scenario import TrainScenario
from leaderboard.scenarios.nocrash_train_scenario import NoCrashTrainScenario
from leaderboard.scenarios.nocrash_eval_scenario import NoCrashEvalScenario
from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper import AgentWrapper, AgentError
from leaderboard.utils.route_indexer import RouteIndexer

# TODO whatch out there is the new statistics manager and not the original / nocrash one!
from leaderboard.utils.statistics_manager_local import StatisticsManager

sensors_to_icons = {
    "sensor.camera.rgb": "carla_camera",
    "sensor.lidar.ray_cast": "carla_lidar",
    "sensor.other.radar": "carla_radar",
    "sensor.other.gnss": "carla_gnss",
    "sensor.other.imu": "carla_imu",
    "sensor.opendrive_map": "carla_opendrive_map",
    "sensor.speedometer": "carla_speedometer",
    # Training sensors
    "sensor.map": "carla_map",
    "sensor.pretty_map": "carla_map",
    "sensor.collision": "carla_collision",
    "sensor.stitch_camera.rgb": "carla_stich_rgb",
    "sensor.stitch_camera.semantic_segmentation": "carla_stich_sem",
    "sensor.camera.semantic_segmentation": "carla_sem",
}


class NoCrashEvaluator(object):

    """
    TODO: document me!
    """

    ego_vehicles = []

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    wait_for_world = 20.0  # in seconds
    frame_rate = 20.0  # in Hz

    def __init__(self, args, statistics_manager):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        import os
        import torch.distributed as dist

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = find_free_port()
        dist.init_process_group(
            backend="nccl",
            init_method=f"env://127.0.0.1:{os.environ.get('MASTER_PORT')}",
            world_size=1,
            rank=0,
        )
        self.statistics_manager = statistics_manager
        self.sensors = None
        self.sensor_icons = []
        config_path=os.path.join(os.path.dirname(os.path.dirname(args.coil_checkpoint)), "config_training.pkl")
        with open(os.path.join(config_path), 'rb') as f:
            config = pickle.load(f)
        merge_with_command_line_args(config, args)
        if args.override_seq_len:
            setattr(config, "replay_seq_len", args.override_seq_len)
        self.config = config
        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        self.client = carla.Client(args.host, int(args.port))
        if args.timeout:
            self.client_timeout = float(args.timeout)
        self.client.set_timeout(self.client_timeout)

        self.traffic_manager = self.client.get_trafficmanager(int(args.trafficManagerPort))

        dist = pkg_resources.get_distribution("carla")
        if dist.version != "leaderboard":
            if LooseVersion(dist.version) < LooseVersion("0.9.10"):
                raise ImportError("CARLA version 0.9.10.1 or newer required. CARLA version found: {}".format(dist))

        # Load agent
        module_name = os.path.basename(args.agent).split(".")[0]
        sys.path.insert(0, os.path.dirname(args.agent))
        self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        self.manager = ScenarioManager(args.timeout, args.debug > 1)

        # Time control for summary purposes
        self._start_time = GameTime.get_time()
        self._end_time = None

        # Create the agent timer
        self._agent_watchdog = Watchdog(int(float(args.timeout)))
        signal.signal(signal.SIGINT, self._signal_handler)

        self.town = args.town

    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Timeout: Agent took too long to setup")
        elif self.manager:
            self.manager.signal_handler(signum, frame)

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """

        self._cleanup()
        if hasattr(self, "manager") and self.manager:
            del self.manager
        if hasattr(self, "world") and self.world:
            del self.world

    def _cleanup(self):
        """
        Remove and destroy all actors
        """

        # Simulation still running and in synchronous mode?
        if self.manager and self.manager.get_running_status() and hasattr(self, "world") and self.world:
            # Reset to asynchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(False)

        if self.manager:
            self.manager.cleanup()

        CarlaDataProvider.cleanup()

        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                self.ego_vehicles[i].destroy()
                self.ego_vehicles[i] = None
        self.ego_vehicles = []

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        if hasattr(self, "agent_instance") and self.agent_instance:
            self.agent_instance.destroy()
            self.agent_instance = None

        if hasattr(self, "statistics_manager") and self.statistics_manager:
            self.statistics_manager.scenario = None

    def _load_and_wait_for_world(self, args):
        """
        Load a new CARLA world and provide data to CarlaDataProvider
        """

        self.world = self.client.load_world(args.town)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        self.world.reset_all_traffic_lights()

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(int(args.trafficManagerPort))

        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(int(args.trafficManagerSeed))

        # Wait for the world to be ready
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

        if CarlaDataProvider.get_map().name != args.town:
            raise Exception(
                "The CARLA server uses the wrong map!" "This scenario requires to use map {}".format(args.town)
            )

    # def _register_statistics(self, config, checkpoint, entry_status, crash_message=""):
    #     """
    #     Computes and saved the simulation statistics
    #     """
    #     # register statistics
    #     current_stats_record = self.statistics_manager.compute_route_statistics(
    #         config,
    #         self.manager.scenario_duration_system,
    #         self.manager.scenario_duration_game,
    #         crash_message
    #     )

    #     print("\033[1m> Registering the route statistics\033[0m")
    #     self.statistics_manager.save_record(current_stats_record, config.index, checkpoint)
    #     self.statistics_manager.save_entry_status(entry_status, False, checkpoint)

    def _load_and_run_scenario(self, args, route, weather_idx, traffic_idx):
        """
        Load and run the scenario given by config.

        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.
        """
        crash_message = ""
        entry_status = "Started"

        start_idx, target_idx = route
        traffic_lvl = ["Empty", "Regular", "Dense"][traffic_idx]

        print(
            "\n\033[1m========= Preparing {} {}: {} to {}, weather {} =========".format(
                args.town, traffic_lvl, start_idx, target_idx, weather_idx
            )
        )
        print("> Setting up the agent\033[0m")

        # Set up the user's agent, and the timer to avoid freezing the simulation
        try:
            self._agent_watchdog.start()
            agent_class_name = getattr(self.module_agent, "get_entry_point")()
            if agent_class_name == "CoILAgent":
                loaded_checkpoint = torch.load(args.coil_checkpoint)

                self.agent_instance = getattr(self.module_agent, agent_class_name)(
                    config=self.config,
                    checkpoint=loaded_checkpoint,
                    city_name=self.town,
                    baseline=args.baseline_folder_name,
                )
            else:
                self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config)

            # Check and store the sensors
            if not self.sensors:
                self.sensors = self.agent_instance.sensors()
                track = self.agent_instance.track

                AgentWrapper.validate_sensor_configuration(self.sensors, track, args.track)

            self._agent_watchdog.stop()

        except SensorConfigurationInvalid as e:
            # The sensors are invalid -> set the ejecution to rejected and stop
            print("\n\033[91mThe sensor's configuration used is invalid:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent's sensors were invalid"
            entry_status = "Rejected"

            self._cleanup()
            sys.exit(-1)

        except Exception as e:
            # The agent setup has failed -> start the next route
            print("\n\033[91mCould not set up the required agent:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent couldn't be set up"

            self._cleanup()
            return

        print("\033[1m> Loading the world\033[0m")

        # Load the world and the scenario
        try:
            self._load_and_wait_for_world(args)
            scenario = NoCrashEvalScenario(
                world=self.world,
                agent=self.agent_instance,
                start_idx=start_idx,
                target_idx=target_idx,
                weather_idx=weather_idx,
                traffic_idx=traffic_idx,
                debug_mode=args.debug,
                config=self.config,
            )

            self.manager.load_scenario(scenario, self.agent_instance, 0)

        except Exception as e:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"
            entry_status = "Crashed"

            if args.record:
                self.client.stop_recorder()

            self._cleanup()
            sys.exit(-1)

        print("\033[1m> Running the route\033[0m")

        # Run the scenario
        try:
            self.manager.run_scenario()

        except AgentError as e:
            # The agent has failed -> stop the route
            print("\n\033[91mStopping the route, the agent has crashed:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()
            crash_message = "Agent crashed"

        except Exception as e:
            print("\n\033[91mError during the simulation:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"
            entry_status = "Crashed"

        # Stop the scenario
        try:
            print("\033[1m> Stopping the route\033[0m")
            self.manager.stop_scenario()

            (
                route_completion, outside_route,stops_ran,inroute,lights_ran, collision,duration, timeout_blocked
            ) = self.manager.get_nocrash_diagnostics()
            if self.config.debug:
                self.manager.video_writer.release()
            fail=False
            for criterion in self.manager.scenario_class.scenario.test_criteria:
                if criterion.test_status=="FAILURE" and criterion._terminate_on_failure:
                    fail=True
            if fail and (self.config.visualize_without_rgb or self.config.visualize_combined):
                observations,curr_pred,target_points, roads, pred_residual=self.manager.replay_parameter.values()
                if collision>0:
                    failure_case="collision"
                elif timeout_blocked>0:
                    failure_case="timeout_blocked"
                else:
                    failure_case="misc"
                root=os.path.join(os.environ.get("WORK_DIR"),"visualisation", "closed_loop", self.config.baseline_folder_name,
                                  failure_case,self.config.eval_id,self.manager.scenario_class.scenario.name)
                observations.reverse()
                curr_pred.reverse()
                target_points.reverse()
                roads.reverse()
                pred_residual.reverse()
                fps = self.config.fps_closed_loop_infractions
                os.makedirs(root,exist_ok=True)
                video_writer = cv2.VideoWriter(os.path.join(root,f"{self.manager.scenario.name}.avi"),cv2.VideoWriter_fourcc(*'MJPG'),fps, (512,1080))
                for iteration, (pred_i, target_point_i, roads_i, pred_residual_i) in enumerate(zip(curr_pred, target_points, roads,pred_residual)):
                    #to prevent the problem of not having a history
                    if iteration<self.config.max_img_seq_len_baselines:
                        continue
                    if iteration==0:
                        prev_wp=None
                    else:
                        prev_wp=curr_pred[iteration-1]["wp_predictions"].squeeze()
                    image_sequence=self.build_image_sequence(list(observations), iteration)
                    if self.config.detectboxes:
                        batch_of_bbs_pred=self.manager.model.module.convert_features_to_bb_metric(pred_i["pred_bb"])
                    else:
                        batch_of_bbs_pred=None
                    if not self.config.bev:
                        road=roads_i
                    else:
                        road=None
                    image=visualize_model(rgb=image_sequence,config=self.config,closed_loop=True,
                                          generate_video=True,
                                    save_path_root=root,
                                    target_point=target_point_i, pred_wp=pred_i["wp_predictions"].squeeze().detach().cpu().numpy(),
                                    pred_bb=batch_of_bbs_pred,step=np.round(-1/self.config.carla_fps*(len(observations)-iteration),2),
                                    pred_bev_semantic=pred_i["pred_bev_semantic"].squeeze().detach().cpu().numpy() if "pred_bev_semantic" in pred_i.keys() else None,
                                    road=road, parameters={"pred_residual": pred_residual_i}, pred_wp_prev=prev_wp, args=args)
                    
                    image = np.array(image.resize((512,1080)))
                    video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                   
                video_writer.release()
                
                with open(os.path.join(root, "predictions.pkl"), "wb") as file:
                    pickle.dump({"predictions":curr_pred}, file)
                
            self.statistics_manager.log(
                self.town,
                args.baseline_folder_name,
                args.eval_rep,
                self.config.training_repetition,
                self.config.eval_id,
                traffic_idx,
                weather_idx,
                start_idx,
                target_idx,
                route_completion,
                outside_route,
                stops_ran,
                inroute,
                lights_ran,
                collision,
                duration,
                timeout_blocked,
            )

            if args.record:
                self.client.stop_recorder()

            # Remove all actors
            scenario.remove_all_actors()

            self._cleanup()

        except Exception as e:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"

        if crash_message == "Simulation crashed":
            sys.exit(-1)
    def build_image_sequence(self,recorded_images, index):
        prev_images=recorded_images[index-self.config.max_img_seq_len_baselines:index]
        current_image=recorded_images[index]
        
        if self.config.img_seq_len<self.config.max_img_seq_len_baselines:
            empties=np.concatenate([np.zeros_like(recorded_images[0])]*(self.config.max_img_seq_len_baselines+1-self.config.img_seq_len), axis=1)
            image_sequence=np.concatenate([empties, current_image], axis=1)
        else:
            prev_images.append(current_image)
            image_sequence=np.concatenate(prev_images, axis=1)
        image_sequence=np.transpose(image_sequence,(1,2,0))*255
        return image_sequence
    def run(self, args):
        """
        Run the challenge mode
        """

        # if args.resume:
        # route_indexer.resume(args.checkpoint)
        # self.statistics_manager.resume(args.checkpoint)
        # else:
        # self.statistics_manager.clear_record(args.checkpoint)
        # route_indexer.save_state(args.checkpoint)

        # Load routes
        with open(args.route,
            "r",
        ) as f:
            routes = [tuple(map(int, l.split())) for l in f.readlines()]
        #weathers = {"train": [1, 6, 10, 14], "test": [3,8]}.get(args.weather)
        weathers = {"train": [14], "test": [8]}.get(args.weather)
        traffics = [0, 1, 2]
        for traffic, route, weather in itertools.product(traffics, routes, weathers):
            if self.statistics_manager.is_finished(args, route,weather, traffic):
                continue

            self._load_and_run_scenario(args, route, weather, traffic)

        # save global statistics
        print("\033[1m> Registering the global statistics\033[0m")
        # global_stats_record = self.statistics_manager.compute_global_statistics(route_indexer.total)
        # StatisticsManager.save_global_record(global_stats_record, self.sensor_icons, route_indexer.total, args.checkpoint)


def main():
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument("--host", default="localhost", help="IP of the host server (default: localhost)")
    parser.add_argument("--port", default="2000", help="TCP port to listen to (default: 2000)")
    parser.add_argument(
        "--trafficManagerPort",
        default="8000",
        help="Port to use for the TrafficManager (default: 8000)",
    )
    parser.add_argument(
        "--trafficManagerSeed",
        default="0",
        help="Seed used by the TrafficManager (default: 0)",
    )
    parser.add_argument("--debug", type=int, help="Run with debug output", default=0)
    parser.add_argument(
        "--record",
        type=str,
        default="",
        help="Use CARLA recording feature to create a recording of the scenario",
    )
    parser.add_argument(
        "--timeout",
        default="60.0",
        help="Set the CARLA client timeout value in seconds",
    )

    # simulation setup
    parser.add_argument(
        "--routes",
        help="Name of the route to be executed. Point to the route_xml_file to be executed.",
        required=True,
    )
    parser.add_argument(
        "--scenarios",
        help="Name of the scenario annotation file to be mixed with the route.",
        required=True,
    )
    parser.add_argument("--repetitions", type=int, default=1, help="Number of repetitions per route.")

    # agent-related options
    parser.add_argument(
        "-a",
        "--agent",
        type=str,
        help="Path to Agent's py file to evaluate",
        required=True,
    )
    parser.add_argument(
        "--agent-config",
        type=str,
        help="Path to Agent's configuration file",
        default="",
    )

    parser.add_argument("--track", type=str, default="SENSORS", help="Participation track: SENSORS, MAP")
    parser.add_argument(
        "--resume",
        type=bool,
        default=False,
        help="Resume execution from last checkpoint?",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./simulation_results.json",
        help="Path to checkpoint used for saving statistics and resuming",
    )
    parser.add_argument(
        "--override_seq_len",
        type=int,
        )

    arguments = parser.parse_args()

    statistics_manager = StatisticsManager()

    try:
        leaderboard_evaluator = NoCrashEvaluator(arguments, statistics_manager)
        leaderboard_evaluator.run(arguments)

    except Exception as e:
        traceback.print_exc()
    finally:
        del leaderboard_evaluator


if __name__ == "__main__":
    main()
