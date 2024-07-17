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
from coil_utils.baseline_helpers import visualize_model
from argparse import RawTextHelpFormatter
from datetime import datetime
import cv2
import numpy as np
import csv
import pickle
from distutils.version import LooseVersion
import importlib
import os
import pkg_resources
import sys
import carla
import signal
import numpy.random as random

from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.scenarios.scenario_manager_local import ScenarioManager
from leaderboard.scenarios.route_scenario_local import RouteScenario
from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper_local import AgentWrapper, AgentError
from leaderboard.utils.statistics_manager_local import StatisticsManager
from leaderboard.utils.route_indexer import RouteIndexer

import pathlib

sensors_to_icons = {
    "sensor.camera.rgb": "carla_camera",
    "sensor.lidar.ray_cast": "carla_lidar",
    "sensor.other.radar": "carla_radar",
    "sensor.other.gnss": "carla_gnss",
    "sensor.other.imu": "carla_imu",
    "sensor.opendrive_map": "carla_opendrive_map",
    "sensor.speedometer": "carla_speedometer",
    "sensor.stitch_camera.rgb": "carla_camera",  # for local World on Rails evaluation
    "sensor.camera.semantic_segmentation": "carla_camera",  # for datagen
    "sensor.camera.depth": "carla_camera",  # for datagen
}


class LeaderboardEvaluator(object):

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
        self.statistics_manager = statistics_manager
        self.sensors = None
        self.sensor_icons = []
        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam

        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        self.client = carla.Client(args.host, int(args.port))
        if args.timeout:
            self.client_timeout = float(args.timeout)
        self.client.set_timeout(self.client_timeout)

        try:
            self.world = self.client.load_world("Town01")
        except RuntimeError:
            # For cases where load_world crashes, but the world was properly
            # loaded anyways
            self.world = self.client.get_world()
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

    def _cleanup(self, results=None):
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
            self.agent_instance.destroy(results)
            del self.agent_instance

        if hasattr(self, "statistics_manager") and self.statistics_manager:
            self.statistics_manager.scenario = None

    def _prepare_ego_vehicles(self, ego_vehicles, wait_for_ego_vehicles=False):
        """
        Spawn or update the ego vehicles
        """

        if not wait_for_ego_vehicles:
            for vehicle in ego_vehicles:
                self.ego_vehicles.append(
                    CarlaDataProvider.request_new_actor(
                        vehicle.model,
                        vehicle.transform,
                        vehicle.rolename,
                        color=vehicle.color,
                        vehicle_category=vehicle.category,
                    )
                )

        else:
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in ego_vehicles:
                    ego_vehicle_found = False
                    carla_vehicles = CarlaDataProvider.get_world().get_actors().filter("vehicle.*")
                    for carla_vehicle in carla_vehicles:
                        if carla_vehicle.attributes["role_name"] == ego_vehicle.rolename:
                            ego_vehicle_found = True
                            self.ego_vehicles.append(carla_vehicle)
                            break
                    if not ego_vehicle_found:
                        ego_vehicle_missing = True
                        break

            for i, _ in enumerate(self.ego_vehicles):
                self.ego_vehicles[i].set_transform(ego_vehicles[i].transform)

        # sync state
        CarlaDataProvider.get_world().tick()

    def _load_and_wait_for_world(self, args, town, ego_vehicles=None):
        """
        Load a new CARLA world and provide data to CarlaDataProvider
        """
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

        if CarlaDataProvider.get_map().name != town:
            raise Exception("The CARLA server uses the wrong map!" "This scenario requires to use map {}".format(town))

    def _register_statistics(self, config, route_date_string, checkpoint, entry_status, crash_message=""):
        """
        Computes and saved the simulation statistics
        """
        # register statistics
        current_stats_record = self.statistics_manager.compute_route_statistics(
            config,
            route_date_string,
            self.manager.scenario_duration_system,
            self.manager.scenario_duration_game,
            crash_message,
        )

        print("\033[1m> Registering the route statistics\033[0m")
        self.statistics_manager.save_record(current_stats_record, config.index, checkpoint)
        self.statistics_manager.save_entry_status(entry_status, False, checkpoint)

        return current_stats_record

    def _load_and_run_scenario(self, args, config):
        """
        Load and run the scenario given by config.

        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.
        """
        crash_message = ""
        entry_status = "Started"

        print(
            "\n\033[1m========= Preparing {} (repetition {}) =========".format(config.name, config.repetition_index),
            flush=True,
        )
        print("> Setting up the agent\033[0m")

        # Prepare the statistics of the route
        self.statistics_manager.set_route(config.name, config.index)
        # Randomize during data collection.
        # Deterministic seed during evaluation.
        if int(os.environ.get("DATAGEN", 0)) == 1:
            CarlaDataProvider._rng = random.RandomState(seed=None)
        else:
            CarlaDataProvider._rng = random.RandomState(seed=config.index)

        now = datetime.now()
        route_string = pathlib.Path(os.environ.get("ROUTES", "")).stem + "_"
        route_string += f"route{config.index}"
        route_date_string = (
            route_string
            + "_"
            + "_".join(
                map(
                    lambda x: "%02d" % x,
                    (now.month, now.day, now.hour, now.minute, now.second),
                )
            )
        )

        # Set up the user's agent, and the timer to avoid freezing the simulation
        try:
            self._agent_watchdog.start()
            agent_class_name = getattr(self.module_agent, "get_entry_point")()
            if int(os.environ.get("DATAGEN", 0)) == 1:
                self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config, config.index)
            else:
                self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config,route_date_string,args)
            config.agent = self.agent_instance

            # Check and store the sensors
            if not self.sensors:
                self.sensors = self.agent_instance.sensors()
                track = self.agent_instance.track

                AgentWrapper.validate_sensor_configuration(self.sensors, track, args.track)

                self.sensor_icons = [sensors_to_icons[sensor["type"]] for sensor in self.sensors]
                self.statistics_manager.save_sensors(self.sensor_icons, args.checkpoint)

            self._agent_watchdog.stop()

        except SensorConfigurationInvalid as e:
            # The sensors are invalid -> set the ejecution to rejected and stop
            print("\n\033[91mThe sensor's configuration used is invalid:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent's sensors were invalid"
            entry_status = "Rejected"

            result = self._register_statistics(config, route_date_string, args.checkpoint, entry_status, crash_message)
            self._cleanup(result)
            sys.exit(-1)

        except Exception as e:
            # The agent setup has failed -> start the next route
            print("\n\033[91mCould not set up the required agent:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent couldn't be set up"

            result = self._register_statistics(config, route_date_string, args.checkpoint, entry_status, crash_message)
            self._cleanup(result)
            return

        print("\033[1m> Loading the world\033[0m")

        # Load the world and the scenario
        try:
            self.world = self.client.load_world(config.town)
        except RuntimeError:
            # For cases where load_world crashes, but the world was properly
            # loaded anyways
            self.world = self.client.get_world()
        try:
            self._load_and_wait_for_world(args, config.town, config.ego_vehicles)
            self._prepare_ego_vehicles(config.ego_vehicles, False)
            scenario = RouteScenario(world=self.world, config=config, debug_mode=args.debug)
            self.statistics_manager.set_scenario(scenario.scenario)

            # Night mode
            if config.weather.sun_altitude_angle < 0.0:
                for vehicle in scenario.ego_vehicles:
                    vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))

            # Load scenario and run it
            if args.record:
                self.client.start_recorder("{}/{}_rep{}.log".format(args.record, config.name, config.repetition_index))
            self.manager.load_scenario(scenario, self.agent_instance, config.repetition_index)

        except Exception as e:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"
            entry_status = "Crashed"

            result = self._register_statistics(config, route_date_string, args.checkpoint, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            self._cleanup(result)
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
            result = self._register_statistics(config, route_date_string, args.checkpoint, entry_status, crash_message)
            collision_ped, collision_veh, collision_lay, red_light, stop_infraction, outside_route, route_dev,timeout,blocked=result.infractions.values()
            save_path_timing=os.path.join(os.environ.get("WORK_DIR"),"visualisation", "closed_loop", self.manager.scenario_class.config.agent.config.baseline_folder_name,
                                  self.manager.scenario_class.config.agent.config.experiment_id,self.manager.scenario_class.scenario.name)
            os.makedirs(save_path_timing,exist_ok=True)
            with open(os.path.join(save_path_timing,"inference_time.csv"), mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=["forward_time_in_s"])
                writer.writeheader()
                writer.writerow({"forward_time_in_s": list(self.manager.replay_parameters.values())[-1]})
            if self.manager.scenario_class.config.agent.config.debug:
                self.manager.video_writer.release()
            # fail=False
            # for criterion in self.manager.scenario_class.scenario.test_criteria:
            #     if criterion.test_status=="FAILURE" and criterion._terminate_on_failure:
            #         fail=True
            if self.manager.scenario_class.config.agent.config.visualize_without_rgb or self.manager.scenario_class.config.agent.config.visualize_combined:
                observations,curr_pred,target_points, roads, pred_residual,_=self.manager.replay_parameters.values()
                if collision_ped or collision_veh or collision_lay:
                    failure_case="collision"
                elif red_light:
                    failure_case="red_light"
                elif stop_infraction:
                    failure_case="stop_infraction"
                elif outside_route:
                    failure_case="outside_route"
                elif route_dev:
                    failure_case="route_dev"
                elif timeout:
                    failure_case="timeout"
                elif blocked:
                    failure_case="blocked"
                else:
                    failure_case="misc"
                root=os.path.join(os.environ.get("WORK_DIR"),"visualisation", "closed_loop", self.manager.scenario_class.config.agent.config.baseline_folder_name,
                                  failure_case,self.manager.scenario_class.config.agent.config.experiment_id,self.manager.scenario_class.scenario.name)
                observations.reverse()
                curr_pred.reverse()
                target_points.reverse()
                roads.reverse()
                pred_residual.reverse()
                fps = 5
                os.makedirs(root,exist_ok=True)
                video_writer = cv2.VideoWriter(os.path.join(root,f"{self.manager.scenario.name}.avi"),cv2.VideoWriter_fourcc(*'MJPG'),fps, (512,1080))
                for iteration, (pred_i, target_point_i, roads_i, pred_residual_i) in enumerate(zip(curr_pred, target_points, roads,pred_residual)):
                    #to prevent the problem of not having a history
                    if iteration<self.manager.scenario_class.config.agent.config.considered_images_incl_current:
                        continue
                    if iteration==0:
                        prev_wp=None
                    else:
                        prev_wp=curr_pred[iteration-1]["wp_predictions"].squeeze()
                    image_sequence=self.build_image_sequence(list(observations), iteration)
                    if self.manager.scenario_class.config.agent.config.detectboxes:
                        batch_of_bbs_pred=self.manager.model.module.convert_features_to_bb_metric(pred_i["pred_bb"])
                    else:
                        batch_of_bbs_pred=None
                    if not self.manager.scenario_class.config.agent.config.bev:
                        road=roads_i
                    else:
                        road=None
                    image=visualize_model(rgb=image_sequence,config=self.manager.scenario_class.config.agent.config,closed_loop=True,
                                          generate_video=True,
                                    save_path_root=root,
                                    target_point=target_point_i, pred_wp=pred_i["wp_predictions"].squeeze().detach().cpu().numpy(),
                                    pred_bb=batch_of_bbs_pred,step=np.round(-1/self.manager.scenario_class.config.agent.config.carla_fps*(len(observations)-iteration),2),
                                    pred_bev_semantic=pred_i["pred_bev_semantic"].squeeze().detach().cpu().numpy() if "pred_bev_semantic" in pred_i.keys() else None,
                                    road=road, parameters={"pred_residual": pred_residual_i}, pred_wp_prev=prev_wp, args=args)
                    
                    image = np.array(image.resize((512,1080)))
                    video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                   
                video_writer.release()
                
                with open(os.path.join(root, "predictions.pkl"), "wb") as file:
                    pickle.dump({"predictions":curr_pred}, file)
            
            
            
            
            if args.record:
                self.client.stop_recorder()

            # Remove all actors
            scenario.remove_all_actors()

            self._cleanup(result)

        except Exception as e:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"

        if crash_message == "Simulation crashed":
            sys.exit(-1)
    def build_image_sequence(self,recorded_images, index):
        prev_images=recorded_images[index-self.manager.scenario_class.config.agent.config.considered_images_incl_current:index]
        current_image=recorded_images[index]
        
        if self.manager.scenario_class.config.agent.config.img_seq_len<self.manager.scenario_class.config.agent.config.considered_images_incl_current:
            empties=np.concatenate([np.zeros_like(recorded_images[0])]*(self.manager.scenario_class.config.agent.config.considered_images_incl_current+1-self.manager.scenario_class.config.agent.config.img_seq_len), axis=1)
            image_sequence=np.concatenate([empties, current_image], axis=1)
        else:
            prev_images.append(current_image)
            image_sequence=np.concatenate(prev_images, axis=1)
        image_sequence=np.transpose(image_sequence,(1,2,0))
        return image_sequence
    def run(self, args):
        """
        Run the challenge mode
        """
        route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)

        if args.resume:
            route_indexer.resume(args.checkpoint)
            self.statistics_manager.resume(args.checkpoint)
        else:
            self.statistics_manager.clear_record(args.checkpoint)
            route_indexer.save_state(args.checkpoint)

        while route_indexer.peek():
            print("Starting new route.", flush=True)
            # setup
            config = route_indexer.next()
            print("Load and run scenarios.", flush=True)

            # run
            self._load_and_run_scenario(args, config)

            print("Save state.", flush=True)
            route_indexer.save_state(args.checkpoint)

        # save global statistics
        print("\033[1m> Registering the global statistics\033[0m")
        global_stats_record = self.statistics_manager.compute_global_statistics(route_indexer.total)
        StatisticsManager.save_global_record(
            global_stats_record, self.sensor_icons, route_indexer.total, args.checkpoint
        )


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
    #baseline required arguments
    parser.add_argument("--repetitions", type=int, default=1, help="Number of repetitions per route.")
    parser.add_argument("--visualize-combined",type=int,default=0)
    parser.add_argument("--visualize-without-rgb",type=int,default=0)

    
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
        "--experiment-id",
        type=str
    )
    arguments = parser.parse_args()

    statistics_manager = StatisticsManager()

    try:
        leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager)
        leaderboard_evaluator.run(arguments)

    except Exception as e:
        traceback.print_exc()
    finally:
        del leaderboard_evaluator


if __name__ == "__main__":
    main()
