#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementations.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import signal
import sys
import random
import time

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.autoagents.agent_wrapper import AgentWrapper, AgentError
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.utils.result_writer import ResultOutputProvider


class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, run and stop a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. If needed, cleanup with manager.stop_scenario()
    """

    def __init__(self, timeout, debug_mode=False):
        """
        Setups up the parameters, which will be filled at load_scenario()
        """
        self.scenario = None
        self.scenario_tree = None
        self.scenario_class = None
        self.ego_vehicles = None
        self.other_actors = None

        self._debug_mode = debug_mode
        self._agent = None
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = float(timeout)

        # Used to detect if the simulation is down
        watchdog_timeout = max(5, self._timeout - 2)
        self._watchdog = Watchdog(watchdog_timeout)

        # Avoid the agent from freezing the simulation
        agent_timeout = watchdog_timeout - 1
        self._agent_watchdog = Watchdog(agent_timeout)

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None

        # Register the scenario tick as callback for the CARLA world
        # Use the callback_id inside the signal handler to allow external interrupts
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        self._running = False

    def cleanup(self):
        """
        Reset all parameters
        """
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None

    def load_scenario(self, scenario, agent, rep_number):
        """
        Load a new scenario
        """

        GameTime.restart()
        self._agent = AgentWrapper(agent)
        self.scenario_class = scenario
        self.scenario = scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors
        self.repetition_number = rep_number
        if self.scenario_class.agent.config.debug:
            import os
            import cv2
            fps = self.scenario_class.agent.config.video_fps*10
            root=os.path.join(os.environ.get("WORK_DIR"),"visualisation", "closed_loop",  self.scenario_class.agent.config.baseline_folder_name,
                    "debug", self.scenario_class.agent.config.eval_id)
            os.makedirs(root,exist_ok=True)
            self.width_video= self.scenario_class.agent.config.video_width_two_cam if self.scenario_class.agent.config.rear_cam else  self.scenario_class.agent.config.video_width_single_cam
            self.height_video= self.scenario_class.agent.config.video_height
            self.video_writer = cv2.VideoWriter(os.path.join(root,f"{self.scenario.name}.avi"),cv2.VideoWriter_fourcc(*'MJPG'),fps, (self.width_video,self.height_video))
        
        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

        self._agent.setup_sensors(self.ego_vehicles[0], self._debug_mode)

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True
        self._agent._agent.scenario_identifier= random.randint(10000, 99999)
        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                self._tick_scenario(timestamp)

    def _tick_scenario(self, timestamp):
        """
        Run next tick of scenario and the agent and tick the world.
        """

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()
            try:
                ego_action, replay_params, model, image = self._agent()
                self.replay_parameter=replay_params
                self.model=model
                if self.scenario_class.agent.config.debug:
                    import cv2
                    import numpy as np
                    if image is not None:
                        if not self.scenario_class.agent.config.rear_cam:
                            image = np.array(image.resize((self.width_video,self.height_video)))
                        else:
                            image = np.array(image.resize((self.width_video,self.height_video)))
                        self.video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)

            self.ego_vehicles[0].apply_control(ego_action)

            # Tick scenario
            self.scenario_tree.tick_once()

            if self._debug_mode:
                print("\n")
                py_trees.display.print_ascii_tree(self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

            spectator = CarlaDataProvider.get_world().get_spectator()
            ego_trans = self.ego_vehicles[0].get_transform()
            spectator.set_transform(
                carla.Transform(ego_trans.location + carla.Location(z=50), carla.Rotation(pitch=-90))
            )

        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)

    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        return self._watchdog.get_status()

    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        self._watchdog.stop()

        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

        if self.get_running_status():
            if self.scenario is not None:
                self.scenario.terminate()

            if self._agent is not None:
                self._agent.cleanup()
                self._agent = None

            self.analyze_scenario()

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        global_result = "\033[92m" + "SUCCESS" + "\033[0m"

        for criterion in self.scenario.get_criteria():
            if criterion.test_status != "SUCCESS":
                global_result = "\033[91m" + "FAILURE" + "\033[0m"

        if self.scenario.timeout_node.timeout:
            global_result = "\033[91m" + "FAILURE" + "\033[0m"

        ResultOutputProvider(self, global_result)

    def get_nocrash_diagnostics(self):
        duration = round(self.scenario_duration_game, 2)
        if self.scenario_duration_game > self.scenario.timeout:
            timeout = 1
        else:
            timeout = 0
        for criterion in self.scenario.get_criteria():
            actual_value = criterion.actual_value
            name = criterion.name

            if name == "RouteCompletionTest":
                route_completion = float(actual_value)
            if name == "OutsideRouteLanesTest":
                outside_route = float(actual_value)
            if name == "RunningStopTest":
                stops_ran = int(actual_value)
            if name == "InRouteTest":
                inroute = int(actual_value)
            if name == "RunningRedLightTest":
                lights_ran = int(actual_value)
            if name == "CollisionTest":
                collision=int(actual_value)

        return route_completion, outside_route,stops_ran,inroute,lights_ran, collision,duration, timeout
