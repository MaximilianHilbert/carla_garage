import numpy as np
import scipy
import sys
import os
import glob
import torch
import cv2
import random
import time
import cv2
from collections import deque
from coil_configuration.coil_config import g_conf
import matplotlib.pyplot as plt
from leaderboard.envs.sensor_interface import SensorInterface
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from leaderboard.autoagents.autonomous_agent import Track
from srunner.scenariomanager.timer import GameTime
from coil_network.coil_model import CoILModel
from coil_planner.planner import Planner
from team_code.transfuser_utils import preprocess_compass, inverse_conversion_2d
import carla
from team_code.coil_train import merge_config_files

def get_entry_point():
  return 'CoILAgent'

class CoILAgent(AutonomousAgent):
    #TODO check for double image input in original_image_list; change from priviledged input to gps/velocity dont forget to correctly transform to new ego vehicle 
    #TODO Before retraining, adapt to modern logging 
    def __init__(self, checkpoint, city_name="Town01", baseline_folder_name="arp", baseline_name="arp_vanilla", carla_version='0.9'):
        # Set the carla version that is going to be used by the interface
        self._carla_version = carla_version
        self.checkpoint = checkpoint  # We save the checkpoint for some interesting future use.
        self.config=merge_config_files(baseline_folder_name, baseline_name)
        if baseline_folder_name in ["bcoh", "bcso"]:
            self.model=CoILModel("coil-icra", g_conf.MODEL_CONFIGURATION)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.cuda()
            self.model.eval()
        else:
            self._policy = CoILModel("coil-policy", g_conf.MODEL_CONFIGURATION)
            self._mem_extract = CoILModel("coil-memory", g_conf.MEM_EXTRACT_MODEL_CONFIGURATION)
            self._policy.load_state_dict(checkpoint['policy_state_dict'])
            self._mem_extract.load_state_dict(checkpoint['mem_extract_state_dict'])
            self._mem_extract.cuda()
            self._mem_extract.eval()
            self._policy.cuda()
            self._policy.eval()
        
        self.first_iter = True
        #self.rgb_queue=deque(maxlen=g_conf.NUMBER_FRAMES_FUSION)
        self.rgb_queue=deque(maxlen=self.config.img_seq_len-1)
        self.previous_actions=deque(maxlen=self.config.number_previous_actions)

        #TODO watch out, this is the old planner from coiltraine!
        self._planner=Planner(city_name)
        #Carla 0.9 related attributes
        self.track=Track.SENSORS
        self._global_plan = None
        self._global_plan_world_coord = None
        self.sensor_interface = SensorInterface()
        self.wallclock_t0 = None
        #carla 0.9.10 gps to world_coordinates related transformations
        self.mean = np.array([0.0, 0.0])
        self.scale = np.array([111324.60662786, 111319.490945])
        # Load the model and prepare set it for evaluation
        self.latest_image = None
        self.latest_image_tensor = None
        self.target_point_prev=0
    def sensors(self):
        return [{'type': 'sensor.camera.rgb', 'x': self.config.camera_pos[0], 'y': self.config.camera_pos[1], 'z': self.config.camera_pos[2], 'roll': self.config.camera_rot_0[0], 'pitch': self.config.camera_rot_0[1], 'yaw': self.config.camera_rot_0[2],
                      'width': self.config.camera_width, 'height': self.config.camera_height, 'fov': self.config.camera_fov, 'id': 'CentralRGB', 'sensor_tick': self.config.carla_fps},
                       {
        'type': 'sensor.other.imu',
        'x': 0.0,
        'y': 0.0,
        'z': 0.0,
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': 0.0,
        'sensor_tick': self.config.carla_fps,
        'id': 'imu'
    },
    ]
    def __call__(self):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        
        self.vehicle = CarlaDataProvider.get_hero_actor()
        current_data = self.sensor_interface.get_data()
        timestamp = GameTime.get_time()

        if not self.wallclock_t0:
            self.wallclock_t0 = GameTime.get_wallclocktime()
        wallclock = GameTime.get_wallclocktime()
        wallclock_diff = (wallclock - self.wallclock_t0).total_seconds()
        if timestamp<1:
            control = carla.VehicleControl()
            control.steer=0.
            control.brake=0.
            control.throttle=0.
            current_image=current_data.get("CentralRGB")[1][...,:3]
            current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
            past_actions=[0.,0.,0.]
        else:
            control, current_image = self.run_step(current_data, list(self.rgb_queue), timestamp)
        control.manual_gear_shift = False
        past_actions=(control.steer, control.throttle, control.brake)
        self.rgb_queue.append(current_image)
        self.previous_actions.extend(past_actions)
        return control
    def yaw_to_orientation(self, yaw):
        # Calculate the orientation vector in old carla convention
        x=np.cos(-yaw)
        y=np.sin(yaw)
        z=0
        return x, y, z
    def run_step(self, sensor_data,original_image_list,timestamp,avoid_stop=True, perturb_speed=False):
        """
            Run a step on the benchmark simulation
        Args:
            measurements: The measurements
            sensor_data: The sensor data
            original_image_list: All the original images used on this benchmark, the input format is a list, including a series of continous frames.
            processed_image_list: All the processed images, the input format is a list, including a series of continous frames.
            directions: The directions, high level commands
            target: Final objective. Not used when the agent is predicting all outputs.
            previous_actions_list: All the previous actions used on this benchmark, optional

        Returns:
            Controls for the vehicle on the CARLA simulator.

        """
        #retrieve location data from sensors and normalize/transform to ego vehicle system
        measurements=sensor_data.get("imu")
        current_location=self.vehicle.get_location()
        current_location=np.array([current_location.x, current_location.y])
        waypoint_route=self._route_planner.run_step(current_location)
        if len(waypoint_route) > 2:
            target_point_location, end_point_yaw,high_level_command = waypoint_route[1]
        elif len(waypoint_route) > 1:
            target_point_location, end_point_yaw,high_level_command = waypoint_route[1]
        else:
            target_point_location, end_point_yaw,high_level_command = waypoint_route[0]
        
        if (target_point_location != self.target_point_prev).all():
            self.target_point_prev=target_point_location
        
        current_yaw=measurements[1][-1]
        current_yaw_ego_system=preprocess_compass(current_yaw)
        current_orientation_ego_system=np.array([*self.yaw_to_orientation(current_yaw_ego_system)])

        #do the same for the end_point position/orientation
        end_point_yaw_ego_system=preprocess_compass(end_point_yaw)
        end_point_orientation_ego_system=np.array([*self.yaw_to_orientation(end_point_yaw_ego_system)])
        end_point_location_ego_system=inverse_conversion_2d(target_point_location, current_location, current_yaw_ego_system)

        #Conversion to old convention necessary in carla >=0.9, only take BGR Values without alpha channel and convert to RGB for the model
        current_image=sensor_data.get("CentralRGB")[1][...,:3]
        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)

       
        directions = self._get_directions(current_location, current_orientation_ego_system, target_point_location, end_point_orientation_ego_system)
        
        
        velocity_vector=self.vehicle.get_velocity()
        # Take the forward speed and normalize it for it to go from 0-1
        norm_speed=np.sqrt(np.square(velocity_vector.x)+np.square(velocity_vector.y))/g_conf.SPEED_FACTOR
        if perturb_speed and norm_speed < 0.01:
            norm_speed += random.uniform(0.05, 0.15)
        norm_speed = torch.cuda.FloatTensor([norm_speed]).unsqueeze(0)
        if self.config.speed_input:
            measurement_input = norm_speed
        else:
            measurement_input = torch.zeros_like(norm_speed)
        directions_tensor = torch.cuda.LongTensor([directions])
        single_image, observation_history = self._process_sensors(current_image, original_image_list)
    
        if self.config.baseline_folder_name=="arp":
            _, memory = self._mem_extract(torch.unsqueeze(observation_history,0))
            model_outputs = self._policy.forward_branch(torch.unsqueeze(single_image,0), measurement_input, directions_tensor, memory)

            predicted_speed = self._policy.extract_predicted_speed().cpu().detach().numpy()
        else:
            if self.config.baseline_folder_name=="bcoh":
                merged_history_and_current=torch.cat([single_image, observation_history], dim=0)
                if self.config.train_with_actions_as_input:
                    model_outputs = self.model.forward_branch(torch.unsqueeze(merged_history_and_current,0), measurement_input,
                                                        directions_tensor,torch.from_numpy(np.array(self.previous_actions).astype(np.float)).type(torch.FloatTensor).unsqueeze(0).cuda())
                else:
                    model_outputs = self.model.forward_branch(torch.unsqueeze(merged_history_and_current,0), measurement_input,
                                                        directions_tensor)
            else:
                if self.config.train_with_actions_as_input:
                    model_outputs = self.model.forward_branch(torch.unsqueeze(single_image,0), measurement_input,directions_tensor,
                                                              torch.from_numpy(np.array(self.previous_actions).astype(np.float)).type(torch.FloatTensor).unsqueeze(0).cuda())
                else:
                    model_outputs = self.model.forward_branch(torch.unsqueeze(single_image,0), measurement_input,
                                                        directions_tensor)
            predicted_speed = self.model.extract_predicted_speed().cpu().detach().numpy()
        
        steer, throttle, brake = self._process_model_outputs(model_outputs[0], norm_speed, predicted_speed, avoid_stop)
        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle=float(throttle)
        control.brake = float(brake)
    
        # There is the posibility to replace some of the predictions with oracle predictions.
        if g_conf.USE_ORACLE:
            _, control.throttle, control.brake = self._get_oracle_prediction(
                measurements, target)
        self.first_iter = False

       
        print(steer, throttle, brake, directions)
        print("target")
        print(target_point_location)
        print("current location")
        print(current_location)
        return control, current_image

    def get_attentions(self, layers=None):
        """

        Returns
            The activations obtained from the first layers of the latest iteration.

        """
        if layers is None:
            layers = [0, 1, 2]
        if self.latest_image_tensor is None:
            raise ValueError('No step was ran yet. '
                             'No image to compute the activations, Try Running ')
        all_layers = self._model.get_perception_layers(self.latest_image_tensor)
        cmap = plt.get_cmap('inferno')
        attentions = []
        for layer in layers:
            y = all_layers[layer]
            att = torch.abs(y).mean(1)[0].data.cpu().numpy()
            att = att / att.max()
            att = cmap(att)
            att = np.delete(att, 3, 2)
            attentions.append(cv2.resize(att, [200, 88]))
        return attentions
    def _get_directions(self, current_location, current_orientation_ego_system, end_point_location_ego_system, end_point_orientation_ego_system):
        """
        Class that should return the directions to reach a certain goal
        """
        directions = self._planner.get_next_command(
            (current_location[0],
             current_location[1], 0.22),
            (current_orientation_ego_system[0],
             current_orientation_ego_system[1],
             current_orientation_ego_system[2]),
            (end_point_location_ego_system[0], end_point_location_ego_system[1], 0.22),
            (end_point_orientation_ego_system[0], end_point_orientation_ego_system[1], end_point_orientation_ego_system[2]))
        return directions
    def _process_single_frame(self, current_image):
        self.latest_image = current_image
        current_image = np.swapaxes(current_image, 0, 1)
        current_image = np.transpose(current_image, (2, 1, 0))
        current_image = torch.from_numpy(current_image / 255.0).type(torch.FloatTensor).cuda()
        return current_image


    def _process_sensors(self, current_image, image_list):
        processed_images=[]
        current_image=self._process_single_frame(current_image)
        for image in image_list:
            processed_image=self._process_single_frame(image)
            processed_images.append(processed_image)
        if processed_images:
            multi_image_input=torch.cat(processed_images, dim=0)
        else:
            multi_image_input=torch.empty((1,1))
        self.latest_image_tensor=multi_image_input
        return current_image, multi_image_input

    def _process_model_outputs(self, outputs, norm_speed, predicted_speed, avoid_stop=True):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """
        assert len(g_conf.TARGETS) == len(outputs), 'the dimension of outputs does not match the TARGETS!'
        if len(g_conf.TARGETS) == 3:
            steer, throttle, brake = float(outputs[0]), float(outputs[1]), float(outputs[2])
            if brake < 0.05:
                brake = 0.0

            if throttle > brake:
                brake = 0.0
        elif len(g_conf.TARGETS) == 2:
            steer, throttle_brake = float(outputs[0]), float(outputs[1])
            if throttle_brake >= 0:
                throttle = throttle_brake
                brake = 0
            else:
                throttle = 0
                brake = -throttle_brake
        else:
            raise Exception('only support 2 or 3 dimensional outputs')

        if avoid_stop:
            real_speed = norm_speed * g_conf.SPEED_FACTOR
            real_predicted_speed = predicted_speed * g_conf.SPEED_FACTOR

            if real_speed < 5.0 and real_predicted_speed > 6.0:  # If (Car Stooped) and ( It should not have stoped)
                throttle += 20.0 / g_conf.SPEED_FACTOR - norm_speed
                brake = 0.0

        return steer, throttle, brake


    def _process_model_outputs_wp(self, outputs):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """
        # with waypoint
        wpa1, wpa2, throttle, brake = outputs[3], outputs[4], outputs[1], outputs[2]
        if brake < 0.2:
            brake = 0.0

        if throttle > brake:
            brake = 0.0

        steer = 0.7 * wpa2

        if steer > 0:
            steer = min(steer, 1)
        else:
            steer = max(steer, -1)

        return steer, throttle, brake

    def _get_oracle_prediction(self, measurements, target):
        # For the oracle, the current version of sensor data is not really relevant.
        control, _, _, _, _ = self.control_agent.run_step(measurements, [], [], target)

        return control.steer, control.throttle, control.brake
