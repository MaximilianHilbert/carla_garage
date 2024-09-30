import numpy as np
import os
import torch
import csv
import cv2
import cv2
from pytictoc import TicToc
import torch.distributed as dist
import pickle
import math
from collections import deque
from torch.nn.parallel import DistributedDataParallel as DDP
from coil_utils.baseline_helpers import merge_with_command_line_args
from coil_utils.baseline_helpers import get_ablations_dict
import matplotlib.pyplot as plt
from team_code.transfuser_utils import non_maximum_suppression
from leaderboard.envs.sensor_interface import SensorInterface
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from leaderboard.autoagents.autonomous_agent import Track
from coil_utils.copycat_helper import norm
from coil_utils.baseline_helpers import visualize_model
from nav_planner import RoutePlanner
from srunner.scenariomanager.timer import GameTime
from team_code.timefuser_model import TimeFuser
from team_code.transfuser_utils import preprocess_compass, inverse_conversion_2d
import team_code.transfuser_utils as t_u
import carla
from team_code.transfuser_utils import PIDController
from coil_utils.baseline_helpers import find_free_port
def get_entry_point():
    return "CoILAgent"


class CoILAgent(AutonomousAgent):
    def __init__(self,config_path=None, route_index=None,args=None,config=None,checkpoint=None, baseline=None, nocrash=False):
        self.setup(config=config,config_path=config_path,route_index=route_index,args=args,checkpoint=checkpoint,baseline=baseline,nocrash=nocrash)
    def setup(self, args=None,config_path=None, baseline=None,route_index=None, nocrash=False, config=None, checkpoint=None):
        if not nocrash:
            with open(os.path.join(config_path, "config_training.pkl"), 'rb') as f:
                config = pickle.load(f)
            merge_with_command_line_args(config, args)
            for ablation, value in get_ablations_dict().items():
                if ablation not in config.__dict__:
                    setattr(config, ablation, value)
            checkpoint = torch.load(os.path.join(config_path, "checkpoints", "31.pth"))
        self.config = config
        self.initialized = False
    
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = find_free_port()
        self.speed_histogram = []
        if not torch.distributed.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method=f"env://127.0.0.1:{os.environ.get('MASTER_PORT')}",
                world_size=1,
                rank=0,
            )
        if config.baseline_folder_name in ["bcoh", "bcso", "keyframes"]:
            model = TimeFuser(config.baseline_folder_name, self.config, training=False)
            model.to("cuda:0")
            self.model = DDP(model, device_ids=["cuda:0"])
            self.model.load_state_dict(checkpoint["state_dict"],strict=False)
            self.model.cuda()
            self.model.eval()
        else:
            policy = TimeFuser("arp-policy", self.config, training=False)
            policy.to("cuda:0")
            self._policy = DDP(policy, device_ids=["cuda:0"])

            mem_extract = TimeFuser("arp-memory", self.config, training=False)
            mem_extract.to("cuda:0")
            self._mem_extract = DDP(mem_extract, device_ids=["cuda:0"])
            self._policy.load_state_dict(checkpoint["policy_state_dict"])
            self._mem_extract.load_state_dict(checkpoint["mem_extract_state_dict"])
            self._mem_extract.eval()
            self._policy.eval()
        if self.config.tf_pp_rep:
            self.turn_controller_direct = PIDController(
            k_p=config.turn_kp, k_i=config.turn_ki, k_d=config.turn_kd, n=config.turn_n
        )
            self.speed_controller_direct = PIDController(
                k_p=config.speed_kp,
                k_i=config.speed_ki,
                k_d=config.speed_kd,
                n=config.speed_n,
            )
        else:

            self.turn_controller = PIDController(
                k_p=config.turn_kp, k_i=config.turn_ki, k_d=config.turn_kd, n=config.turn_n
            )
            self.speed_controller = PIDController(
                k_p=config.speed_kp,
                k_i=config.speed_ki,
                k_d=config.speed_kd,
                n=config.speed_n,
            )
        #TODO remove after configs are correctly trained
        self.config.replay_seq_len=100
        self.first_iter = True
        self.prev_rgb_queue = deque(maxlen=self.config.img_seq_len*self.config.data_save_freq*self.config.sampling_rate)
        self.prev_speeds_queue = deque(maxlen=self.config.data_save_freq*self.config.sampling_rate) #we fuse the previous and the current timestep velocity
        self.prev_location_queue=deque(maxlen=self.config.considered_images_incl_current*self.config.data_save_freq*self.config.sampling_rate) # we fuse only the previous 6 timesteps locations
        #queues for replay simulation
        self.replay_image_queue=deque(maxlen=self.config.replay_seq_len*self.config.data_save_freq*self.config.sampling_rate)
        self.replay_current_predictions_queue=deque(maxlen=self.config.replay_seq_len*self.config.data_save_freq*self.config.sampling_rate)
        self.replay_target_points_queue=deque(maxlen=self.config.replay_seq_len*self.config.data_save_freq*self.config.sampling_rate)
        self.replay_road_queue=deque(maxlen=self.config.replay_seq_len*self.config.data_save_freq*self.config.sampling_rate)
        self.replay_pred_residual_queue=deque(maxlen=self.config.replay_seq_len*self.config.data_save_freq*self.config.sampling_rate)
        self.track = Track.SENSORS
        self._global_plan = None
        self._global_plan_world_coord = None
        self.sensor_interface = SensorInterface()
        self.wallclock_t0 = None
        # carla 0.9.10 gps to world_coordinates related transformations
        self.mean = np.array([0.0, 0.0])
        self.scale = np.array([111324.60662786, 111319.490945])
        # Load the model and prepare set it for evaluation
        self.latest_image = None
        self.latest_image_tensor = None
        self.target_point_prev = 0
    def destroy(self, results=None):  # pylint: disable=locally-disabled, unused-argument
        """
        Gets called after a route finished.
        The leaderboard client doesn't properly clear up the agent after the route finishes so we need to do it here.
        Also writes logging files to disk.
        """

        if self.config.baseline_folder_name == "arp":
            del self._policy
            del self._mem_extract
        else:
            del self.model
    def init_visualization(self):
        # Privileged map access for visualization
        from birds_eye_view.chauffeurnet import (
            ObsManager,
        )  # pylint: disable=locally-disabled, import-outside-toplevel
        from srunner.scenariomanager.carla_data_provider import (
            CarlaDataProvider,
        )  # pylint: disable=locally-disabled, import-outside-toplevel

        obs_config = {
            "width_in_pixels": self.config.lidar_resolution_width * 4,
            "pixels_ev_to_bottom": self.config.lidar_resolution_height / 2.0 * 4,
            "pixels_per_meter": self.config.pixels_per_meter * 4,
            "history_idx": [-1],
            "scale_bbox": True,
            "scale_mask_col": 1.0,
            "map_folder": "maps_high_res",
        }
        self._vehicle = CarlaDataProvider.get_hero_actor()
        self.ss_bev_manager = ObsManager(obs_config, self.config)
        self.ss_bev_manager.attach_ego_vehicle(self._vehicle, criteria_stop=None)
    def sensors(self):
        sensors_lst=[
            {
                "type": "sensor.camera.rgb",
                "x": self.config.camera_pos[0],
                "y": self.config.camera_pos[1],
                "z": self.config.camera_pos[2],
                "roll": self.config.camera_rot_0[0],
                "pitch": self.config.camera_rot_0[1],
                "yaw": self.config.camera_rot_0[2],
                "width": self.config.camera_width,
                "height": self.config.camera_height,
                "fov": self.config.camera_fov,
                "id": "CentralRGB",
                "sensor_tick": self.config.carla_fps,
            },
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": self.config.carla_fps,
                "id": "imu",
            },
                        {
                "type": "sensor.other.gnss",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.01,
                "id": "gps",
            },
            {
                "type": "sensor.speedometer",
                "reading_frequency": self.config.carla_fps,
                "id": "speed",
            },
        ]
        if hasattr(self.config, "rear_cam"):
            if self.config.rear_cam:
                sensors_lst.append(    {
                        "type": "sensor.camera.rgb",
                        "x": self.config.camera_pos_rear[0],
                        "y": self.config.camera_pos_rear[1],
                        "z": self.config.camera_pos_rear[2],
                        "roll": self.config.camera_rot_0_rear[0],
                        "pitch": self.config.camera_rot_0_rear[1],
                        "yaw": self.config.camera_rot_0_rear[2],
                        "width": self.config.camera_width,
                        "height": self.config.camera_height,
                        "fov": self.config.camera_fov,
                        "id": "rgb_rear",
                        "sensor_tick": self.config.carla_fps,
                    })
    def __call__(self, sensor_list_names=None):
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
        if timestamp < 2:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.brake = 0.0
            control.throttle = 0.0
            image = current_data.get("CentralRGB")[1][..., :3]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image=torch.from_numpy(image).type(torch.FloatTensor)
            image=t_u.normalization_wrapper(x=image,config=self.config, type="normalize")
            if self.config.rear_cam:
                current_image_rear= current_data.get("rgb_rear")[1][..., :3]
                current_image_rear = cv2.cvtColor(current_image_rear, cv2.COLOR_BGR2RGB)
                current_image_rear=torch.from_numpy(current_image_rear).type(torch.FloatTensor)
                current_image_rear=t_u.normalization_wrapper(x=current_image_rear,config=self.config, type="normalize")
                image=torch.cat([image, current_image_rear], axis=1)

            current_speed=current_data.get("speed")[1]["speed"]
            current_position = self.vehicle.get_location()
            current_position = np.array([current_position.x, current_position.y])
            replay_params={}
            model=None
            self.prev_rgb_queue.append(image)
            self.prev_speeds_queue.append(current_speed)
            self.prev_location_queue.append(current_position)
            saveable_image=None
        else:
            control, replay_params, model, saveable_image = self.run_step(current_data, timestamp)
        control.manual_gear_shift = False
        return control, replay_params, model, saveable_image

    def yaw_to_orientation(self, yaw):
        # Calculate the orientation vector in old carla convention
        x = np.cos(-yaw)
        y = np.sin(yaw)
        z = 0
        return x, y, z
    
    def _init(self):
        from srunner.scenariomanager.carla_data_provider import (
            CarlaDataProvider,
        )  # pylint: disable=locally-disabled, import-outside-toplevel
        from nav_planner import (
            interpolate_trajectory,
        )  # pylint: disable=locally-disabled, import-outside-toplevel

        self.world_map = CarlaDataProvider.get_map()
        trajectory = [item[0].location for item in self._global_plan_world_coord]
        #dense_route_gps, dense_route_normal = interpolate_trajectory(self.world_map, trajectory, hop_resolution=self.config.interpolation_resolution)  # privileged

        self._waypoint_planner = RoutePlanner(
            self.config.dense_route_planner_min_distance,
            self.config.dense_route_planner_max_distance,
        )
        self._waypoint_planner.set_route(self._global_plan_world_coord, gps=False)
        self.initialized = True
        self.timer=TicToc()
        self.timing=[]

    def run_step(
        self,
        sensor_data,
        timestamp,
    ):
       
        # retrieve location data from sensors and normalize/transform to ego vehicle system
        if not self.initialized:
            self._init()
            if self.config.visualize_without_rgb or self.config.visualize_combined:
                self.init_visualization()
        measurements = sensor_data.get("imu")
        current_location = self.vehicle.get_location()
        current_location = np.array([current_location.x, current_location.y])
        waypoint_route = self._waypoint_planner.run_step(current_location)
        if len(waypoint_route) > 2:
            target_point_location, _ = waypoint_route[1]
        elif len(waypoint_route) > 1:
            target_point_location, _ = waypoint_route[1]
        else:
            target_point_location, _ = waypoint_route[0]

        if (target_point_location != self.target_point_prev).all():
            self.target_point_prev = target_point_location

        current_yaw = measurements[1][-1]
        current_yaw_ego_system = preprocess_compass(current_yaw)

        end_point_location_ego_system = inverse_conversion_2d(
            target_point_location, current_location, current_yaw_ego_system
        )
        end_point_location_ego_system = torch.unsqueeze(
            torch.tensor(end_point_location_ego_system, dtype=torch.float32), dim=0
        ).to("cuda:0")

        # Conversion to old convention necessary in carla >=0.9, only take BGR Values without alpha channel and convert to RGB for the model
        current_image = sensor_data.get("CentralRGB")[1][..., :3]
        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
        current_image=torch.from_numpy(current_image).type(torch.FloatTensor)
        current_image=t_u.normalization_wrapper(x=current_image, config=self.config, type="normalize")
        if self.config.rear_cam:
                current_image_rear= sensor_data.get("rgb_rear")[1][..., :3]
                current_image_rear = cv2.cvtColor(current_image_rear, cv2.COLOR_BGR2RGB)
                current_image_rear=torch.from_numpy(current_image_rear).type(torch.FloatTensor)
                current_image_rear=t_u.normalization_wrapper(x=current_image_rear,config=self.config, type="normalize")
                current_image=torch.cat([current_image, current_image_rear], axis=1)
        vehicle_speed=sensor_data.get("speed")[1]["speed"]
        #add data to queues in beginning of the loop, so we have historical information + current information and then subsample before it comes into the model
        # here we add the current one but the self.prev_location_queue does append later, we fuse only historical locations, but also current speeds and observations
        self.prev_rgb_queue.append(current_image)
        self.prev_speeds_queue.append(vehicle_speed)
        
        
        prev_speeds=list(self.prev_speeds_queue)[::-self.config.data_save_freq*self.config.sampling_rate]
        prev_speeds.reverse()
        prev_speeds=torch.tensor(prev_speeds, dtype=torch.float32).unsqueeze(0).cuda("cuda:0")

        image_input_queue=list(self.prev_rgb_queue)[::-self.config.data_save_freq]
        image_input_queue.reverse()

        all_images = self._process_sensors(image_input_queue)
        if self.config.prevnum>0:
            prev_locations=list(self.prev_location_queue)[::-self.config.data_save_freq*self.config.sampling_rate]
            prev_locations.reverse()
            vehicle_prev_positions=torch.tensor(np.array([inverse_conversion_2d(wp, current_location, current_yaw_ego_system) for wp in prev_locations], dtype=np.float32)).unsqueeze(0).to("cuda:0")
        else:
            vehicle_prev_positions=None
        if timestamp>5 and timestamp<6:
            self.timer.tic()
        if self.config.baseline_folder_name == "arp":
            with torch.no_grad():
                pred_dict_memory = self._mem_extract.module.forward(
                    x=all_images[:-1].unsqueeze(0),
                    speed=None, target_point=end_point_location_ego_system,
                    prev_wp=vehicle_prev_positions,
                )
                pred_dict= self._policy.module.forward(
                    x=all_images[-1].unsqueeze(0).unsqueeze(0),
                    speed=prev_speeds[:,-1:,...] if self.config.speed else None,
                    prev_wp=None,
                    memory_to_fuse=pred_dict_memory["memory"].detach(),
                    target_point=end_point_location_ego_system,
                )
        else:
            with torch.no_grad():
                pred_dict = self.model.module.forward(
                        x=all_images.unsqueeze(0),speed=prev_speeds if self.config.speed else None,
                        target_point=end_point_location_ego_system,
                    prev_wp=vehicle_prev_positions,
                    )
        if timestamp>3 and timestamp<4:
            time_value=self.timer.tocvalue(restart=True)
            self.timing.append(time_value)
        
        current_predictions_wp=pred_dict["wp_predictions"].squeeze().detach().cpu().numpy()
        if len(self.replay_current_predictions_queue)!=0:
            previous_waypoints=list(self.replay_current_predictions_queue)[-1]["wp_predictions"].squeeze().detach().cpu().numpy()
        else:
            previous_waypoints=None
        if previous_waypoints is not None:
            prediction_residual=norm(current_predictions_wp-previous_waypoints, ord=self.config.norm)
        else:
            prediction_residual=None
        if self.config.debug:
            if self.config.baseline_folder_name=="bcso":
                empties=np.concatenate([np.zeros_like(current_image)]*(7-self.config.img_seq_len), axis=0)
                image_sequence=np.concatenate([empties, current_image], axis=0)
                image_sequence=current_image
            else:
                image_sequence=np.concatenate(image_input_queue, axis=0)
 

            if self.config.detectboxes:
                if "arp" in self.config.baseline_folder_name:
                    batch_of_bbs_pred,vel_vecs, accel_vecs=self._policy.module.convert_features_to_bb_metric(pred_dict["pred_bb"])
                else:
                    batch_of_bbs_pred,vel_vecs, accel_vecs=self.model.module.convert_features_to_bb_metric(pred_dict["pred_bb"])
                batch_of_bbs_pred = non_maximum_suppression(np.expand_dims(batch_of_bbs_pred,0), self.config.iou_treshold_nms)
            else:
                batch_of_bbs_pred=None
            if not self.config.bev==1 and (self.config.visualize_without_rgb or self.config.visualize_combined):
                road=self.ss_bev_manager.get_road()
            else:
                road=None
            image_to_save=visualize_model(rgb=image_sequence,config=self.config,closed_loop=True,generate_video=True,
                            velocity_vectors_pred=vel_vecs if self.config.predict_vectors else None,
                            acceleration_vectors_pred=accel_vecs if self.config.predict_vectors else None,
                            save_path_root="",
                            pred_speed=torch.nn.functional.softmax(pred_dict["pred_target_speed"], dim=1).squeeze().detach().cpu().numpy() if self.config.tf_pp_rep else None,
                            target_point=end_point_location_ego_system.squeeze().detach().cpu().numpy(), pred_wp=pred_dict["wp_predictions"].squeeze().detach().cpu().numpy(),
                            pred_bb=batch_of_bbs_pred,step=f"{timestamp:.2f}",
                            pred_bev_semantic=pred_dict["pred_bev_semantic"].squeeze().detach().cpu().numpy() if "pred_bev_semantic" in pred_dict.keys() else None,
                            road=road, parameters=None, args=self.config)
        
        else:
            image_to_save=None
        
        self.replay_current_predictions_queue.append(pred_dict)
        #we need an additional rgb queue, because it is way longer, to be able to visualize collisions with a long horizon
        self.replay_image_queue.append(all_images[-1].detach().cpu().numpy())
        self.replay_target_points_queue.append(end_point_location_ego_system.squeeze().detach().cpu().numpy())
        if self.config.visualize_without_rgb or self.config.visualize_combined:
            self.replay_road_queue.append(self.ss_bev_manager.get_road())
        self.replay_pred_residual_queue.append(prediction_residual)
        if self.config.tf_pp_rep:
            pred_aim_wp = pred_dict["wp_predictions"][:,1]
            pred_aim_wp = pred_aim_wp.squeeze().detach().cpu().numpy()
            pred_angle = -math.degrees(math.atan2(-pred_aim_wp[1], pred_aim_wp[0])) / 90.0
            uncertainty = torch.nn.functional.softmax(pred_dict["pred_target_speed"], dim=1).squeeze().detach().cpu().numpy()
            if uncertainty[0] > self.config.brake_uncertainty_threshold:
                pred_target_speed = self.config.target_speeds[0]
            else:
                pred_target_speed = sum(uncertainty * self.config.target_speeds)
            steer, throttle, brake = self.control_pid_direct(
            pred_target_speed, pred_angle, vehicle_speed
        )
        else:
            steer, throttle, brake = self.control_pid(
                waypoints=pred_dict["wp_predictions"].cpu().detach(), speed=vehicle_speed
            )
 
        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)
        #now we append the current location, that it will be a previous_location in the next step
        self.prev_location_queue.append(current_location)

        print(timestamp, steer, throttle, brake)
        print("target")
        print(target_point_location)
        print("current location")
        print(current_location)
    
        return control,{"image_sequence": list(self.replay_image_queue)[::-self.config.data_save_freq*self.config.sampling_rate],
                "current_predictions": list(self.replay_current_predictions_queue)[::-self.config.data_save_freq*self.config.sampling_rate],
                "target_points": list(self.replay_target_points_queue)[::-self.config.data_save_freq*self.config.sampling_rate],
                "roads": list(self.replay_road_queue)[::-self.config.data_save_freq*self.config.sampling_rate],
                "pred_residual": list(self.replay_pred_residual_queue)[::-self.config.data_save_freq*self.config.sampling_rate],
                "forward_time": np.nanmean(np.array(self.timing))},self.model if self.config.baseline_folder_name!="arp" else self._policy,image_to_save

    def control_pid(self, waypoints, speed):
        """
        Predicts vehicle control with a PID controller.
        Used for waypoint predictions
        """
        assert waypoints.size(0) == 1
        waypoints = waypoints[0]


        # m / s required to drive between waypoint 0.5 and 1.0 second in the future
        one_second = int(self.config.carla_fps // (self.config.wp_dilation * self.config.data_save_freq*self.config.sampling_rate))
        half_second = one_second // 2
        desired_speed = np.linalg.norm(waypoints[half_second - 1] - waypoints[one_second - 1]) * 2.0

        brake = (desired_speed < self.config.brake_speed) or ((speed / desired_speed) > self.config.brake_ratio)

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.clip_throttle)
        throttle = throttle if not brake else 0.0

        # To replicate the slow TransFuser behaviour we have a different distance
        # inside and outside of intersections (detected by desired_speed)
        if desired_speed < self.config.aim_distance_threshold:
            aim_distance = self.config.aim_distance_slow
        else:
            aim_distance = self.config.aim_distance_fast

        # We follow the waypoint that is at least a certain distance away
        aim_index = waypoints.shape[0] - 1
        for index, predicted_waypoint in enumerate(waypoints):
            if np.linalg.norm(predicted_waypoint) >= aim_distance:
                aim_index = index
                break

        aim = waypoints[aim_index]
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        if speed < 0.01:
            # When we don't move we don't want the angle error to accumulate in the integral
            angle = 0.0
        if brake:
            angle = 0.0

        steer = self.turn_controller.step(angle)

        steer = np.clip(steer, -1.0, 1.0)  # Valid steering values are in [-1,1]

        return steer, throttle, brake
    def control_pid_direct(self, pred_target_speed, pred_angle, speed):
        """
        PID controller used for direct predictions
        """
        self.speed_histogram.append(pred_target_speed * 3.6)

        # Target speed of 0 means brake
        brake = pred_target_speed < 0.01

        # We can't steer while the car is stopped
        if speed < 0.01:
            pred_angle = 0.0

        steer = self.turn_controller_direct.step(pred_angle)

        steer = np.clip(steer, -1.0, 1.0)
        steer = round(float(steer), 3)

        if not brake:
            if (speed / pred_target_speed) > self.config.brake_ratio:
                brake = True

        if brake:
            target_speed = 0.0
        else:
            target_speed = pred_target_speed

        delta = np.clip(target_speed - speed, 0.0, self.config.clip_delta)

        throttle = self.speed_controller_direct.step(delta)

        throttle = np.clip(throttle, 0.0, self.config.clip_throttle)

        if brake:
            throttle = 0.0

        return steer, throttle, brake
    def get_attentions(self, layers=None):
        """

        Returns
            The activations obtained from the first layers of the latest iteration.

        """
        if layers is None:
            layers = [0, 1, 2]
        if self.latest_image_tensor is None:
            raise ValueError("No step was ran yet. " "No image to compute the activations, Try Running ")
        all_layers = self._model.get_perception_layers(self.latest_image_tensor)
        cmap = plt.get_cmap("inferno")
        attentions = []
        for layer in layers:
            y = all_layers[layer]
            att = torch.abs(y).mean(1)[0].data.cpu().numpy()
            att = att / att.max()
            att = cmap(att)
            att = np.delete(att, 3, 2)
            attentions.append(cv2.resize(att, [200, 88]))
        return attentions

    def _get_directions(
        self,
        current_location,
        current_orientation_ego_system,
        end_point_location_ego_system,
        end_point_orientation_ego_system,
    ):
        """
        Class that should return the directions to reach a certain goal
        """
        directions = self._planner.get_next_command(
            (current_location[0], current_location[1], 0.22),
            (
                current_orientation_ego_system[0],
                current_orientation_ego_system[1],
                current_orientation_ego_system[2],
            ),
            (end_point_location_ego_system[0], end_point_location_ego_system[1], 0.22),
            (
                end_point_orientation_ego_system[0],
                end_point_orientation_ego_system[1],
                end_point_orientation_ego_system[2],
            ),
        )
        return directions

    def _process_single_frame(self, current_image):
        self.latest_image = current_image
        current_image = current_image.transpose(0, 1)
        current_image = current_image.permute(2, 1, 0)
        return current_image

    def _process_sensors(self, image_list):
        processed_images = []
        for image in image_list:
            processed_image = self._process_single_frame(image)
            processed_images.append(processed_image)
        return torch.stack(processed_images).to("cuda:0")

    def _process_model_outputs_actions(self, outputs, norm_speed, predicted_speed, avoid_stop=False):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """
        assert len(self.config.targets) == len(outputs), "the dimension of outputs does not match the TARGETS!"
        if len(self.config.targets) == 3:
            steer, throttle, brake = (
                float(outputs[0]),
                float(outputs[1]),
                float(outputs[2]),
            )
            if brake < 0.05:
                brake = 0.0

            if throttle > brake:
                brake = 0.0
        elif len(self.config.targets) == 2:
            steer, throttle_brake = float(outputs[0]), float(outputs[1])
            if throttle_brake >= 0:
                throttle = throttle_brake
                brake = 0
            else:
                throttle = 0
                brake = -throttle_brake
        else:
            raise Exception("only support 2 or 3 dimensional outputs")

        if avoid_stop:
            real_speed = norm_speed * self.config.speed_factor
            real_predicted_speed = predicted_speed * self.config.speed_factor

            if real_speed < 5.0 and real_predicted_speed > 6.0:  # If (Car Stooped) and ( It should not have stoped)
                throttle += 20.0 / self.config.speed_factor - norm_speed
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