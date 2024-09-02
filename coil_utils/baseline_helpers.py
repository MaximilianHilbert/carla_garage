import torch
from team_code.config import GlobalConfig
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
from PIL import Image
import cv2
from team_code import transfuser_utils as t_u
from pathlib import Path
from copy import deepcopy
from torch.utils.data import Sampler

import os
import requests

def download_file(url, destination):
    if os.path.exists(destination):
        print(f"File already exists at {destination}. Skipping download.")
        return
    print("Pretrained file is being downloaded...")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    response = requests.get(url)
    with open(destination, 'wb') as file:
        file.write(response.content)
    print(f"File downloaded and saved to {destination}.")
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def generate_experiment_name(args, distributed_baseline_folder_name=None):
    ablations_dict=get_ablations_dict()
    args_dict=vars(args)
    ablations_dict.update({arg:args_dict[arg] for arg in args_dict.keys() for ablation in ablations_dict.keys() if arg==ablation})
    if distributed_baseline_folder_name is None:
        return f"{args_dict['baseline_folder_name']}_"+"_".join([f'{ablation}-{",".join(map(str,value)) if isinstance(value, list) else value}' for ablation, value in ablations_dict.items()]),ablations_dict
    else:
        return f"{distributed_baseline_folder_name}_"+"_".join([f'{ablation}-{",".join(map(str,value)) if isinstance(value, list) else value}' for ablation, value in ablations_dict.items()]),ablations_dict
def normalize_vectors(x, config,case, normalize="normalize"):
    ret_lst=[]
    for vec_5d in x:
        vec_3d,id,class_=extract_id_class_from_vector(vec_5d)
        if normalize=="normalize":
            vec_3d=(vec_3d-config.normalization_vectors[case]["mean"])/config.normalization_vectors[case]["std"]
        if normalize=="unnormalize":
            vec_3d=vec_3d*config.normalization_vectors[case]["std"]+config.normalization_vectors[case]["mean"]
        final_vector=append_id_class_to_vector(vec_3d, id, class_)
        ret_lst.append(final_vector)
    return ret_lst
def extract_and_normalize_data(args, device_id, merged_config_object, data):
    if merged_config_object.rear_cam:
        all_images_rear=data["rgb_rear"].to(device_id).to(torch.float32)
        all_images_rear=t_u.normalization_wrapper(x=all_images_rear,config=merged_config_object,type="normalize")
        all_images=data["rgb"].to(device_id).to(torch.float32)
        all_images=t_u.normalization_wrapper(x=all_images,config=merged_config_object,type="normalize")
        all_images=torch.cat([all_images, all_images_rear], axis=4)
    else:
        all_images=data["rgb"].to(device_id).to(torch.float32)
        all_images=t_u.normalization_wrapper(x=all_images,config=merged_config_object,type="normalize")
    
    if merged_config_object.speed or merged_config_object.ego_velocity_prediction==1:
        all_speeds = data["speed"].to(device_id).unsqueeze(2)
    else:
        all_speeds = None
    target_point = data["target_point"].to(device_id)
    wp_targets = data["ego_waypoints"].to(device_id)
    if merged_config_object.number_previous_waypoints>0:
        previous_wp_targets = data["previous_ego_waypoints"].to(device_id)
    else:
        previous_wp_targets=None
    if merged_config_object.prevnum>0:
        additional_previous_wp_targets = torch.flatten(data["additional_waypoints_ego_system"].to(device_id), start_dim=1)
    else:
        additional_previous_wp_targets=None

    bev_semantic_labels=data["bev_semantic"].to(torch.long).to(device_id)
    if merged_config_object.detectboxes:
        
        bb=data["bounding_boxes"].to(device_id, dtype=torch.float32)
        gt_bb={"center_heatmap_target":data["center_heatmap"].to(device_id, dtype=torch.float32),
"wh_target":data["wh"].to(device_id, dtype=torch.float32),
"yaw_class_target":data["yaw_class"].to(device_id, dtype=torch.long),
"yaw_res_target":data["yaw_res"].to(device_id, dtype=torch.float32),
"offset_target":data["offset"].to(device_id, dtype=torch.float32),
"velocity_target":data["velocity"].to(device_id, dtype=torch.float32),
"brake_target":data["brake_target"].to(device_id, dtype=torch.long),
"pixel_weight":data["pixel_weight"].to(device_id, dtype=torch.float32),
"avg_factor":data["avg_factor"].to(device_id, dtype=torch.float32)}
    else:
        gt_bb=None
        bb=None
    if merged_config_object.predict_vectors:
        vel_vecs, accel_vecs=data["velocity_vectors"], data["acceleration_vectors"]
        gt_bb.update({"velocity_vector_target": data["velocity_vector_target"].to(device_id, dtype=torch.float32),"acceleration_vector_target":data["acceleration_vector_target"].to(device_id, dtype=torch.float32) })
    else:
        vel_vecs, accel_vecs=None, None
    return all_images,all_speeds,target_point,wp_targets,previous_wp_targets,additional_previous_wp_targets,bev_semantic_labels, gt_bb,bb, vel_vecs, accel_vecs
def find_free_port():
    """https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number"""
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def align_previous_prediction(pred, matrix_previous, matrix_current):
    matrix_prev = matrix_previous[:3]
    translation_prev = matrix_prev[:, 3:4].flatten()
    rotation_prev = matrix_prev[:, :3]

    matrix_curr = matrix_current[:3]
    translation_curr = matrix_curr[:, 3:4].flatten()
    rotation_curr = matrix_curr[:, :3]

    waypoints=[]
    for waypoint in pred:
        waypoint_3d=np.append(waypoint, 0)
        waypoint_world_frame = (rotation_prev@waypoint_3d) + translation_prev
        waypoint_aligned_frame=rotation_curr.T @(waypoint_world_frame-translation_curr)
        waypoints.append(waypoint_aligned_frame[:-1])
    return np.array(waypoints)

class SequentialSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)
def merge_with_command_line_args(config, args):
    args_dict = vars(args)
    for key, value in args_dict.items():
        try:
            if key=="prevnum" and int(value)==1:
                setattr(config, "prevnum", config.max_img_seq_len_baselines)
            else:
                setattr(config, key, int(value))
        except:
            setattr(config, key, value)

def get_ablations_dict():
    return {"bev":0, "detectboxes": 0,"speed":0, "prevnum":0, "framehandling": "unrolling", "datarep":1, 
            "augment": 0, "freeze": 0, "backbone": "resnet", "pretrained": 0, "subsampling": 0, "velocity_brake_prediction": 1, "ego_velocity_prediction": 0, "init": 0, "predict_vectors": 0, "rear_cam": "0"}

def set_baseline_specific_args(config, experiment_name, args):
    setattr(config, "experiment", experiment_name)
    if "bcoh" in config.baseline_folder_name or "arp" in config.baseline_folder_name or "keyframes" in config.baseline_folder_name:
        setattr(config, "img_seq_len", config.considered_images_incl_current)
        setattr(config, "number_previous_waypoints", 0)
    if "keyframes" in config.baseline_folder_name:
        setattr(config, "correlation_weights", True)
        setattr(config, "number_previous_waypoints", 0)
    if "bcso" in config.baseline_folder_name: #only single observation model
        setattr(config, "img_seq_len", 1)
        setattr(config, "number_previous_waypoints", 0)
    if "arp" in config.baseline_folder_name:
        setattr(config, "number_previous_waypoints", 1) #means that we use the current-1 as first step for the sequence of arp (a_n-1-a_n)
    if "waypoint_weight_generation" in config.baseline_folder_name:
        setattr(config, "img_seq_len", 0)
        setattr(config, "lidar_seq_len", 0)
        setattr(config, "number_previous_waypoints", 9)
        setattr(config, "number_future_waypoints", 9)
        setattr(config, "epochs_baselines", 300)
        setattr(config, "waypoint_weight_generation", True)
    return config
def set_subsampling_strategy(config):
    if config.subsampling:
        if config.backbone!="resnet" and config.backbone!="videoresnet":
            our_fps=config.carla_fps/config.data_save_freq
            backbone_fps=config.video_model_transform_params[config.backbone]["fps"]
            backbone_sampling_rate=config.video_model_transform_params[config.backbone]["sampling_rate"]
            backbone_temporal_delta=backbone_sampling_rate/backbone_fps
            our_sampling_rate=int(np.ceil(backbone_temporal_delta*our_fps))
            setattr(config, "sampling_rate", our_sampling_rate)
        else:
            setattr(config, "sampling_rate", 1)
    else:
        #resnet is pretrained on 1 single image, so the spacing between images is irrelevant or if we dont want to subsample at all
        setattr(config, "sampling_rate", 1)
    return config
def merge_config(args, experiment_name, training=True):
    # due to the baselines using the coiltraine framework, we merge our config file config.py with the .yaml files configuring the baseline models
    # init transfuser config file, necessary for the dataloader
    shared_configuration = GlobalConfig()
    if training:
        shared_configuration.initialize(root_dir=shared_configuration.root_dir, setting=args.setting, num_repetitions=args.datarep)
    
    merge_with_command_line_args(shared_configuration, args)
    shared_configuration=set_baseline_specific_args(shared_configuration, experiment_name, args)
    shared_configuration=set_subsampling_strategy(shared_configuration)
    return shared_configuration




def visualize_model(  # pylint: disable=locally-disabled, unused-argument
    config,
    args,
    save_path_root,
    step,
    rgb, #takes rgb normalized
    target_point,
    training=False,
    road=None,
    closed_loop=False,
    lidar_bev=None,
    pred_wp=None,
    pred_wp_prev=None,
    pred_semantic=None,
    pred_bev_semantic=None,
    pred_depth=None,
    pred_checkpoint=None,
    pred_speed=None,
    pred_bb=None,
    gt_wp=None,
    gt_bbs=None,
    gt_speed=None,
    gt_bev_semantic=None,
    wp_selected=None,
    detect_our=None,
    detect_kf=None,
    parameters=None,
    velocity_vectors_gt=None,
    acceleration_vectors_gt=None,
    velocity_vectors_pred=None,
    acceleration_vectors_pred=None,
    frame=None,
    loss=None,
    condition=None,
    prev_gt=None,
    ego_speed=None,
    correlation_weight=None,
    generate_video=False,
    loss_brake=None,
    loss_velocity=None
):
    
    rgb=t_u.normalization_wrapper(x=rgb, config=config, type="unnormalize")
    if velocity_vectors_pred is not None:
        velocity_vectors_pred=normalize_vectors(velocity_vectors_pred,config,"velocity", "unnormalize")
        acceleration_vectors_pred=normalize_vectors(acceleration_vectors_pred,config,"acceleration", "unnormalize")
    if velocity_vectors_gt is not None:
        velocity_vectors_gt=normalize_vectors(velocity_vectors_gt,config,"velocity", "unnormalize")
        acceleration_vectors_gt=normalize_vectors(acceleration_vectors_gt,config,"acceleration", "unnormalize")
    # 0 Car, 1 Pedestrian, 2 Red light, 3 Stop sign
    color_classes = [
        np.array([144, 238, 144]),
        np.array([0, 255, 0]),
        np.array([255, 0, 0]),
        np.array([250, 160, 160]),
    ]

    size_width = int((config.max_y - config.min_y) * config.pixels_per_meter)
    size_height = int((config.max_x - config.min_x) * config.pixels_per_meter)

    scale_factor = 4
    origin = ((size_width * scale_factor) // 2, (size_height * scale_factor) // 2)
    loc_pixels_per_meter = config.pixels_per_meter * scale_factor
    if not closed_loop:
        ## add rgb image and lidar
        if config.use_ground_plane:
            images_lidar = np.concatenate(list(lidar_bev[:1]), axis=1)
        else:
            images_lidar = np.concatenate(list(lidar_bev[:1]), axis=1)

        images_lidar = 255 - (images_lidar * 255).astype(np.uint8)
        images_lidar = np.stack([images_lidar, images_lidar, images_lidar], axis=-1)

        images_lidar = cv2.resize(
            images_lidar,
            dsize=(
                images_lidar.shape[1] * scale_factor,
                images_lidar.shape[0] * scale_factor,
            ),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        if road is None:
            images_lidar = np.full((1024,1024,3), 255, dtype=np.uint8)
            images_lidar = cv2.resize(
                images_lidar,
                dsize=(
                    images_lidar.shape[1] * 1,
                    images_lidar.shape[0] * 1,
                ),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            images_lidar=road[:,:,:3]
            images_lidar[(images_lidar == [0, 0, 0]).all(axis=2)]=[255.,255.,255.]
            images_lidar = cv2.resize(
                images_lidar,
                dsize=(
                    images_lidar.shape[1] * 1,
                    images_lidar.shape[0] * 1,
                ),
                interpolation=cv2.INTER_NEAREST,
            )

    if pred_bev_semantic is not None:
        bev_semantic_indices = np.argmax(pred_bev_semantic, axis=0)
        converter = np.array(config.bev_classes_list)
        converter[1][0:3] = 40
        bev_semantic_image = converter[bev_semantic_indices, ...].astype("uint8")
        alpha = np.ones_like(bev_semantic_indices) * 0.33
        alpha = alpha.astype(np.float)
        alpha[bev_semantic_indices == 0] = 0.0
        #alpha[bev_semantic_indices == 1] = 0.1

        alpha = cv2.resize(
            alpha,
            dsize=(alpha.shape[1] * 4, alpha.shape[0] * 4),
            interpolation=cv2.INTER_NEAREST,
        )
        alpha = np.expand_dims(alpha, 2)
        bev_semantic_image = cv2.resize(
            bev_semantic_image,
            dsize=(bev_semantic_image.shape[1] * 4, bev_semantic_image.shape[0] * 4),
            interpolation=cv2.INTER_NEAREST,
        )

        images_lidar = bev_semantic_image * alpha + (1 - alpha) * images_lidar

    if gt_bev_semantic is not None:
        bev_semantic_indices = gt_bev_semantic
        converter = np.array(config.bev_classes_list)
        converter[1][0:3] = 40
        bev_semantic_image = converter[bev_semantic_indices, ...].astype("uint8")
        alpha = np.ones_like(bev_semantic_indices) * 0.33
        alpha = alpha.astype(np.float)
        alpha[bev_semantic_indices == 0] = 0.0
        alpha[bev_semantic_indices == 1] = 0.1

        alpha = cv2.resize(
            alpha,
            dsize=(alpha.shape[1] * 4, alpha.shape[0] * 4),
            interpolation=cv2.INTER_NEAREST,
        )
        alpha = np.expand_dims(alpha, 2)
        bev_semantic_image = cv2.resize(
            bev_semantic_image,
            dsize=(bev_semantic_image.shape[1] * 4, bev_semantic_image.shape[0] * 4),
            interpolation=cv2.INTER_NEAREST,
        )
        images_lidar = bev_semantic_image * alpha + (1 - alpha) * images_lidar

        images_lidar = np.ascontiguousarray(images_lidar, dtype=np.uint8)

    #point colors
    light_blue=(104,	195,	212)
    dark_blue=(22,	71,	80	)

    light_yellow=(	252,	209,	78	)
    dark_yellow=(214,	168,	31)
    firebrick=(178, 34, 34)
    gt_wp_color = dark_blue
    pred_wp_color=dark_yellow

    prev_gt_wp_color =  light_blue
    pred_wp_prev_color= firebrick

    
    gt_size=12
    prev_gt_size=9
    if closed_loop:
        pred_size=12
        prev_pred_size=9
    else:
        pred_size=7
        prev_pred_size=4


    # Green predicted checkpoint
    if pred_checkpoint is not None:
        for wp in pred_checkpoint:
            wp_x = wp[0] * loc_pixels_per_meter + origin[0]
            wp_y = wp[1] * loc_pixels_per_meter + origin[1]
            cv2.circle(
                images_lidar,
                (int(wp_x), int(wp_y)),
                radius=8,
                lineType=cv2.LINE_AA,
                color=(0, 128, 255),
                thickness=-1,
            )

    # Blue predicted wp
    if velocity_vectors_pred is None or velocity_vectors_pred==[]:
        if gt_wp is not None:
            for wp in gt_wp:
                wp_x = wp[0] * loc_pixels_per_meter + origin[0]
                wp_y = wp[1] * loc_pixels_per_meter + origin[1]
                cv2.circle(
                    images_lidar,
                    (int(wp_x), int(wp_y)),
                    radius=gt_size,
                    color=gt_wp_color,
                    thickness=-1,
                )
        # Draw wps previous

        if prev_gt is not None and not isinstance(prev_gt, np.float):
            for wp_prev in prev_gt:
                wp_x = wp_prev[0] * loc_pixels_per_meter + origin[0]
                wp_y = wp_prev[1] * loc_pixels_per_meter + origin[1]
                cv2.circle(
                    images_lidar,
                    (int(wp_x), int(wp_y)),
                    radius=prev_gt_size,
                    color=prev_gt_wp_color,
                    thickness=-1,
                )

        if pred_wp is not None:
            pred_wps = pred_wp
            num_wp = len(pred_wps)
            for idx, wp in enumerate(pred_wps):
                color_weight = 0.5 + 0.5 * float(idx) / num_wp
                wp_x = wp[0] * loc_pixels_per_meter + origin[0]
                wp_y = wp[1] * loc_pixels_per_meter + origin[1]
                cv2.circle(
                    images_lidar,
                    (int(wp_x), int(wp_y)),
                    radius=pred_size,
                    lineType=cv2.LINE_AA,
                    color=pred_wp_color,
                    thickness=-1,
                )
        if pred_wp_prev is not None and not isinstance(pred_wp_prev, np.float):
            num_wp = len(pred_wp_prev)
            for idx, wp in enumerate(pred_wp_prev):
                color_weight = 0.5 + 0.5 * float(idx) / num_wp
                wp_x = wp[0] * loc_pixels_per_meter + origin[0]
                wp_y = wp[1] * loc_pixels_per_meter + origin[1]
                cv2.circle(
                    images_lidar,
                    (int(wp_x), int(wp_y)),
                    radius=prev_pred_size,
                    lineType=cv2.LINE_AA,
                    color=pred_wp_prev_color,
                    thickness=-1,
                )
    # Draw target points
    if config.use_tp:
        x_tp = target_point[0] * loc_pixels_per_meter + origin[0]
        y_tp = target_point[1] * loc_pixels_per_meter + origin[1]
        cv2.circle(
            images_lidar,
            (int(x_tp), int(y_tp)),
            radius=12,
            lineType=cv2.LINE_AA,
            color=(255, 0, 0),
            thickness=-1,
        )

    # Visualize Ego vehicle
    sample_box = np.array(
        [
            int(images_lidar.shape[0] / 2),
            int(images_lidar.shape[1] / 2),
            config.ego_extent_x * loc_pixels_per_meter,
            config.ego_extent_y * loc_pixels_per_meter,
            np.deg2rad(90.0),
            0.0,
        ]
    )
    images_lidar = t_u.draw_box(images_lidar, config,sample_box, color=firebrick, pixel_per_meter=16, thickness=4)
    if pred_bb is not None:
        for box in pred_bb:
            inv_brake = 1.0 - box[6]
            color_box = deepcopy(color_classes[int(box[7])])
            color_box[1] = color_box[1] * inv_brake
            box = t_u.bb_vehicle_to_image_system(box, loc_pixels_per_meter, config.min_x, config.min_y)
            images_lidar = t_u.draw_box(images_lidar, config,box, color=color_box, pixel_per_meter=loc_pixels_per_meter)

    if gt_bbs is not None:
        real_boxes = gt_bbs.sum(axis=-1) != 0.0
        gt_bbs = gt_bbs[real_boxes]
        for box in gt_bbs:
            box[:4] = box[:4] * scale_factor
            images_lidar = t_u.draw_box(
                images_lidar,
                config,
                box,
                color=(0, 100, 0),
                pixel_per_meter=loc_pixels_per_meter,
            )
    if velocity_vectors_gt is not None:
        plot_vectors_gt(bbs=gt_bbs, vectors=velocity_vectors_gt, res=loc_pixels_per_meter, image=images_lidar, color=dark_blue, thickness=1, scale_factor=2)
    if acceleration_vectors_gt is not None:
        plot_vectors_gt(bbs=gt_bbs, vectors=acceleration_vectors_gt, res=loc_pixels_per_meter, image=images_lidar, color=dark_blue, thickness=3, scale_factor=2)
    if velocity_vectors_pred is not None:
        plot_vectors_pred(vectors=velocity_vectors_pred, bbs=pred_bb,res=loc_pixels_per_meter, image=images_lidar, color=firebrick, thickness=1, scale_factor=2)
    if acceleration_vectors_pred is not None:
        plot_vectors_pred(vectors=acceleration_vectors_pred, bbs=pred_bb,res=loc_pixels_per_meter, image=images_lidar, color=firebrick, thickness=3, scale_factor=2)
    
    images_lidar = np.rot90(images_lidar, k=1)

    
    if wp_selected is not None:
        colors_name = ["blue", "yellow"]
        colors_idx = [(0, 0, 255), (255, 255, 0)]
        images_lidar = np.ascontiguousarray(images_lidar, dtype=np.uint8)
        cv2.putText(
            images_lidar,
            "Selected: ",
            (700, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            images_lidar,
            f"{colors_name[wp_selected]}",
            (850, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            colors_idx[wp_selected],
            2,
            cv2.LINE_AA,
        )

    if pred_speed is not None:
        pred_speed = pred_speed
        images_lidar = np.ascontiguousarray(images_lidar, dtype=np.uint8)
        t_u.draw_probability_boxes(images_lidar, pred_speed, config.target_speeds)
    if closed_loop:
        images_lidar_zoomed = cv2.resize(
                images_lidar,
                dsize=(
                    images_lidar.shape[1] * 1,
                    images_lidar.shape[0] * 1,
                ),
                interpolation=cv2.INTER_NEAREST,
            )
            
        start_row = (images_lidar_zoomed.shape[0] - images_lidar.shape[0]) // 2
        start_col = (images_lidar_zoomed.shape[1] - images_lidar.shape[1]) // 2

        # Calculate the ending row and column indices for the ROI
        end_row = start_row + images_lidar.shape[0]
        end_col = start_col + images_lidar.shape[1]

        # Extract the ROI from the zoomed-in image
        images_lidar = images_lidar_zoomed[start_row:end_row, start_col:end_col]
        
    if config.rear_cam:

        total_padding = 2048 - images_lidar.shape[1]
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding

        images_lidar = np.pad(images_lidar, ((0, 0), (left_padding, right_padding), (0, 0)), mode='constant', constant_values=0)
    all_images = np.concatenate([rgb,images_lidar],axis=0)
    font = ImageFont.truetype("Ubuntu-B.ttf", 40)
    font_baseline = ImageFont.truetype("Ubuntu-B.ttf", 100)
    font_copycat=ImageFont.truetype("Ubuntu-B.ttf", 70)
    distance_from_left=600
    distance_from_left_left_side=50
    options=[]
    if args.visualize_without_rgb:
        options.append("lidar_only")
    if args.visualize_combined:
        options.append("combined")
    for image_name in options:
            if image_name=="combined":
                start=1800
                image=Image.fromarray(all_images.astype(np.uint8))
            else:
                start=20
                image=Image.fromarray(images_lidar.astype(np.uint8))
    if not training:
            draw = ImageDraw.Draw(image)
            if closed_loop:
                draw.text((distance_from_left,start), f"time {step}", fill=(0, 0, 0), font=font)
            else:
                draw.text((distance_from_left,start), f"frame {frame}", fill=(0, 0, 0), font=font)
            if parameters is not None:
                if "pred_residual" in parameters.keys():
                    if parameters["pred_residual"] is not None:
                        draw.text((distance_from_left,start+40*3), f"pred. res.: {parameters['pred_residual']:.2f}", fill=firebrick, font=font)
            else:
                draw.text((distance_from_left,start+40*3), f"pred. res.: None", fill=firebrick, font=font)
            if not closed_loop:
                if loss_velocity is not None:
                    draw.text((distance_from_left_left_side,start+40*3), f"velocity_loss: {loss_velocity:.2f}", fill=firebrick, font=font)
                else:
                    draw.text((distance_from_left_left_side,start+40*3), f"velocity_loss: None", fill=firebrick, font=font)
                if loss_brake is not None:
                    draw.text((distance_from_left_left_side,start+40*4), f"accel._loss: {loss_brake:.2f}", fill=firebrick, font=font)
                else:
                    draw.text((distance_from_left_left_side,start+40*4), f"accel._loss: None", fill=firebrick, font=font)

                draw.text((distance_from_left,start+40*4), f"gt. res.: {parameters['gt_residual']:.2f}", fill=firebrick, font=font)
                draw.text((distance_from_left,start+40*5), f"loss: {loss:.2f}", fill=firebrick, font=font)
    
                draw.text((distance_from_left,start+40*6), f"previous ground truth", fill=prev_gt_wp_color, font=font)
                draw.text((distance_from_left,start+40*7), f"current ground truth", fill=gt_wp_color, font=font)
                draw.text((distance_from_left,start+40*16), f"condition: {condition}", fill=firebrick, font=font)
                if condition=="loss":
                    draw.text((distance_from_left,start+40*18), f"loss. th. > {parameters['condition_value_2']:.2f}", fill=firebrick, font=font)
                else:
                    draw.text((distance_from_left,start+40*18), f"gt. res. th. > {parameters['condition_value_2']:.2f}", fill=firebrick, font=font)    
                draw.text((distance_from_left,start+40*17), f"pred. res. th. < {parameters['condition_value_1']:.2f}", fill=firebrick, font=font)
                
                draw.text((distance_from_left,start+40*19), f"kf. th. > {parameters['condition_value_keyframes']:.2f}", fill=firebrick, font=font)

                draw.text((distance_from_left,start+40*20), f"kf. score: {correlation_weight:.2f}", fill=firebrick, font=font)

                if detect_our:
                    #font.set_variation_by_name("Bold")
                    draw.text((distance_from_left,start+40*10), f"our copycat", fill=firebrick, font=font_copycat)
                if detect_kf:
                    #font.set_variation_by_name("Bold")
                    draw.text((distance_from_left,start+40*13), f"kf. copycat", fill=dark_blue, font=font_copycat)

            draw.text((distance_from_left,start+40*8), f"previous predictions", fill=pred_wp_prev_color, font=font)
            draw.text((distance_from_left,start+40*9), f"current predictions", fill=pred_wp_color, font=font)
            if ego_speed is not None:
                draw.text((distance_from_left,start+40*22), f"ego speed: {ego_speed[-1]:.2f} km/h", fill=firebrick, font=font)

            draw.text((50,50), f"{config.baseline_folder_name.upper()}", fill=(255,255,255) if image_name=="combined" else (0,0,0), font=font_baseline)
    else:
        image=Image.fromarray(all_images.astype(np.uint8)) 
    
    final_image=np.array(image)
    final_image_object = Image.fromarray(final_image.astype(np.uint8))
    if generate_video:
        return final_image_object
    else:
        if image_name=="combined":
            store_path = os.path.join(save_path_root, "with_rgb", f"{step}.jpg")
        else:
            store_path = os.path.join(save_path_root, "without_rgb", f"{step}.jpg")
        Path(store_path).parent.mkdir(parents=True, exist_ok=True)
        final_image_object.save(store_path, quality=95)
def append_id_class_to_vector(vector_3d, id, car_class=None):
        if car_class is not None:
            vector_4d=np.zeros(vector_3d.shape[0]+2)
        else:
            vector_4d=np.zeros(vector_3d.shape[0]+1)
        vector_4d[:3]=vector_3d[:3]
        if car_class is not None:
            vector_4d[-2]=id
            vector_4d[-1]=1 if car_class=="ego_car" else 0
        else:
            vector_4d[-1]=id
        return vector_4d
def extract_id_class_from_vector(vector_5d):
    vector_3d=vector_5d[:3]
    id=int(vector_5d[-2])
    car_class="ego_car" if int(vector_5d[-1])==1 else "other"
    return vector_3d, id, car_class
def pad_detected_vectors(vectors, config):
    vectors_padded = np.zeros((config.max_num_bbs,5), dtype=np.float32)
    if vectors.shape[0]>0:
        if vectors.shape[0]<=config.max_num_bbs:
            vectors_padded[: vectors.shape[0],:]=vectors
        else:
            vectors_padded[: config.max_num_bbs,:]=vectors[:config.max_num_bbs]
    return vectors_padded
def plot_vectors_gt(bbs, vectors, res, image, color,thickness, scale_factor):
    for actor_bb in bbs:
        for vector_5d in vectors:
            vector_3d, id,car_class=extract_id_class_from_vector(vector_5d)
            vector_3d=vector_3d*res/scale_factor
            #dummy values
            if id==0:
                continue
            #ego vehicle
            if car_class==1:
                cv2.arrowedLine(image, (1024//2, 1024//2), (int(1024//2+vector_3d[0]), int(1024//2+vector_3d[1])), color, thickness)
            start_x, start_y, id_bb=int(actor_bb[0]),int(actor_bb[1]), int(actor_bb[-2])
            if id_bb==id:
                cv2.arrowedLine(image, (start_y, start_x), (start_y+int(vector_3d[0]), start_x+int(vector_3d[1])), color, thickness)

def plot_vectors_pred(vectors, res, image, color,thickness, bbs, scale_factor):
    for bb,vector in zip(bbs,vectors):
        start_x, start_y=int(bb[0]),int(bb[1])
        vector=vector*res/scale_factor
        cv2.arrowedLine(image, (start_y, start_x), (start_y+int(vector[0]), start_x+int(vector[1])), color, thickness)
def set_not_included_ablation_args(config):
    ablations_default_dict=get_ablations_dict()
    for ablation, default_value in ablations_default_dict.items():
        try:
            getattr(config, ablation)
        except:
            setattr(config, ablation, default_value)
def is_ready_to_save(epoch, iteration, data_loader, merged_config):
    """Returns if the iteration is a iteration for saving a checkpoint"""
    if epoch % merged_config.every_epoch == 0 and epoch != 0 and iteration == len(data_loader):
        return True
    else:
        return False
def save_checkpoint_and_delete_prior(state, merged_config_object, args, epoch):
    prior_epoch=epoch-merged_config_object.every_epoch
    checkpoint_dir=os.path.join(
        os.environ.get("WORK_DIR"),
        "_logs",
        merged_config_object.baseline_folder_name,
        args.experiment_id,
        f"repetition_{str(args.training_repetition)}",
        args.setting,
        "checkpoints")
    torch.save(
    state,
    os.path.join(checkpoint_dir
        ,f"{epoch}.pth")
    )
    for checkpoint in os.listdir(checkpoint_dir):
        if checkpoint==f"{prior_epoch}.pth":
            os.remove(os.path.join(checkpoint_dir, checkpoint))
    
    

def get_latest_saved_checkpoint(basepath):
    """
    Returns the , latest checkpoint number that was saved

    """
    checkpoint_files = os.listdir(
        os.path.join(basepath,
            "checkpoints",
        )
    )
    if checkpoint_files == []:
        return None
    else:
        checkpoint_files=sorted([int(x.strip(".pth")) for x in checkpoint_files])
        return checkpoint_files[-1]