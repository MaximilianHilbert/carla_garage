import yaml
import torch
from team_code.config import GlobalConfig
from PIL import Image, ImageDraw, ImageFont
import os
import heapq
from coil_network.loss_functional import compute_branches_masks
import numpy as np
from PIL import Image
import cv2
from team_code import transfuser_utils as t_u
from pathlib import Path

def norm(differences, ord):
    if ord==1:
        return np.sum(np.absolute(differences))
    if ord==2:
        return np.sqrt(np.sum(np.absolute(differences)**2))
    
def get_copycat_criteria(data_df, which_norm):
    data_df['running_diff'] = data_df['pred'].diff()
    data_df['running_diff']=data_df['running_diff'].apply(lambda x: norm(x,ord=which_norm))
    data_df['running_diff_gt'] = data_df['gt'].diff()
    data_df['running_diff_gt']=data_df['running_diff_gt'].apply(lambda x: norm(x,ord=which_norm))

    std_value_gt=np.std(data_df["running_diff_gt"])
    std_value_data=np.std(data_df["running_diff"])

    mean_value_gt=np.mean(data_df["running_diff_gt"])
    mean_value_data=np.mean(data_df["running_diff"])

    mean_value_loss=np.mean(data_df["loss"])
    std_value_loss=np.std(data_df["loss"])

    return {"mean_pred": mean_value_data,"mean_gt": mean_value_gt, "std_pred":std_value_data, "std_gt":std_value_gt, "loss_mean": mean_value_loss, "loss_std": std_value_loss}

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def find_free_port():
    """https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number"""
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def merge_with_yaml(transfuser_config_object, baseline_name, experiment):
    with open(
        os.path.join(os.environ.get("CONFIG_ROOT"), baseline_name, experiment + ".yaml"),
        "r",
    ) as f:
        yaml_config = yaml.safe_load(f)
    for key, value in yaml_config.items():
        if not hasattr(transfuser_config_object, key):
            raise ValueError("Wrong Attribute set in yaml of baseline")
        setattr(transfuser_config_object, key, value)


def merge_with_command_line_args(config, args):
    args_dict = vars(args)
    for key, value in args_dict.items():
        setattr(config, key, value)


def merge_config_files(args, training=True):
    # due to the baselines using the coiltraine framework, we merge our config file config.py with the .yaml files configuring the baseline models
    # init transfuser config file, necessary for the dataloader
    shared_configuration = GlobalConfig()
    if training:
        shared_configuration.initialize(root_dir=shared_configuration.root_dir, setting=args.setting)
    merge_with_yaml(shared_configuration, args.baseline_folder_name, args.experiment)
    merge_with_command_line_args(shared_configuration, args)
    return shared_configuration


def get_predictions(controls, branches):
    controls_mask = compute_branches_masks(controls, branches[0].shape[1])
    loss_branches_vec = []
    for i in range(len(branches) - 1):
        loss_branches_vec.append(branches[i] * controls_mask[i])
    return loss_branches_vec, branches[-1]



def visualize_model(  # pylint: disable=locally-disabled, unused-argument
    args,
    config,
    save_path_with_rgb,
    save_path_without_rgb,
    step,
    rgb, #tensor of one or multiple images shape img,h,w,c
    target_point,
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

    frame=None,
    loss=None,
    condition=None,
    prev_gt=None,
    ego_speed=None,
    correlation_weight=None
):
    # 0 Car, 1 Pedestrian, 2 Red light, 3 Stop sign
    color_classes = [
        np.array([255, 165, 0]),
        np.array([0, 255, 0]),
        np.array([255, 0, 0]),
        np.array([250, 160, 160]),
    ]

    size_width = int((config.max_y - config.min_y) * config.pixels_per_meter)
    size_height = int((config.max_x - config.min_x) * config.pixels_per_meter)

    scale_factor = 4
    origin = ((size_width * scale_factor) // 2, (size_height * scale_factor) // 2)
    loc_pixels_per_meter = config.pixels_per_meter * scale_factor

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
    # # Render road over image
    # road = self.ss_bev_manager.get_road()
    # # Alpha blending the road over the LiDAR
    # images_lidar = road[:, :, 3:4] * road[:, :, :3] + (1 - road[:, :, 3:4]) * images_lidar

    if pred_bev_semantic is not None:
        bev_semantic_indices = np.argmax(pred_bev_semantic, axis=0)
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
    pred_wp_prev_color= light_yellow

    
    gt_size=12
    prev_gt_size=9

    pred_size=7
    prev_pred_size=4

    # Draw wps
    # Red ground truth
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

    if prev_gt is not None:
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
    if pred_wp_prev is not None:
        pred_wp_prev = pred_wp_prev
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
    images_lidar = t_u.draw_box(images_lidar, sample_box, color=firebrick, pixel_per_meter=16, thickness=4)
    if pred_bb is not None:
        for box in pred_bb:
            inv_brake = 1.0 - box[6]
            color_box = deepcopy(color_classes[int(box[7])])
            color_box[1] = color_box[1] * inv_brake
            box = t_u.bb_vehicle_to_image_system(box, loc_pixels_per_meter, config.min_x, config.min_y)
            images_lidar = t_u.draw_box(images_lidar, box, color=color_box, pixel_per_meter=loc_pixels_per_meter)

    if gt_bbs is not None:
        gt_bbs = gt_bbs
        real_boxes = gt_bbs.sum(axis=-1) != 0.0
        gt_bbs = gt_bbs[real_boxes]
        for box in gt_bbs:
            box[:4] = box[:4] * scale_factor
            images_lidar = t_u.draw_box(
                images_lidar,
                box,
                color=(0, 255, 255),
                pixel_per_meter=loc_pixels_per_meter,
            )

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
   
    
    all_images = np.concatenate([rgb,images_lidar],axis=0)
    font = ImageFont.truetype("Ubuntu-B.ttf", 40)
    font_baseline = ImageFont.truetype("Ubuntu-B.ttf", 100)
    font_copycat=ImageFont.truetype("Ubuntu-B.ttf", 70)
    distance_from_left=600
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
        draw = ImageDraw.Draw(image)

        draw.text((distance_from_left,start), f"frame {frame}", fill=(0, 0, 0), font=font)

        draw.text((distance_from_left,start+40*3), f"pred. res.: {parameters['pred_residual']:.2f}", fill=firebrick, font=font)
        draw.text((distance_from_left,start+40*4), f"gt. res.: {parameters['gt_residual']:.2f}", fill=firebrick, font=font)
        draw.text((distance_from_left,start+40*5), f"loss: {loss:.2f}", fill=firebrick, font=font)

        draw.text((distance_from_left,start+40*6), f"previous ground truth", fill=prev_gt_wp_color, font=font)
        draw.text((distance_from_left,start+40*7), f"current ground truth", fill=gt_wp_color, font=font)
        draw.text((distance_from_left,start+40*8), f"previous predictions", fill=pred_wp_prev_color, font=font)
        draw.text((distance_from_left,start+40*9), f"current predictions", fill=pred_wp_color, font=font)
        
        draw.text((distance_from_left,start+40*16), f"condition: {condition}", fill=firebrick, font=font)
        draw.text((distance_from_left,start+40*17), f"pred. res. th. < {parameters['condition_value_1']:.2f}", fill=firebrick, font=font)
        draw.text((distance_from_left,start+40*18), f"gt. res. th. > {parameters['condition_value_2']:.2f}", fill=firebrick, font=font)
        draw.text((distance_from_left,start+40*19), f"kf. th. > {parameters['condition_value_keyframes']:.2f}", fill=firebrick, font=font)

        draw.text((distance_from_left,start+40*20), f"kf. score: {correlation_weight:.2f}", fill=firebrick, font=font)

        draw.text((distance_from_left,start+40*22), f"ego speed: {ego_speed:.2f} km/h", fill=firebrick, font=font)

        draw.text((50,50), f"{config.baseline_folder_name.upper()}", fill=(255,255,255) if image_name=="combined" else (0,0,0), font=font_baseline)
        if detect_our:
            font.set_variation_by_name("Bold")
            draw.text((distance_from_left,start+40*10), f"our copycat", fill=firebrick, font=font_copycat)
        if detect_kf:
            font.set_variation_by_name("Bold")
            draw.text((distance_from_left,start+40*13), f"kf. copycat", fill=dark_blue, font=font_copycat)

        final_image=np.array(image)
        final_image_object = Image.fromarray(final_image.astype(np.uint8))
        if image_name=="combined":
            store_path = str(str(save_path_with_rgb) + (f"/{step}.jpg"))
        else:
            store_path = str(str(save_path_without_rgb) + (f"/{step}.jpg"))
        Path(store_path).parent.mkdir(parents=True, exist_ok=True)
        final_image_object.save(store_path, quality=95)


def is_ready_to_save(epoch, iteration, data_loader, merged_config):
    """Returns if the iteration is a iteration for saving a checkpoint"""
    if epoch % merged_config.every_epoch == 0 and epoch != 0 and iteration == len(data_loader):
        return True
    else:
        return False


def get_latest_saved_checkpoint(shared_config_object, repetition, setting):
    """
    Returns the , latest checkpoint number that was saved

    """
    checkpoint_files = os.listdir(
        os.path.join(
            f'{os.environ.get("WORK_DIR")}',
            "_logs",
            shared_config_object.baseline_folder_name,
            shared_config_object.experiment,
            f"repetition_{str(repetition)}",
            setting,
            "checkpoints",
        )
    )
    if checkpoint_files == []:
        return None
    else:
        checkpoint_files=sorted([int(x.strip(".pth")) for x in checkpoint_files])
        return checkpoint_files[-1]


def get_controls_from_data(data, batch_size, device_id):
    one_hot_tensor = data["command"].to(device_id)
    indices = torch.argmax(one_hot_tensor, dim=1)
    controls = indices.reshape(batch_size, 1)
    return controls


def get_action_predict_loss_threshold(correlation_weights, ratio):
    _action_predict_loss_threshold = {}
    if ratio in _action_predict_loss_threshold:
        return _action_predict_loss_threshold[ratio]
    else:
        action_predict_losses = correlation_weights
        threshold = heapq.nlargest(int(len(action_predict_losses) * ratio), action_predict_losses)[-1]
        _action_predict_loss_threshold[ratio] = threshold
        return threshold
