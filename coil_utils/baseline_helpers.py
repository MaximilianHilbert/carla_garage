import yaml
import torch
from team_code.config import GlobalConfig
import os
import heapq
from coil_network.loss_functional import compute_branches_masks
import numpy as np
from PIL import Image
import cv2
from team_code import transfuser_utils as t_u
from pathlib import Path
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def merge_with_yaml(transfuser_config_object, baseline_name, experiment):
    with open(
        os.path.join(os.environ.get("CONFIG_ROOT"), baseline_name, experiment + ".yaml"), "r"
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
        shared_configuration.initialize(
            root_dir=shared_configuration.root_dir, setting=args.setting
        )
    merge_with_yaml(shared_configuration, args.baseline_folder_name, args.experiment)
    merge_with_command_line_args(shared_configuration, args)
    return shared_configuration


def get_predictions(controls, branches):
    controls_mask = compute_branches_masks(controls, branches[0].shape[1])
    loss_branches_vec = []
    for i in range(len(branches) - 1):
        loss_branches_vec.append(branches[i] * controls_mask[i])
    return loss_branches_vec, branches[-1]


def visualize_model(
    iteration, current_image, current_speed, action_labels, controls, branches, loss
):
    actions, speed = get_predictions(controls, branches)
    actions = [action_tuple.detach().cpu().numpy() for action_tuple in actions]
    speed = speed[0].detach().cpu().numpy()

    from PIL import Image, ImageDraw, ImageFont

    current_image = torch.squeeze(current_image)
    current_image = current_image.cpu().numpy()
    current_image = current_image * 255
    # Load the image
    image = np.transpose(current_image, axes=(1, 2, 0)).astype(np.uint8)
    image = Image.fromarray(image)

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Define the text and its position
    speed_text = f"current_speed_label: {current_speed}"
    speed_text_pos = (10, 10)
    loss_text = f"current_loss: {loss}"
    loss_test_pos = (500, 20)

    controls_text = f"current_control: {controls}"
    controls_test_pos = (500, 10)
    speed_pred_text = f"speed_prediction: {speed}"
    speed_pred_pos = (10, 20)

    actions_labels_text = f"action_labels: {action_labels}"
    actions_labels_pos = (10, 40)

    # Choose a font and size
    actions_pred_text = f"action_predictions: {np.array2string(np.array(actions))}"
    actions_pred_pos = (10, 50)
    # Render the text onto the image
    draw.text(controls_test_pos, controls_text, fill="white")
    draw.text(loss_test_pos, loss_text, fill="white")
    draw.text(speed_text_pos, speed_text, fill="white")
    draw.text(speed_pred_pos, speed_pred_text, fill="white")

    draw.text(actions_labels_pos, actions_labels_text, fill="white")
    draw.text(actions_pred_pos, actions_pred_text, fill="white")

    # Save the image with the text
    image.save(f"/home/maximilian/Master/carla_garage/vis/{iteration}.jpg")
def visualize_model(  # pylint: disable=locally-disabled, unused-argument
      baseline,
      config,
      save_path,
      step,
      rgb,
      lidar_bev,
      target_point,
      pred_wp=None,
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
      wp_selected=None):
    # 0 Car, 1 Pedestrian, 2 Red light, 3 Stop sign
    color_classes = [np.array([255, 165, 0]), np.array([0, 255, 0]), np.array([255, 0, 0]), np.array([250, 160, 160])]

    size_width = int((config.max_y - config.min_y) * config.pixels_per_meter)
    size_height = int((config.max_x - config.min_x) * config.pixels_per_meter)

    scale_factor = 4
    origin = ((size_width * scale_factor) // 2, (size_height * scale_factor) // 2)
    loc_pixels_per_meter = config.pixels_per_meter * scale_factor

    ## add rgb image and lidar
    if config.use_ground_plane:
      images_lidar = np.concatenate(list(lidar_bev.detach().cpu().numpy()[0][:1]), axis=1)
    else:
      images_lidar = np.concatenate(list(lidar_bev.detach().cpu().numpy()[0][:1]), axis=1)

    images_lidar = 255 - (images_lidar * 255).astype(np.uint8)
    images_lidar = np.stack([images_lidar, images_lidar, images_lidar], axis=-1)

    images_lidar = cv2.resize(images_lidar,
                              dsize=(images_lidar.shape[1] * scale_factor, images_lidar.shape[0] * scale_factor),
                              interpolation=cv2.INTER_NEAREST)
    # # Render road over image
    # road = self.ss_bev_manager.get_road()
    # # Alpha blending the road over the LiDAR
    # images_lidar = road[:, :, 3:4] * road[:, :, :3] + (1 - road[:, :, 3:4]) * images_lidar

    if pred_bev_semantic is not None:
      bev_semantic_indices = np.argmax(pred_bev_semantic[0].detach().cpu().numpy(), axis=0)
      converter = np.array(config.bev_classes_list)
      converter[1][0:3] = 40
      bev_semantic_image = converter[bev_semantic_indices, ...].astype('uint8')
      alpha = np.ones_like(bev_semantic_indices) * 0.33
      alpha = alpha.astype(np.float)
      alpha[bev_semantic_indices == 0] = 0.0
      alpha[bev_semantic_indices == 1] = 0.1

      alpha = cv2.resize(alpha, dsize=(alpha.shape[1] * 4, alpha.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
      alpha = np.expand_dims(alpha, 2)
      bev_semantic_image = cv2.resize(bev_semantic_image,
                                      dsize=(bev_semantic_image.shape[1] * 4, bev_semantic_image.shape[0] * 4),
                                      interpolation=cv2.INTER_NEAREST)

      images_lidar = bev_semantic_image * alpha + (1 - alpha) * images_lidar

    if gt_bev_semantic is not None:
      bev_semantic_indices = gt_bev_semantic[0].detach().cpu().numpy()
      converter = np.array(config.bev_classes_list)
      converter[1][0:3] = 40
      bev_semantic_image = converter[bev_semantic_indices, ...].astype('uint8')
      alpha = np.ones_like(bev_semantic_indices) * 0.33
      alpha = alpha.astype(np.float)
      alpha[bev_semantic_indices == 0] = 0.0
      alpha[bev_semantic_indices == 1] = 0.1

      alpha = cv2.resize(alpha, dsize=(alpha.shape[1] * 4, alpha.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
      alpha = np.expand_dims(alpha, 2)
      bev_semantic_image = cv2.resize(bev_semantic_image,
                                      dsize=(bev_semantic_image.shape[1] * 4, bev_semantic_image.shape[0] * 4),
                                      interpolation=cv2.INTER_NEAREST)
      images_lidar = bev_semantic_image * alpha + (1 - alpha) * images_lidar

      images_lidar = np.ascontiguousarray(images_lidar, dtype=np.uint8)

    # Draw wps
    # Red ground truth
    if gt_wp is not None:
      gt_wp_color = (255, 255, 0)
      for wp in gt_wp.detach().cpu().numpy()[0]:
        wp_x = wp[0] * loc_pixels_per_meter + origin[0]
        wp_y = wp[1] * loc_pixels_per_meter + origin[1]
        cv2.circle(images_lidar, (int(wp_x), int(wp_y)), radius=10, color=gt_wp_color, thickness=-1)

    # Green predicted checkpoint
    if pred_checkpoint is not None:
      for wp in pred_checkpoint.detach().cpu().numpy()[0]:
        wp_x = wp[0] * loc_pixels_per_meter + origin[0]
        wp_y = wp[1] * loc_pixels_per_meter + origin[1]
        cv2.circle(images_lidar, (int(wp_x), int(wp_y)),
                   radius=8,
                   lineType=cv2.LINE_AA,
                   color=(0, 128, 255),
                   thickness=-1)

    # Blue predicted wp
    if pred_wp is not None:
      pred_wps = pred_wp.detach().cpu().numpy()[0]
      num_wp = len(pred_wps)
      for idx, wp in enumerate(pred_wps):
        color_weight = 0.5 + 0.5 * float(idx) / num_wp
        wp_x = wp[0] * loc_pixels_per_meter + origin[0]
        wp_y = wp[1] * loc_pixels_per_meter + origin[1]
        cv2.circle(images_lidar, (int(wp_x), int(wp_y)),
                   radius=8,
                   lineType=cv2.LINE_AA,
                   color=(0, 0, int(color_weight * 255)),
                   thickness=-1)

    # Draw target points
    if config.use_tp:
      x_tp = target_point[0][0] * loc_pixels_per_meter + origin[0]
      y_tp = target_point[0][1] * loc_pixels_per_meter + origin[1]
      cv2.circle(images_lidar, (int(x_tp), int(y_tp)), radius=12, lineType=cv2.LINE_AA, color=(255, 0, 0), thickness=-1)

    # Visualize Ego vehicle
    sample_box = np.array([
        int(images_lidar.shape[0] / 2),
        int(images_lidar.shape[1] / 2), config.ego_extent_x * loc_pixels_per_meter,
        config.ego_extent_y * loc_pixels_per_meter,
        np.deg2rad(90.0), 0.0
    ])
    images_lidar = t_u.draw_box(images_lidar, sample_box, color=(0, 200, 0), pixel_per_meter=16, thickness=4)

    if pred_bb is not None:
      for box in pred_bb:
        inv_brake = 1.0 - box[6]
        color_box = deepcopy(color_classes[int(box[7])])
        color_box[1] = color_box[1] * inv_brake
        box = t_u.bb_vehicle_to_image_system(box, loc_pixels_per_meter, config.min_x, config.min_y)
        images_lidar = t_u.draw_box(images_lidar, box, color=color_box, pixel_per_meter=loc_pixels_per_meter)

    if gt_bbs is not None:
      gt_bbs = gt_bbs.detach().cpu().numpy()[0]
      real_boxes = gt_bbs.sum(axis=-1) != 0.
      gt_bbs = gt_bbs[real_boxes]
      for box in gt_bbs:
        box[:4] = box[:4] * scale_factor
        images_lidar = t_u.draw_box(images_lidar, box, color=(0, 255, 255), pixel_per_meter=loc_pixels_per_meter)

    images_lidar = np.rot90(images_lidar, k=1)
    if baseline not in ["bcoh", "arp", "keyframes"]:
        rgb_image = rgb.permute(1, 2, 0).detach().cpu().numpy()
    else:
       rgb_image = rgb[0].permute(1, 2, 0).detach().cpu().numpy()
    if wp_selected is not None:
      colors_name = ['blue', 'yellow']
      colors_idx = [(0, 0, 255), (255, 255, 0)]
      images_lidar = np.ascontiguousarray(images_lidar, dtype=np.uint8)
      cv2.putText(images_lidar, 'Selected: ', (700, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
      cv2.putText(images_lidar, f'{colors_name[wp_selected]}', (850, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                  colors_idx[wp_selected], 2, cv2.LINE_AA)

    if pred_speed is not None:
      pred_speed = pred_speed.detach().cpu().numpy()[0]
      images_lidar = np.ascontiguousarray(images_lidar, dtype=np.uint8)
      t_u.draw_probability_boxes(images_lidar, pred_speed, config.target_speeds)

    all_images = np.concatenate((rgb_image, images_lidar), axis=0)
    all_images = Image.fromarray(all_images.astype(np.uint8))

    store_path = str(str(save_path) + (f'/{step:04}.png'))
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)
    all_images.save(store_path)

def is_ready_to_save(epoch, iteration, data_loader, merged_config):
    """Returns if the iteration is a iteration for saving a checkpoint"""
    if (
        epoch % merged_config.every_epoch == 0
        and epoch != 0
        and iteration == len(data_loader)
    ):
        return True
    else:
        return False


def get_latest_saved_checkpoint(shared_config_object, repetition):
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
            "checkpoints",
        )
    )
    if checkpoint_files == []:
        return None
    else:
        sorted(checkpoint_files)
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
        threshold = heapq.nlargest(
            int(len(action_predict_losses) * ratio), action_predict_losses
        )[-1]
        _action_predict_loss_threshold[ratio] = threshold
        return threshold
