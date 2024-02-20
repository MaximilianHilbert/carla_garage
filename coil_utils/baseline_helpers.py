import yaml
import torch
from team_code.config import GlobalConfig
import os
import heapq
from coil_network.loss_functional import compute_branches_masks


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
