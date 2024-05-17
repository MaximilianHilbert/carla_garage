from torch.nn import functional as F
import torch


def normalize(x, dim):
    x_normed = x / x.max(dim, keepdim=True)[0]
    return x_normed


def weight_decay_l1(loss, model, intention_factors, alpha, gating):
    wdecay = 0
    for w in model.parameters():
        if w.requires_grad:
            wdecay = torch.add(torch.sum(torch.abs(w)), wdecay)

    if intention_factors is not None:
        intention, _ = torch.min(intention_factors, 1)
        intention = (1.0 > intention).float()
        if gating == "hard":
            # Multiply by a factor proportional to the size of the number of non 1
            wdecay = wdecay * intention.shape[0] / torch.sum(intention)

        elif gating == "easy":
            wdecay = wdecay * torch.sum(intention) / intention.shape[0]

    loss = torch.add(loss, alpha * wdecay)
    return loss


def weight_decay_l2(loss, model, intention_factors, alpha, gating):
    wdecay = 0
    for w in model.parameters():
        if w.requires_grad:
            wdecay = torch.add(torch.sum(w**2), wdecay)

    if intention_factors is not None:
        intention, _ = torch.min(intention_factors, 1)
        intention = (1.0 > intention).float()
        if gating == "hard":
            # Multiply by a factor proportional to the size of the number of non 1
            wdecay = wdecay * intention.shape[0] / torch.sum(intention)

        elif gating == "easy":
            wdecay = wdecay * torch.sum(intention) / intention.shape[0]

    loss = torch.add(loss, alpha * wdecay)
    return loss


def compute_branches_masks(controls, config):
    """
    Args
        controls
        the control values that have the following structure
        command flags: 2 - follow lane; 3 - turn left; 4 - turn right; 5 - go straight
        size of targets:
        How many targets is produced by the network so we can produce the masks properly
    Returns
        a mask to have the loss function applied
        only on over the correct branch.
    """

    """ A vector with a mask for each of the control branches"""
    if config.use_wp_gru:
        number_targets = 2
    else:
        number_targets = 3

    controls_masks = []
    # hardcoded and changes due to new carla dataset
    # 0: LEFT, 1: RIGHT, 2: STRAIGHT, 3: LANEFOLLOW, 4: CHANGELANELEFT, 5: CHANGELANERIGHT
    controls_b0 = controls == 0
    controls_b0 = controls_b0.to(torch.float32)
    controls_b0 = (
        torch.cat([controls_b0] * number_targets, 1).unsqueeze(1).repeat(1, config.pred_len, 1)
        if config.use_wp_gru
        else torch.cat([controls_b0] * number_targets, 1)
    )
    controls_masks.append(controls_b0)

    controls_b1 = controls == 1
    controls_b1 = controls_b1.to(torch.float32)
    controls_b1 = (
        torch.cat([controls_b1] * number_targets, 1).unsqueeze(1).repeat(1, config.pred_len, 1)
        if config.use_wp_gru
        else torch.cat([controls_b1] * number_targets, 1)
    )
    controls_masks.append(controls_b1)

    controls_b2 = controls == 2
    controls_b2 = controls_b2.to(torch.float32)
    controls_b2 = (
        torch.cat([controls_b2] * number_targets, 1).unsqueeze(1).repeat(1, config.pred_len, 1)
        if config.use_wp_gru
        else torch.cat([controls_b2] * number_targets, 1)
    )
    controls_masks.append(controls_b2)

    controls_b3 = controls == 3
    controls_b3 = controls_b3.to(torch.float32)
    controls_b3 = (
        torch.cat([controls_b3] * number_targets, 1).unsqueeze(1).repeat(1, config.pred_len, 1)
        if config.use_wp_gru
        else torch.cat([controls_b3] * number_targets, 1)
    )
    controls_masks.append(controls_b3)

    controls_b4 = controls == 4
    controls_b4 = controls_b4.to(torch.float32)
    controls_b4 = (
        torch.cat([controls_b4] * number_targets, 1).unsqueeze(1).repeat(1, config.pred_len, 1)
        if config.use_wp_gru
        else torch.cat([controls_b4] * number_targets, 1)
    )
    controls_masks.append(controls_b4)

    controls_b5 = controls == 5
    controls_b5 = controls_b5.to(torch.float32)
    controls_b5 = (
        torch.cat([controls_b5] * number_targets, 1).unsqueeze(1).repeat(1, config.pred_len, 1)
        if config.use_wp_gru
        else torch.cat([controls_b5] * number_targets, 1)
    )
    controls_masks.append(controls_b5)

    return controls_masks


def l2_loss(params):
    """
    Functional LOSS L2
    Args
        params dictionary that should include:
            branches: The tensor containing all the branches branches output from the network
            targets: The ground truth targets that the network should produce
            controls_mask: the masked already expliciting the branches tha are going to be used
            branches weights: the weigths that each branch will have on the loss function
            speed_gt: the ground truth speed for these data points

    Returns
        A vector with the loss function

    """
    """ It is a vec for each branch"""
    
    return (params["predictions"] - params["targets"]) ** 2
    

def l1_loss(params):
    """
    Functional LOSS L1
    Args
        params dictionary that should include:
            branches: The tensor containing all the branches branches output from the network
            targets: The ground truth targets that the network should produce
            controls_mask: the masked already expliciting the branches tha are going to be used
            branches weights: the weigths that each branch will have on the loss function
            speed_gt: the ground truth speed for these data points

    Returns
        A vector with the loss function

    """
    """ It is a vec for each branch"""
    return torch.abs((params["predictions"] - params["targets"]))
