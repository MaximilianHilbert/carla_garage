from torch.nn.functional import l1_loss, mse_loss
from torch.nn import CrossEntropyLoss
import torch
import numpy as np

def l1(params):
    return branched_loss(l1_loss, params)


def l2(params):
    return branched_loss(mse_loss, params)


def branched_loss(loss_function, params):
    losses=[]
    main_loss = loss_function(params["wp_predictions"], params["targets"])
    losses.append(main_loss)
    if params["config"].bev:
        if params["config"].use_label_smoothing:
            label_smoothing = params["config"].label_smoothing_alpha
        else:
            label_smoothing = 0.0
        loss_bev_semantic = CrossEntropyLoss(
                weight=torch.tensor(params["config"].bev_semantic_weights, dtype=torch.float32),
                label_smoothing=label_smoothing,
                ignore_index=-1,
            ).to(device=params["device_id"])
        visible_bev_semantic_label = params["valid_bev_pixels"].squeeze(1).int() * params["bev_targets"]
            # Set 0 class to ignore index -1
        visible_bev_semantic_label = (params["valid_bev_pixels"].squeeze(1).int() - 1) + visible_bev_semantic_label
        loss_bev=loss_bev_semantic(params["pred_bev_semantic"], visible_bev_semantic_label)
        losses.append(loss_bev)
    else:
        return torch.sum(torch.stack(losses)).to(device=params["device_id"])
    return torch.sum(torch.stack(losses)*torch.tensor(params["config"].lossweights).to(device=params["device_id"]))


def Loss(loss_name):
    """Factory function

    Note: It is defined with the first letter as uppercase even though is a function to contrast
    the actual use of this function that is making classes
    """

    if loss_name == "L1":
        return l1

    elif loss_name == "L2":
        return l2

    else:
        raise ValueError(" Not found Loss name")
