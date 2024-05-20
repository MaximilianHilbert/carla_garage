from . import loss_functional as LF
import torch
import torch.nn as nn


def l1(params):
    return branched_loss(LF.l1_loss, params)


def l2(params):
    return branched_loss(LF.l2_loss, params)


def branched_loss(loss_function, params):
    """
    Args
        loss_function: The loss functional that is actually computing the loss
        params: all the parameters, including
                branches: The tensor containing all the branches branches output from the network
                targets: The ground truth targets that the network should produce
                controls: the controls used for each point
                branches weights: the weigths that each branch will have on the loss function
                speed_gt: the ground truth speed for these data points
                variable_weights: The weights for each of the variables used

                For other losses it could contain more parameters

    Returns
        The computed loss function, but also a dictionary with plotable variables for tensorboard
    """

    loss = loss_function(params)


    """ importance sampling """
    importance_sampling_method = params["importance_sampling_method"]
    if importance_sampling_method == "mean":
        weighted_loss = torch.mean(loss)
    else:
        weight_importance_sampling = params["action_predict_loss"]

        if importance_sampling_method == "softmax":
            weighted_loss_function = loss_function * nn.functional.softmax(
                weight_importance_sampling / params["importance_sampling_softmax_temper"],
                dim=0,
            )
        elif importance_sampling_method == "threshold":
            scaled_weight_importance = (weight_importance_sampling > params["importance_sampling_threshold"]).type(
                torch.float
            ) * (params["importance_sampling_threshold_weight"] - 1) + 1
            weighted_loss_function = loss * scaled_weight_importance
        else:
            raise ValueError
        weighted_loss = torch.mean(weighted_loss_function)
    return weighted_loss


def Loss(loss_name):
    """Factory function

    Note: It is defined with the first letter as uppercase even though is a function to contrast
    the actual use of this function that is making classes
    """
    # TODO: this could be extended to some more arbitrary definition

    if loss_name == "L1":
        return l1

    elif loss_name == "L2":
        return l2

    else:
        raise ValueError(" Not found Loss name")
