from . import loss_functional as LF
import torch


def l1(params):
    return branched_loss(LF.l1_loss, params)


def l2(params):
    return branched_loss(LF.l2_loss, params)


def l1_steer_weak_supervise(params, alpha=1):
    l1_loss, plotable_params = branched_loss(LF.l1_loss, params)
    steer_weak_supervise_loss = alpha * (
        torch.max(
            params["branches"][3][:, 0] - params["branches"][2][:, 0],
            torch.zeros_like(params["branches"][0][:, 0]),
        )
        + torch.max(
            params["branches"][1][:, 0] - params["branches"][3][:, 0],
            torch.zeros_like(params["branches"][0][:, 0]),
        )
    )

    steer_weak_supervise_loss = torch.sum(steer_weak_supervise_loss) / (2 * params["branches"][0].shape[0])

    return l1_loss + steer_weak_supervise_loss, plotable_params


def l1_attention(params):
    return branched_loss(LF.l1_attention_loss, params)


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
    # calculate loss for each branch with specific activation
    loss_branches_vec, plotable_params = loss_function(params)

    speed_loss = loss_branches_vec[-1]
    return (
        torch.mean(loss_branches_vec[0]) + torch.sum(speed_loss) / params["branches"][0].shape[0],
        plotable_params,
    )


def Loss(loss_name):
    """Factory function

    Note: It is defined with the first letter as uppercase even though is a function to contrast
    the actual use of this function that is making classes
    """

    if loss_name == "L1":
        return l1

    elif loss_name == "L2":
        return l2

    elif loss_name == "l1_steer_weak_supervise":
        return l1_steer_weak_supervise

    else:
        raise ValueError(" Not found Loss name")
