"""
Center Net Head implementation adapted from MM Detection
"""

import team_code.transfuser_utils as t_u
import numpy as np
import torch
from torch import nn
import team_code.gaussian_target as g_t


class LidarCenterNetHead(nn.Module):
    """
    Objects as Points Head. CenterHead use center_point to indicate object's position.
    Paper link <https://arxiv.org/abs/1904.07850>
    Args:
        config: Gobal TransFuser config.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.heatmap_head = self._build_head(config.bb_input_channel, config.num_bb_classes)
        self.wh_head = self._build_head(config.bb_input_channel, 2)
        self.offset_head = self._build_head(config.bb_input_channel, 2)
        self.yaw_class_head = self._build_head(config.bb_input_channel, config.num_dir_bins)
        self.yaw_res_head = self._build_head(config.bb_input_channel, 1)
        if config.velocity_brake_prediction:
            self.velocity_head = self._build_head(config.bb_input_channel, 1)
            self.brake_head = self._build_head(config.bb_input_channel, 2)
        if config.predict_vectors:
            self.velocity_vector_head = self._build_head(config.bb_input_channel, 2)
            self.acceleration_vector_head = self._build_head(config.bb_input_channel, 2)
        # We use none reduction because we weight each pixel according to the number of bounding boxes.
        self.loss_center_heatmap = t_u.gaussian_focal_loss
        self.loss_wh = nn.L1Loss(reduction="none")
        self.loss_offset = nn.L1Loss(reduction="none")
        self.loss_dir_class = nn.CrossEntropyLoss(reduction="none")
        self.loss_dir_res = nn.SmoothL1Loss(reduction="none")
        if self.config.velocity_brake_prediction:
            self.loss_velocity = nn.L1Loss(reduction="none")
            self.loss_brake = nn.CrossEntropyLoss(reduction="none")
        if self.config.predict_vectors:
            self.loss_velocity_vector = nn.L1Loss(reduction="none")
            self.loss_acceleration_vector = nn.L1Loss(reduction="none")
    def _build_head(self, in_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
        )
        return layer

    def forward(self, feat):
        """
        Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the channels number is num_classes.
            wh_pred (Tensor): wh predicts, the channels number is 2.
            offset_pred (Tensor): offset predicts, the channels number is 2.
        """
        return_dict={}
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        yaw_class_pred = self.yaw_class_head(feat)
        yaw_res_pred = self.yaw_res_head(feat)
        return_dict["wh_pred"]=wh_pred
        return_dict["offset_pred"]=offset_pred
        return_dict["yaw_class_pred"]=yaw_class_pred
        return_dict["yaw_res_pred"]=yaw_res_pred
        return_dict["center_heatmap_pred"]=center_heatmap_pred

        if self.config.velocity_brake_prediction:
            velocity_pred = self.velocity_head(feat)
            brake_pred = self.brake_head(feat)
            return_dict["velocity_pred"]=velocity_pred
            return_dict["brake_pred"]=brake_pred
        if self.config.predict_vectors:
            velocity_vector_pred = self.velocity_vector_head(feat)
            acceleration_vector_pred=self.acceleration_vector_head(feat)
            return_dict["velocity_vector_pred"]=velocity_vector_pred
            return_dict["acceleration_vector_pred"]=acceleration_vector_pred
        return return_dict

    def loss(
        self,
        predictions,
        targets
    ):
        """
        Compute losses of the head.

        Args:
            center_heatmap_preds (Tensor): center predict heatmaps for all levels with shape (B, num_classes, H, W).
            wh_preds (Tensor): wh predicts for all levels with shape (B, 2, H, W).
            offset_preds (Tensor): offset predicts for all levels with shape (B, 2, H, W).

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
        """
        # The number of valid bounding boxes can vary.
        # The avg factor represents the amount of valid bounding boxes in the batch.
        # We don't want the empty bounding boxes to have an impact therefore we use reduction sum and divide by the actual
        # number of bounding boxes instead of using standard mean reduction.
        # The weight sets all pixels without a bounding box to 0.
        # Add small epsilon to have numerical stability in the case where there are no boxes in the batch.
        
        avg_factor = targets["avg_factor"].sum()
        avg_factor = avg_factor + torch.finfo(torch.float32).eps
        loss_center_heatmap = (
            self.loss_center_heatmap(predictions["center_heatmap_pred"], targets["center_heatmap_target"], reduction="sum") / avg_factor
        )
        # The avg factor is multiplied by the number of channels to yield a proper mean.
        # For the other predictions this value is 1 so it is omitted.
        loss_wh = (self.loss_wh(predictions["wh_pred"], targets["wh_target"]) * targets["pixel_weight"]).sum() / (avg_factor * predictions["wh_pred"].shape[1])
        loss_offset = (self.loss_offset(predictions["offset_pred"], targets["offset_target"]) * targets["pixel_weight"]).sum() / (
            avg_factor * predictions["wh_pred"].shape[1]
        )
        loss_yaw_class = (self.loss_dir_class(predictions["yaw_class_pred"], targets["yaw_class_target"]) * targets["pixel_weight"][:, 0]).sum() / avg_factor
        loss_yaw_res = (self.loss_dir_res(predictions["yaw_res_pred"], targets["yaw_res_target"]) * targets["pixel_weight"][:, 0:1]).sum() / avg_factor

        losses = dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset,
            loss_yaw_class=loss_yaw_class,
            loss_yaw_res=loss_yaw_res,
        )
        if self.config.velocity_brake_prediction:
            loss_velocity = (
                self.loss_velocity(predictions["velocity_pred"], targets["velocity_target"]) * targets["pixel_weight"][:, 0:1]
            ).sum() / avg_factor
            loss_brake = (self.loss_brake(predictions["brake_pred"], targets["brake_target"]) * targets["pixel_weight"][:, 0]).sum() / avg_factor
            losses["loss_velocity"] = loss_velocity
            losses["loss_brake"] = loss_brake
        if self.config.predict_vectors:
            loss_velocity_vector = (
                self.loss_velocity_vector(predictions["velocity_vector_pred"], targets["velocity_vector_target"]) * targets["pixel_weight"][:, 0:1]
            ).sum() / (avg_factor*predictions["velocity_vector_pred"].shape[1])
            loss_acceleration_vector=(
                self.loss_acceleration_vector(predictions["acceleration_vector_pred"], targets["acceleration_vector_target"]) * targets["pixel_weight"][:, 0:1]
            ).sum() / (avg_factor*predictions["acceleration_vector_pred"].shape[1])
            losses["loss_velocity_vector"] = loss_velocity_vector
            losses["loss_acceleration_vector"] = loss_acceleration_vector
        return losses

    def class2angle(self, angle_cls, angle_res, limit_period=True):
        """
        Inverse function to angle2class.
        Args:
            angle_cls (torch.Tensor): Angle class to decode.
            angle_res (torch.Tensor): Angle residual to decode.
            limit_period (bool): Whether to limit angle to [-pi, pi].
        Returns:
            torch.Tensor: Angle decoded from angle_cls and angle_res.
        """
        angle_per_class = 2 * np.pi / float(self.config.num_dir_bins)
        angle_center = angle_cls.float() * angle_per_class
        angle = angle_center + angle_res
        if limit_period:
            angle[angle > np.pi] -= 2 * np.pi
        return angle

    def get_bboxes(
        self,
        predictions
    ):
        """
        Transform network output for a batch into bbox predictions.
        Bounding boxes are still in image coordinates.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): Extent predicts for all levels with shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels with shape (B, 2, H, W).

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 8) tensor, where 8 represent
                (tl_x, tl_y, br_x, br_y, yaw, speed, brake, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding box.
        """
        batch_det_bboxes, velocity_vectors, acceleration_vectors = self.decode_heatmap(
            predictions,
            k=self.config.top_k_center_keypoints,
            kernel=self.config.center_net_max_pooling_kernel,
        )

        return batch_det_bboxes,velocity_vectors, acceleration_vectors

    def decode_heatmap(
        self,
        predictions,
        k=100,
        kernel=3,
    ):
        """
        Transform outputs into detections raw bbox prediction.
        Bounding boxes are still in image coordinates.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap, shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels. Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing the following Tensors:
              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 8)
        """
        img_h = self.config.lidar_resolution_height
        img_w = self.config.lidar_resolution_width
        _, _, feat_h, feat_w = predictions["center_heatmap_pred"].shape

        height_ratio = float(img_h / feat_h)
        width_ratio = float(img_w / feat_w)

        center_heatmap_pred = g_t.get_local_maximum(predictions["center_heatmap_pred"], kernel=kernel)

        (
            batch_scores,
            batch_index,
            batch_topk_classes,
            topk_ys,
            topk_xs,
        ) = g_t.get_topk_from_heatmap(center_heatmap_pred, k=k)

        wh = g_t.transpose_and_gather_feat(predictions["wh_pred"], batch_index)
        offset = g_t.transpose_and_gather_feat(predictions["offset_pred"], batch_index)
        yaw_class = g_t.transpose_and_gather_feat(predictions["yaw_class_pred"], batch_index)
        yaw_res = g_t.transpose_and_gather_feat(predictions["yaw_res_pred"], batch_index)

        # convert class + res to yaw
        yaw_class = torch.argmax(yaw_class, -1)
        yaw = self.class2angle(yaw_class, yaw_res.squeeze(2))
        if self.config.predict_vectors:
            velocity_vectors=g_t.transpose_and_gather_feat(predictions["velocity_vector_pred"], batch_index)
            velocity_vectors= torch.cat((velocity_vectors,batch_scores[..., np.newaxis],),dim=-1,)
            acceleration_vectors=g_t.transpose_and_gather_feat(predictions["acceleration_vector_pred"], batch_index)
            acceleration_vectors= torch.cat((acceleration_vectors,batch_scores[..., np.newaxis],),dim=-1,)
        else:
            velocity_vectors=torch.zeros((batch_index.shape[0],batch_index.shape[1], 2))
            acceleration_vectors=torch.zeros((batch_index.shape[0],batch_index.shape[1], 2))
        if self.config.velocity_brake_prediction:
            velocity = g_t.transpose_and_gather_feat(predictions["velocity_pred"], batch_index)
            brake = g_t.transpose_and_gather_feat(predictions["brake_pred"], batch_index)
            brake = torch.argmax(brake, -1)
            velocity = velocity[..., 0]
        else:
            velocity=torch.zeros_like(yaw)
            brake=torch.zeros_like(yaw)
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]

        batch_bboxes = torch.stack([topk_xs, topk_ys, wh[..., 0], wh[..., 1], yaw, velocity, brake], dim=2)
        batch_bboxes = torch.cat(
            (
                batch_bboxes,
                batch_topk_classes[..., np.newaxis],
                batch_scores[..., np.newaxis],
            ),
            dim=-1,
        )
        batch_bboxes[:, :, 0] *= width_ratio
        batch_bboxes[:, :, 1] *= height_ratio
        batch_bboxes[:, :, 2] *= width_ratio
        batch_bboxes[:, :, 3] *= height_ratio

        return batch_bboxes, velocity_vectors.detach().cpu().numpy(),acceleration_vectors.detach().cpu().numpy()


def angle2class(angle, num_dir_bins):
    """
    Convert continuous angle to a discrete class and a small regression number from class center angle to current angle.
    Args:
        angle (float): Angle is from 0-2pi (or -pi~pi),
          class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N).
    Returns:
        tuple: Encoded discrete class and residual.
    """
    angle = angle % (2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_dir_bins)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    angle_cls = shifted_angle // angle_per_class
    angle_res = shifted_angle - (angle_cls * angle_per_class + angle_per_class / 2)
    return int(angle_cls), angle_res
