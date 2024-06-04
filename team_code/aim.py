"""
Implements the AIM vision backbone.
"""
import torch
from torch import nn
import timm
import transfuser_utils as t_u


class AIMBackbone(nn.Module):
    """
    Processes an image with an ImageNet architecture and returns features (grid).
    """

    def __init__(self, config, channels):
        super().__init__()
        self.config = config

        self.image_encoder = timm.create_model(config.image_architecture, pretrained=True, features_only=True, in_chans=channels)

        
        start_index = 0
        # Some networks have a stem layer
        if len(self.image_encoder.return_layers) > 4:
            start_index += 1

        self.num_image_features = self.image_encoder.feature_info.info[start_index + 3]["num_chs"]
        self.num_features = self.image_encoder.feature_info.info[start_index + 3]["num_chs"]

    def forward(self, image):
        """standard forward pass"""
        if self.config.normalize_imagenet:
            image_features = t_u.normalize_imagenet(image)
        else:
            image_features = image

        # Generate an iterator for all the layers in the network that one can loop through.
        image_layers = iter(self.image_encoder.items())

        # Stem layer.
        # In some architectures the stem is not a return layer, so we need to skip it.
        if len(self.image_encoder.return_layers) > 4:
            image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)

        # Loop through the 4 blocks of the network.
        for _ in range(4):
            image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)

   

        return image_features

    def forward_layer_block(self, layers, return_layers, features):
        """
        Run one forward pass to a block of layers from a TIMM neural network and returns the result.
        Advances the whole network by just one block
        :param layers: Iterator starting at the current layer block
        :param return_layers: TIMM dictionary describing at which intermediate layers features are returned.
        :param features: Input features
        :return: Processed features
        """
        for name, module in layers:
            features = module(features)
            if name in return_layers:
                break
        return features
