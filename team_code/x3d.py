""" x3d Video Backbone"""

from torch import nn
import torch

class X3D(nn.Module):
    """x3d for feature extraction
    We adapt the code here so that it matches the structure of timm models and we can interchange them more easily.
    """

    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model=torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=pretrained)
        # Remove layers that we don't need
        self.model = torch.nn.Sequential(*list(self.model.blocks)[:-1])
        #input_tensor = torch.randn(1, 3, 13, 160, 160)  # You may need to adjust dimensions as needed
        self.output_channels=192
        # Forward pass through the modified model to get the features
        # with torch.no_grad():
        #     features = self.model(input_tensor)
        # print(features.shape)

    # Return the iterator we will use to loop through the network.
    def items(self):
        return self.model.named_children()
