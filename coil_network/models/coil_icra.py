import torch.nn as nn
import torch
import importlib

from .building_blocks.gru import GRUWaypointsPredictorTransFuser
from .building_blocks.conv import Conv
from .building_blocks import Branching
from .building_blocks import FC
from .building_blocks import Join
from team_code.transfuser import (
    TransformerDecoderLayerWithAttention,
    TransformerDecoderWithAttention,
)
from team_code.model import PositionEmbeddingSine

class CoILICRA(nn.Module):
    def __init__(self, config):
        super(CoILICRA, self).__init__()
        self.params = config.model_configuration
        self.config = config
        number_first_layer_channels = 0

        number_first_layer_channels = (
            3 * config.img_seq_len
        )  # 3 color channels img_seq_len could be != 1, for instance in bcoh, keyframes baseline

        sensor_input_shape = [
            number_first_layer_channels,
            config.camera_height,
            config.camera_width,
        ]

        # For this case we check if the perception layer is of the type "conv"
        if "conv" in self.params["perception"]:
            perception_convs = Conv(
                params={
                    "channels": [number_first_layer_channels] + self.params["perception"]["conv"]["channels"],
                    "kernels": self.params["perception"]["conv"]["kernels"],
                    "strides": self.params["perception"]["conv"]["strides"],
                    "dropouts": self.params["perception"]["conv"]["dropouts"],
                    "end_layer": True,
                }
            )

            perception_fc = FC(
                params={
                    "neurons": [perception_convs.get_conv_output(sensor_input_shape)]
                    + self.params["perception"]["fc"]["neurons"],
                    "dropouts": self.params["perception"]["fc"]["dropouts"],
                    "end_layer": False,
                }
            )

            self.perception = nn.Sequential(*[perception_convs, perception_fc])

            number_output_neurons = self.params["perception"]["fc"]["neurons"][-1]

        elif "res" in self.params["perception"]:  # pre defined residual networks
            resnet_module = importlib.import_module("coil_network.models.building_blocks.resnet")
            resnet_module = getattr(resnet_module, self.params["perception"]["res"]["name"])
            self.perception = resnet_module(
                pretrained=config.pre_trained,
                input_channels=number_first_layer_channels,
                num_classes=self.params["perception"]["res"]["num_classes"],
            )

            number_output_neurons = self.params["perception"]["res"]["num_classes"]
        elif "res_plus_rnn" in self.params["perception"]:
            number_first_layer_channels=3 #only a single image, no frame stacking
            resnet_module = importlib.import_module("coil_network.models.building_blocks.resnet")
            resnet_module = getattr(resnet_module, self.params["perception"]["res_plus_rnn"]["name"])
            self.single_frame_resnet=resnet_module(
                pretrained=config.pre_trained,
                input_channels=number_first_layer_channels,
                num_classes=self.params["perception"]["res_plus_rnn"]["num_classes"],
            )
            
            self.rnn_single_module=nn.GRU(input_size=self.params["perception"]["res_plus_rnn"]["num_classes"], 
                                              hidden_size=self.config.gru_encoding_hidden_size, num_layers=self.config.num_gru_encoding_layers)
            number_output_neurons = self.config.gru_encoding_hidden_size
        else:
            raise ValueError("invalid convolution layer type")
        self.measurements = FC(
            params={
                "neurons": [len(config.inputs)] + self.params["measurements"]["fc"]["neurons"],
                "dropouts": self.params["measurements"]["fc"]["dropouts"],
                "end_layer": False,
            }
        )
        if "previous_actions" in self.params:
            self.use_previous_actions = True
            self.previous_actions = FC(
                params={
                    "neurons": [len(config.targets) * config.number_previous_actions]
                    + self.params["previous_actions"]["fc"]["neurons"],
                    "dropouts": self.params["previous_actions"]["fc"]["dropouts"],
                    "end_layer": False,
                }
            )
            number_preaction_neurons = self.params["previous_actions"]["fc"]["neurons"][-1]
        else:
            self.use_previous_actions = False
            number_preaction_neurons = 0
        if self.config.transformer_decoder:
            self.wp_query = nn.Parameter(
                            torch.zeros(
                                1,
                                self.config.gru_input_size,
                            )
                        )
            decoder_norm=nn.LayerNorm(self.config.gru_hidden_size)
            decoder_layer = nn.TransformerDecoderLayer(
                        self.config.gru_input_size,
                        self.config.num_decoder_heads,
                        activation=nn.GELU(),
                        batch_first=True,
                    )
            self.join = torch.nn.TransformerDecoder(
                        decoder_layer,
                        num_layers=self.config.num_transformer_decoder_layers,
                        norm=decoder_norm,
                    )
            self.encoder_pos_encoding = PositionEmbeddingSine(self.config.gru_input_size // 2, normalize=True)
            self.extra_sensor_pos_embed = nn.Parameter(torch.zeros(1, self.config.gru_input_size))
            #we use that to get down from 512 channels (resnet output) to 64 channels, used for waypoint gru and measurement encoding
            self.change_channel = nn.Conv1d(
                    self.params["perception"]["res"]["num_classes"],
                    self.config.gru_input_size,
                    kernel_size=1,
                )
        else:
            self.join = Join(
                params={
                    "after_process": FC(
                        params={
                            "neurons": [
                                self.params["measurements"]["fc"]["neurons"][-1]
                                + +number_preaction_neurons
                                + number_output_neurons
                            ]
                            + self.params["join"]["fc"]["neurons"],
                            "dropouts": self.params["join"]["fc"]["dropouts"],
                            "end_layer": False,
                        }
                    ),
                    "mode": "cat",
                }
            )
            # Create the fc vector separatedely
            self.branch_fc_vector=FC(
                    params={
                        "neurons": [self.params["join"]["fc"]["neurons"][-1]]
                        + self.params["branches"]["fc"]["neurons"]
                        + [config.gru_input_size if config.use_wp_gru else len(config.targets)],
                        "dropouts": self.params["branches"]["fc"]["dropouts"] + [0.0],
                        "end_layer": False,
                    }
                )
                


        if config.use_wp_gru:
            self.gru = GRUWaypointsPredictorTransFuser(config, target_point_size=2)
        if "conv" in self.params["perception"]:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.1)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x, a, target_point=None, pa=None):
        bs=x.shape[0] #batch size
        """###### APPLY THE PERCEPTION MODULE"""
        if not self.config.rnn_encoding:
            x, inter = self.perception(x)
        else:
            # in this case the input x is not stacked but of shape B, N, C, H, W
            hidden_state= torch.zeros(self.config.num_gru_encoding_layers, x.size(0), self.config.gru_encoding_hidden_size).to(x.device)
            encodings=[]
            for image_index in range(self.config.img_seq_len):
                single_frame_encoding,_=self.single_frame_resnet(x[:,image_index,...])
                encodings.append(single_frame_encoding)
            encodings=torch.stack(encodings, dim=0)
            _, hidden_state=self.rnn_single_module(encodings, hidden_state)
            x=hidden_state.squeeze(0)
        ## Not a variable, just to store intermediate layers for future vizualization
        # self.intermediate_layers = inter
        """ ###### APPLY THE MEASUREMENT MODULE """
        if self.measurements is not None:
            m = self.measurements(a)
        else:
            m = None
        # wir nehmen x (image branch) adden m als token (measurement encoding) -> gefused input -> positional encoding -> keys, values erzeugt über linear layers -> queries kommen von außen und werden gelernt -> attention -> output (länge über queries
        #definiert) -> input zu GRU
        """ ###### APPLY THE PREVIOUS ACTIONS MODULE, IF THIS MODULE EXISTS"""
        if self.use_previous_actions and m is not None:
            n = self.previous_actions(pa)
            m = torch.cat((m, n), 1)

        if self.config.transformer_decoder:
            #now concat image features and measurement features (consisting of maybe velocity and prev wp predictions)
            x=self.change_channel(x.unsqueeze(2))
            x=x.reshape(bs,-1,int(self.config.gru_input_size**0.5), int(self.config.gru_input_size**0.5))
            x=x.expand(-1, self.config.gru_input_size, -1 , -1)
            pos_enc=self.encoder_pos_encoding(x)
            x=x+pos_enc
            x=torch.flatten(x, start_dim=2)
            if m is not None:
                m=m+self.extra_sensor_pos_embed.repeat(bs, 1)
                m=m.unsqueeze(2)
                x = torch.cat((x, m), axis=2)
            x = torch.permute(x, (0, 2, 1))
            branch_outputs = self.join(self.wp_query.repeat(bs, 1, 1), x)
        else:
            """ Join measurements and perception"""
            if self.join is not None and m is not None:
                j = self.join(x, m)
            else:
                j = x
            branch_outputs = self.branch_fc_vector(j)
        
        return self.gru.forward(branch_outputs.squeeze(1), target_point)

    def forward_branch(self, x, a, target_point=None, pa=None):
        """
        DO a forward operation and return a single branch.

        Args:
            x: the image input
            a: speed measurement
            branch_number: the branch number to be returned
            pa: previous actions, optional

        Returns:
            the forward operation on the selected branch

        """
       
        return self.forward(x, a, target_point, pa)

    def get_perception_layers(self, x):
        return self.perception.get_layers_features(x)

    def extract_predicted_speed(self):
        # return the speed predicted in forward_branch()
        return self.predicted_speed
