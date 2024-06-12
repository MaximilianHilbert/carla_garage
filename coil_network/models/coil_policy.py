import torch.nn as nn
import torch
import importlib

from .building_blocks import Branching
from .building_blocks import FC
from .building_blocks import Join
from .building_blocks.gru import GRUWaypointsPredictorTransFuser


class CoILPolicy(nn.Module):
    def __init__(self, config):
        super(CoILPolicy, self).__init__()
        self.params = config.model_configuration
        self.config = config
        number_first_layer_channels = 3

        sensor_input_shape = [
            number_first_layer_channels,
            config.camera_height,
            config.camera_width,
        ]

        self.predicted_speed = 0

        if "res" in self.params["perception"]:
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
            raise ValueError("perception type is not-defined")

        self.measurements = FC(
            params={
                "neurons": [len(config.inputs)] + self.params["measurements"]["fc"]["neurons"],
                "dropouts": self.params["measurements"]["fc"]["dropouts"],
                "end_layer": False,
            }
        )

        self.join = Join(
            params={
                "after_process": FC(
                    params={
                        "neurons": [
                            self.params["measurements"]["fc"]["neurons"][-1]
                            + number_output_neurons
                            + self.params["memory_dim"]
                        ]
                        + self.params["join"]["fc"]["neurons"],
                        "dropouts": self.params["join"]["fc"]["dropouts"],
                        "end_layer": False,
                    }
                ),
                "mode": "cat",
            }
        )

        self.speed_branch = FC(
            params={
                "neurons": [number_output_neurons] + self.params["speed_branch"]["fc"]["neurons"] + [1],
                "dropouts": self.params["speed_branch"]["fc"]["dropouts"] + [0.0],
                "end_layer": True,
            }
        )

        branch_fc_vector = []
        for i in range(self.params["branches"]["number_of_branches"]):
            branch_fc_vector.append(
                FC(
                    params={
                        "neurons": [self.params["join"]["fc"]["neurons"][-1]]
                        + self.params["branches"]["fc"]["neurons"]
                        + [config.gru_input_size if config.use_wp_gru else len(config.targets)],
                        "dropouts": self.params["branches"]["fc"]["dropouts"] + [0.0],
                        "end_layer": True,
                    }
                )
            )

        self.branches = Branching(branch_fc_vector)
        if config.use_wp_gru:
            self.gru = GRUWaypointsPredictorTransFuser(config, target_point_size=2)

    def forward(self, x, v, memory, target_point=None):
        if not self.config.rnn_encoding:
            x, _ = self.perception(x)
        else:
            # in this case the input x is not stacked but of shape B, N, C, H, W
            hidden_state= torch.zeros(self.config.num_gru_encoding_layers, x.size(0), self.config.gru_encoding_hidden_size).to(x.device)
            encodings=[]
            for image_index in range(1): #hard coded for policy stream that only encodes 1 frame
                single_frame_encoding,_=self.single_frame_resnet(x[:,image_index,...])
                encodings.append(single_frame_encoding)
            encodings=torch.stack(encodings, dim=0)
            _, hidden_state=self.rnn_single_module(encodings, hidden_state)
            x=hidden_state.squeeze(0)

        m = self.measurements(v).unsqueeze(0)

        m = torch.cat((m, memory), 1)
        j = self.join(x, m)

        branch_outputs = self.branches(j)

        speed_branch_output = self.speed_branch(x)
        if self.config.use_wp_gru:
            waypoints = self.gru.forward(branch_outputs[0], target_point)
        else:
            waypoints = None
        return [waypoints] + [speed_branch_output]

    def forward_branch(self, x, v, memory, target_point):
        output = self.forward(x, v, memory, target_point)
        self.predicted_speed = output[-1]

        return output[:-1]

    def extract_predicted_speed(self):
        # return the speed predicted in forward_branch()
        return self.predicted_speed
