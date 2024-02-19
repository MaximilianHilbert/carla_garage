import torch.nn as nn
import torch
import importlib


from .building_blocks import Branching
from .building_blocks import FC

class CoILMemExtract(nn.Module):

    def __init__(self, config):

        super(CoILMemExtract, self).__init__()
        self.params = config.mem_extract_model_configuration

        number_first_layer_channels=3*(config.img_seq_len-1)

        sensor_input_shape = [number_first_layer_channels,
    config.camera_height,config.camera_width]
        self.predicted_speed = 0

        if 'res' in self.params['perception']:
            resnet_module = importlib.import_module('coil_network.models.building_blocks.resnet')
            resnet_module = getattr(resnet_module, self.params['perception']['res']['name'])
            self.perception = resnet_module(
                                    pretrained=config.pre_trained,
                                    input_channels=number_first_layer_channels,
                                    num_classes=self.params['perception']['res']['num_classes']
                                )

            number_output_neurons = self.params['perception']['res']['num_classes']
                
        else:
            raise ValueError("perception type is not-defined")


        self.speed_branch = FC(
                                params={
                                    'neurons': [number_output_neurons] 
                                                + self.params['speed_branch']['fc']['neurons'] + [1],
                                    'dropouts': self.params['speed_branch']['fc']['dropouts'] + [0.0],
                                    'end_layer': True
                                }
                            )

        branch_fc_vector = []
        for i in range(self.params['branches']['number_of_branches']):
            branch_fc_vector.append(
                FC(
                    params={
                        'neurons': [number_output_neurons]
                                    + self.params['branches']['fc']['neurons'] 
                                    + [len(config.targets)],
                        'dropouts': self.params['branches']['fc']['dropouts'] + [0.0],
                        'end_layer': True
                    }
                )
            )
                        
        self.branches = Branching(branch_fc_vector)
        
        
    def forward(self, x):

        x, _ = self.perception(x)

        speed_branch_output = self.speed_branch(x)
        
        branch_outputs = self.branches(x)

        return branch_outputs + [speed_branch_output], x.detach()
