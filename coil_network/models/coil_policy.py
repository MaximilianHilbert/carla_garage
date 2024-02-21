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
        self.config=config
        number_first_layer_channels=3

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

        self.measurements = FC(
                                params={
                                    'neurons': [len(config.inputs)] + self.params['measurements']['fc']['neurons'],
                                    'dropouts': self.params['measurements']['fc']['dropouts'],
                                    'end_layer': False
                                }
                            )


        self.join = Join(
                        params={
                            'after_process':
                                FC(
                                    params={
                                        'neurons':
                                               [self.params['measurements']['fc']['neurons'][-1]
                                                + number_output_neurons
                                                + self.params['memory_dim']] +
                                               self.params['join']['fc']['neurons'],
                                        'dropouts': self.params['join']['fc']['dropouts'],
                                        'end_layer': False
                                    }
                                ),
                            'mode': 'cat'
                        }
                    )


        self.speed_branch = FC(
                                params={
                                    'neurons': [number_output_neurons] + self.params['speed_branch']['fc']['neurons'] + [1],
                                    'dropouts': self.params['speed_branch']['fc']['dropouts'] + [0.0],
                                    'end_layer': True
                                }
                            )


        branch_fc_vector = []
        for i in range(self.params['branches']['number_of_branches']):
            branch_fc_vector.append(
                FC(
                    params={
                        'neurons': [self.params['join']['fc']['neurons'][-1]]
                                    + self.params['branches']['fc']['neurons']
                                    + [config.gru_input_size if config.use_wp_gru else len(config.targets)],
                        'dropouts': self.params['branches']['fc']['dropouts'] + [0.0],
                        'end_layer': True
                    }
                )
            )

        self.branches = Branching(branch_fc_vector)
        if config.use_wp_gru:
            self.gru=GRUWaypointsPredictorTransFuser(config, target_point_size=2)
    def forward(self, x, v, memory, target_point=None):
    
        x, _ = self.perception(x)

        m = self.measurements(v)

        m = torch.cat((m, memory), 1)
        j = self.join(x, m)
            
        branch_outputs = self.branches(j)
        
        speed_branch_output = self.speed_branch(x)
        if self.config.use_wp_gru:
            waypoints=self.gru.forward(self.branch_outputs + [speed_branch_output], target_point)
        else:
            waypoints=None
        return branch_outputs + [speed_branch_output], waypoints

    def forward_branch(self, x, v, branch_number, memory):
    
        output = self.forward(x, v, memory)
        self.predicted_speed = output[-1]
        control = output[0:6]
        output_vec = torch.stack(control)

        return self.extract_branch(output_vec, branch_number)

    def extract_branch(self, output_vec, branch_number):

        #branch_number = command_number_to_index(branch_number)

        if len(branch_number) > 1:
            branch_number = torch.squeeze(branch_number.type(torch.cuda.LongTensor))
        else:
            branch_number = branch_number.type(torch.cuda.LongTensor)

        branch_number = torch.stack([branch_number,
                                     torch.cuda.LongTensor(range(0, len(branch_number)))])

        return output_vec[branch_number[0], branch_number[1], :]

    def extract_predicted_speed(self):
        # return the speed predicted in forward_branch()
        return self.predicted_speed