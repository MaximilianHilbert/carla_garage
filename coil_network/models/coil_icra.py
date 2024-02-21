import torch.nn as nn
import torch
import importlib

from .building_blocks.gru import GRUWaypointsPredictorTransFuser
from .building_blocks.conv import Conv
from .building_blocks import Branching
from .building_blocks import FC
from .building_blocks import Join
class CoILICRA(nn.Module):

    def __init__(self, config):
        # TODO: Improve the model autonaming function

        super(CoILICRA, self).__init__()
        self.params = config.model_configuration
        self.config=config
        number_first_layer_channels = 0

        number_first_layer_channels=3*config.img_seq_len    #3 color channels img_seq_len could be != 1, for instance in bcoh, keyframes baseline

        sensor_input_shape = [number_first_layer_channels,
    config.camera_height,config.camera_width]

        # For this case we check if the perception layer is of the type "conv"
        if 'conv' in self.params['perception']:
            perception_convs = Conv(params={'channels': [number_first_layer_channels] +
                                                          self.params['perception']['conv']['channels'],
                                            'kernels': self.params['perception']['conv']['kernels'],
                                            'strides': self.params['perception']['conv']['strides'],
                                            'dropouts': self.params['perception']['conv']['dropouts'],
                                            'end_layer': True})

            perception_fc = FC(params={'neurons': [perception_convs.get_conv_output(sensor_input_shape)]
                                                  + self.params['perception']['fc']['neurons'],
                                       'dropouts': self.params['perception']['fc']['dropouts'],
                                       'end_layer': False})

            self.perception = nn.Sequential(*[perception_convs, perception_fc])

            number_output_neurons = self.params['perception']['fc']['neurons'][-1]

        elif 'res' in self.params['perception']:  # pre defined residual networks
            resnet_module = importlib.import_module('coil_network.models.building_blocks.resnet')
            resnet_module = getattr(resnet_module, self.params['perception']['res']['name'])
            self.perception = resnet_module(pretrained=config.pre_trained,
                                            input_channels=number_first_layer_channels,
                                             num_classes=self.params['perception']['res']['num_classes'])

            number_output_neurons = self.params['perception']['res']['num_classes']

        else:

            raise ValueError("invalid convolution layer type")
        self.measurements = FC(params={'neurons': [len(config.inputs)] +
                                                self.params['measurements']['fc']['neurons'],
                                    'dropouts': self.params['measurements']['fc']['dropouts'],
                                    'end_layer': False})
        if 'previous_actions' in self.params:
            self.use_previous_actions = True
            self.previous_actions = FC(params={'neurons': [len(config.targets)*config.number_previous_actions] +
                                                          self.params['previous_actions']['fc']['neurons'],
                                               'dropouts': self.params['previous_actions']['fc']['dropouts'],
                                               'end_layer': False})
            number_preaction_neurons = self.params['previous_actions']['fc']['neurons'][-1]
        else:
            self.use_previous_actions = False
            number_preaction_neurons = 0

        self.join = Join(
            params={'after_process':
                        FC(params={'neurons':
                                        [self.params['measurements']['fc']['neurons'][-1] +
                                        + number_preaction_neurons + number_output_neurons] +
                                        self.params['join']['fc']['neurons'],
                                    'dropouts': self.params['join']['fc']['dropouts'],
                                    'end_layer': False}),
                    'mode': 'cat'
                    }
        )

        self.speed_branch = FC(params={'neurons': [number_output_neurons] +
                                                  self.params['speed_branch']['fc']['neurons'] + [1],
                                       'dropouts': self.params['speed_branch']['fc']['dropouts'] + [0.0],
                                       'end_layer': True})

        # Create the fc vector separatedely
        branch_fc_vector = []
        for i in range(self.params['branches']['number_of_branches']):
            branch_fc_vector.append(FC(params={'neurons': [self.params['join']['fc']['neurons'][-1]] +
                                                         self.params['branches']['fc']['neurons'] 
                                                         + [config.gru_input_size if config.use_wp_gru else len(config.targets)],
                                               'dropouts': self.params['branches']['fc']['dropouts'] + [0.0],
                                               'end_layer': True}))

        self.branches = Branching(branch_fc_vector)  # Here we set branching automatically
        if config.use_wp_gru:
            self.gru=GRUWaypointsPredictorTransFuser(config, target_point_size=2)
        if 'conv' in self.params['perception']:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.1)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.1)


    def forward(self, x, a, pa=None,target_point=None):
        """ ###### APPLY THE PERCEPTION MODULE """
        x, inter = self.perception(x)
        ## Not a variable, just to store intermediate layers for future vizualization
        #self.intermediate_layers = inter
        """ ###### APPLY THE MEASUREMENT MODULE """
        if self.measurements is not None:
            m = self.measurements(a)
        else:
            m = None
        
        """ ###### APPLY THE PREVIOUS ACTIONS MODULE, IF THIS MODULE EXISTS"""
        if self.use_previous_actions and m is not None:
            n = self.previous_actions(pa)
            m = torch.cat((m, n), 1)

        """ Join measurements and perception"""
        if self.join is not None and m is not None:
            j = self.join(x, m)
        else:
            j = x
        branch_outputs = self.branches(j)
        speed_branch_output = self.speed_branch(x)
        # We concatenate speed with the rest.
        if self.config.use_wp_gru:
            waypoints_branched=[]
            for single_branch in branch_outputs:
                waypoints_branched.append(self.gru.forward(single_branch, target_point))
            return waypoints_branched+ [speed_branch_output]
        else:
            return branch_outputs+ [speed_branch_output]
    def forward_branch(self, x, a, branch_number, pa=None):
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
        # Convert to integer just in case .
        # TODO: take four branches, this is hardcoded
        output = self.forward(x, a, pa)
        self.predicted_speed = output[-1]
        control = output[0:6]
        output_vec = torch.stack(control)

        return self.extract_branch(output_vec, branch_number)

    def get_perception_layers(self, x):
        return self.perception.get_layers_features(x)

    def extract_branch(self, output_vec, branch_number):


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
