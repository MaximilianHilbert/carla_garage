import torch.nn as nn
import torch
import importlib

from .building_blocks.gru import GRUWaypointsPredictorTransFuser
from .building_blocks import FC
from .building_blocks import Join

from team_code.model import PositionEmbeddingSine

class CoILICRA(nn.Module):
    def __init__(self, name, config):
        super(CoILICRA, self).__init__()
        self.params = config.model_configuration
        self.config = config
        self.name=name
        if self.config.backbone_type=="rnn" or self.name=="coil-policy":
            number_first_layer_channels=3 #only one frame per rnn input
        elif self.name=="coil-memory":
            #-1 because we encode current frame and history differently
            number_first_layer_channels = 3 * (config.img_seq_len-1)
        #if we want to stack everything like in keyframes, or bcoh or bcso (img_seq_len is then 1 anyways)
        else:
            number_first_layer_channels=3*config.img_seq_len
        resnet_module = importlib.import_module("coil_network.models.building_blocks.resnet")
        resnet_module = getattr(resnet_module, self.config.resnet_type)


        number_output_neurons = self.config.resnet_output_feat_dim
        if self.config.backbone_type=="rnn":
            self.single_frame_resnet=resnet_module(
                pretrained=config.pre_trained,
                input_channels=number_first_layer_channels,
                num_classes=number_output_neurons,
            )
            
            self.rnn_single_module=nn.GRU(input_size=number_output_neurons, 
                                              hidden_size=self.config.gru_encoding_hidden_size, num_layers=self.config.num_gru_encoding_layers)
            number_output_neurons = self.config.gru_encoding_hidden_size
        else:
            self.perception = resnet_module(
                pretrained=self.config.pre_trained,
                input_channels=number_first_layer_channels,
                num_classes=number_output_neurons,
            )
        if self.config.speed_input:
            self.measurements = FC(
                params={
                    "neurons": [len(config.inputs)] + self.config.measurement_layers+[self.config.additional_inputs_output_size],
                    "dropouts": self.config.measurement_dropouts,
                    "end_layer": False,
                }
            )
        if self.config.prevnum>0:
            self.previous_wp = FC(
                params={
                    #-1 because we define n-1 as previous wp in the dataloader
                    "neurons": [self.config.target_point_size*self.config.prevnum]
                    + self.config.previous_waypoints_layers+[self.config.additional_inputs_output_size],
                    "dropouts": self.config.previous_waypoints_dropouts,
                    "end_layer": False,
                }
            )

        if self.config.transformer_decoder:
            #we use that to determine the necessary token length, depending on additional inputs into the transformer
            self.multiplier_for_embedding_length=sum([self.config.prevnum>0, self.config.speed_input==1])
            self.wp_query = nn.Parameter(
                            torch.zeros(
                                1,
                                self.config.gru_input_size,
                            )
                        )
            #we use that to get down from 512 channels (resnet output) to 64 channels, used for waypoint gru and measurement encoding
            self.change_channel = nn.Conv1d(
                    self.config.resnet_output_feat_dim,
                    self.config.gru_input_size,
                    kernel_size=1,
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
            self.encoder_pos_encoding = PositionEmbeddingSine(self.config.embedding_size_transformer_decoder//2, normalize=True) #first argument is embedding size
            if self.config.prevnum>0 or self.config.speed_input:
                self.extra_sensor_pos_embed = nn.Parameter(torch.zeros(1, self.config.additional_inputs_output_size*self.multiplier_for_embedding_length))
            
        else:
            measurement_contribution=self.config.additional_inputs_output_size if self.config.speed_input else 0
            previous_wp_contribution=self.config.additional_inputs_output_size if self.config.prevnum>0 else 0
            memory_contribution=self.config.memory_dim if self.name=="coil-policy" else 0
            backbone_contribution=number_output_neurons
            self.join = Join(
                params={
                    "after_process": FC(
                        params={
                            "neurons": [
                                measurement_contribution+previous_wp_contribution+backbone_contribution+memory_contribution
                            ]
                            + self.config.join_layers,
                            "dropouts": self.config.join_layer_dropouts,
                            "end_layer": False,
                        }
                    ),
                    "mode": "cat",
                }
            )
            # Create the fc vector separatedely
            self.fc_vector=FC(
                    params={
                        "neurons": [self.config.join_layers[-1]]
                        + self.config.fc_layers
                        + [config.gru_input_size],
                        "dropouts": self.config.fc_layer_dropouts + [0.0],
                        "end_layer": False,
                    }
                )

        self.wp_gru = GRUWaypointsPredictorTransFuser(config, target_point_size=self.config.target_point_size)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x, speed=None, target_point=None, prev_wp=None, memory_to_fuse=None):
        bs=x.shape[0] #batch size
        if self.config.backbone_type=="stacking":
            x, _ = self.perception(x)
        else:
            # in this case the input x is not stacked but of shape B, N, C, H, W
            hidden_state= torch.zeros(self.config.num_gru_encoding_layers, x.size(0), self.config.gru_encoding_hidden_size).to(x.device)
            encodings=[]
            
            if self.name=="coil-policy":
                length_current_images=self.config.img_seq_len-(self.config.img_seq_len-1)
            elif self.name=="coil-memory":
                length_current_images=self.config.img_seq_len-1 
            elif self.name=="coil-icra":
                length_current_images=1
            for image_index in range(length_current_images):
                #this is done to save gpu memory for gradients
                if image_index%2==0:
                    self.single_frame_resnet.eval()
                else:
                    self.single_frame_resnet.train()
                single_frame_encoding,_=self.single_frame_resnet(x[:,image_index,...])
                encodings.append(single_frame_encoding)
            encodings=torch.stack(encodings, dim=0)
            _, hidden_state=self.rnn_single_module(encodings, hidden_state)
            x=hidden_state.squeeze(0)
        if self.name=="coil-memory":
            #we copy the encoded image branch to return it as memory for arp
            generated_memory=x
        else:
            generated_memory=None
        if self.config.speed_input:
            measurement_enc = self.measurements(speed)
        else:
            measurement_enc = None
        if self.config.prevnum>0:
            prev_wp_enc = self.previous_wp(prev_wp)
        else:
            prev_wp_enc=None
        additional_inputs=[input for input in [memory_to_fuse, measurement_enc, prev_wp_enc] if input is not None]
        if additional_inputs:
            additional_inputs=torch.cat(additional_inputs, dim=1)
        else:
            additional_inputs=torch.empty(0)


        #eventually decode via transformer decoder
        if self.config.transformer_decoder:
            x=self.change_channel(x.unsqueeze(2))
            x=x.reshape(bs,-1,int(self.config.gru_input_size**0.5), int(self.config.gru_input_size**0.5))
            #changes to 128 features
            pos_enc=self.encoder_pos_encoding(x)
            x=x.expand(-1, self.config.gru_input_size, -1 , -1)
            x=x+pos_enc
            x=torch.flatten(x, start_dim=2)
            if additional_inputs.numel()>0:
                additional_inputs=additional_inputs+self.extra_sensor_pos_embed.repeat(bs, 1)
                additional_inputs=additional_inputs.unsqueeze(2)
                x = torch.cat((x, additional_inputs), axis=2)
            x = torch.permute(x, (0, 2, 1))
            output = self.join(self.wp_query.repeat(bs, 1, 1), x)
        else:
            #use fc layer for joining encoding or default to backbone outputs only
            if additional_inputs.numel()>0:
                joined_encoding = self.join(x, additional_inputs)
            else:
                joined_encoding = self.join(x)
            output = self.fc_vector(joined_encoding)
        
        return self.wp_gru.forward(output.squeeze(1), target_point), generated_memory
