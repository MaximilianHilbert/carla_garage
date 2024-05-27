import torch.nn as nn
import torch
import importlib

from .building_blocks.gru import GRUWaypointsPredictorTransFuser
from .building_blocks import FC
from .building_blocks import Join
import team_code.transfuser_utils as t_u
from team_code.model import PositionEmbeddingSine,PositionalEncoding_one_dim

class CoILICRA(nn.Module):
    def __init__(self, name, config):
        super(CoILICRA, self).__init__()
        self.params = config.model_configuration
        self.config = config
        self.name=name
        measurement_contribution=self.config.additional_inputs_memory_output_size if self.config.speed else 0
        previous_wp_contribution=self.config.additional_inputs_memory_output_size if self.config.prevnum>0 else 0
        memory_contribution=self.config.additional_inputs_memory_output_size if self.name=="coil-policy" else 0
        self.extra_sensor_memory_contribution=sum([measurement_contribution, previous_wp_contribution,memory_contribution])

        if self.config.backbone=="rnn" or self.name=="coil-policy":
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
        if self.config.backbone=="rnn":
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
        if self.config.speed:
            self.measurements = FC(
                params={
                    "neurons": [len(config.inputs)] + self.config.measurement_layers+[self.config.additional_inputs_memory_output_size],
                    "dropouts": self.config.measurement_dropouts,
                    "end_layer": False,
                }
            )
        if self.config.prevnum>0:
            self.previous_wp = FC(
                params={
                    #-1 because we define n-1 as previous wp in the dataloader
                    "neurons": [self.config.target_point_size*self.config.prevnum]
                    + self.config.previous_waypoints_layers+[self.config.additional_inputs_memory_output_size],
                    "dropouts": self.config.previous_waypoints_dropouts,
                    "end_layer": False,
                }
            )
        #we use that to get down from 512 channels (resnet output) to 64 channels, used for measurement encoding
        if self.name=="coil-policy":
            self.change_channel_memory = nn.Conv1d(
                    self.config.resnet_output_feat_dim,
                    self.config.additional_inputs_memory_output_size,
                    kernel_size=1,
                )
        if self.config.bev:
            # Computes which pixels are visible in the camera. We mask the others.
            _, valid_voxels = t_u.create_projection_grid(self.config)
            valid_bev_pixels = torch.max(valid_voxels, dim=3, keepdim=False)[0].unsqueeze(1)
            # Conversion from CARLA coordinates x depth, y width to image coordinates x width, y depth.
            # Analogous to transpose after the LiDAR histogram
            valid_bev_pixels = torch.transpose(valid_bev_pixels, 2, 3).contiguous()
            #valid_bev_pixels_inv = 1.0 - valid_bev_pixels
            # Register as parameter so that it will automatically be moved to the correct GPU with the rest of the network
            self.valid_bev_pixels = nn.Parameter(valid_bev_pixels, requires_grad=False)
            #self.valid_bev_pixels_inv = nn.Parameter(valid_bev_pixels_inv, requires_grad=False)
            decoder_norm=nn.LayerNorm(self.config.bev_features_channels)
            self.bev_query = nn.Parameter(
                            torch.zeros(
                                1,
                                self.config.bev_features_channels,
                                
                            )
                        )
            bev_token_decoder= nn.TransformerDecoderLayer(
                        self.config.bev_positional_encoding_dim,
                        self.config.num_decoder_heads_bev,
                        activation=nn.GELU(),
                        batch_first=True,
                    )
            self.bev_token_decoder = torch.nn.TransformerDecoder(
                        bev_token_decoder,
                        num_layers=self.config.bev_decoder_layer,
                        norm=decoder_norm,
                    )
            self.bev_semantic_decoder = nn.Sequential(
                nn.Conv2d(
                    self.config.bev_features_channels,
                    self.config.bev_features_channels,
                    kernel_size=(3, 3),
                    stride=1,
                    padding=(1, 1),
                    bias=True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    self.config.bev_features_channels,
                    self.config.num_bev_semantic_classes,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=0,
                    bias=True,
                ),
                nn.Upsample(
                    size=(
                        self.config.bev_height,
                        self.config.bev_width,
                    ),
                    mode="bilinear",
                    align_corners=False,
                ),
            )

        if self.config.td or self.config.bev:
            self.encoder_pos_encoding_one_dim=PositionalEncoding_one_dim(self.config.positional_embedding_dim)
        
        if self.config.bev:
            self.encoder_pos_encoding_two_dim=PositionEmbeddingSine(self.config.positional_embedding_dim//2, normalize=True)
        if self.extra_sensor_memory_contribution>0:
            self.extra_sensor_pos_embed = nn.Parameter(torch.zeros(1, self.extra_sensor_memory_contribution))
        if self.config.td:
            #we use that to determine the necessary token length, depending on additional inputs into the transformer
            
            self.wp_query = nn.Parameter(
                            torch.zeros(
                                1,
                                self.config.gru_input_size,
                            )
                        )
            decoder_norm=nn.LayerNorm(self.config.gru_hidden_size)
            decoder_layer = nn.TransformerDecoderLayer(
                        self.config.positional_embedding_dim,
                        self.config.num_decoder_heads,
                        activation=nn.GELU(),
                        batch_first=True,
                    )
            self.join = torch.nn.TransformerDecoder(
                        decoder_layer,
                        num_layers=self.config.num_td_layers,
                        norm=decoder_norm,
                    )
 
        else:
            self.join =  FC(
                        params={
                            "neurons": [
                               self.extra_sensor_memory_contribution+self.config.backbone_dim
                            ]
                            + self.config.join_layers,
                            "dropouts": self.config.join_layer_dropouts,
                            "end_layer": False,
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
        if self.config.backbone=="stacking":
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
        if self.config.speed:
            measurement_enc = self.measurements(speed)
        else:
            measurement_enc = None
        if self.config.prevnum>0:
            prev_wp_enc = self.previous_wp(prev_wp)
        else:
            prev_wp_enc=None
        if self.name=="coil-policy":
            memory_to_fuse=self.change_channel_memory(memory_to_fuse.unsqueeze(2)).squeeze(2)
        additional_inputs=[input for input in [memory_to_fuse, measurement_enc, prev_wp_enc] if input is not None]
        if additional_inputs:
            additional_inputs=torch.cat(additional_inputs, dim=1)
        else:
            additional_inputs=torch.empty(0)

        #eventually decode to wp tokens via transformer decoder, later use gru to unroll to actual waypoints
        if self.config.bev:
            encoding_positional=self.encoder_pos_encoding_two_dim(x.reshape(-1, 1, 16,32))
            encoding_positional=x.reshape(-1, 1, 16,32)+encoding_positional.repeat(bs, 1,1,1)
        if self.config.td:
            encoding_positional=self.encoder_pos_encoding_one_dim(x)
            if additional_inputs.numel()>0:
                additional_inputs_with_pos_embedding=additional_inputs+self.extra_sensor_pos_embed.repeat(bs, 1)
                additional_inputs_with_pos_embedding=additional_inputs_with_pos_embedding.unsqueeze(2)
                additional_inputs_with_pos_embedding=additional_inputs_with_pos_embedding.expand(-1, -1, self.config.positional_embedding_dim)
                encoding_positional = torch.cat((encoding_positional, additional_inputs_with_pos_embedding), axis=1) #64 per additional input, so in case of memory input, speed, prev_wp we have 512 (resnet), 64 (memory), 64 (speed), 64 (prev_wp)
                output = self.join(self.wp_query.repeat(bs, 1, 1), encoding_positional)
        if not self.config.td:
            #use fc layer for joining encoding or default to backbone outputs only
            if additional_inputs.numel()>0:
                joined_encoding=torch.cat((x, additional_inputs), axis=1)
                joined_encoding = self.join(joined_encoding)
            else:
                joined_encoding = self.join(x)
            output = self.fc_vector(joined_encoding)
        if self.config.bev:
            bev_tokens=self.bev_token_decoder(self.bev_query.repeat(bs, 1, 1), encoding_positional).squeeze(1)
            pred_bev_grid=self.bev_semantic_decoder(bev_tokens.unsqueeze(2).unsqueeze(2))
            pred_bev_semantic = pred_bev_grid * self.valid_bev_pixels
        else:
            pred_bev_semantic=None
        return pred_bev_semantic, self.wp_gru.forward(output.squeeze(1), target_point), generated_memory