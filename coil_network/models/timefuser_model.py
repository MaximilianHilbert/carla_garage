import torch.nn as nn
import torch
from team_code.aim import AIMBackbone
import team_code.transfuser_utils as t_u
from team_code.model import PositionEmbeddingSine, GRUWaypointsPredictorTransFuser
import random
class TimeFuser(nn.Module):
    def __init__(self, name,config):
        super().__init__()
        self.config = config
        self.name=name
        measurement_contribution=1 if self.config.speed else 0
        previous_wp_contribution=1 if self.config.prevnum>0 else 0
        memory_contribution=1 if self.name=="arp-policy" else 0
        self.extra_sensor_memory_contribution=sum([measurement_contribution, previous_wp_contribution,memory_contribution])
        self.set_img_token_len()
        self.set_required_input_channels()
        
        
        self.image_encoder=AIMBackbone(config, channels= self.input_channels)

        original_channel_dimension = self.image_encoder.image_encoder.feature_info[-1]["num_chs"]
        self.channel_dimension=original_channel_dimension
        
        
        self.time_position_embedding = nn.Parameter(torch.zeros(self.img_token_len, self.channel_dimension,self.config.img_encoding_remaining_spatial_dim[0],self.config.img_encoding_remaining_spatial_dim[1]))
        self.spatial_position_embedding_per_image=PositionEmbeddingSine(num_pos_feats=self.channel_dimension//2, normalize=True)

        if self.config.speed:
            if self.name=="arp-policy":
                # 1 time the velocity (current timestep only)
                self.speed_layer = nn.Linear(in_features=self.total_steps_considered-self.config.max_img_seq_len_baselines, out_features=self.channel_dimension)
            elif self.name=="arp-memory":
                # 6 times the velocity (of previous timesteps only)
                self.speed_layer = nn.Linear(in_features=self.config.max_img_seq_len_baselines, out_features=self.channel_dimension)
            else:
                self.speed_layer = nn.Linear(in_features=self.total_steps_considered, out_features=self.channel_dimension)
        if self.config.prevnum>0 and self.name!="arp-policy":
            #we input the previous waypoints in our ablations only in the memory stream of arp
            self.previous_wp_layer = nn.Linear(in_features=self.config.target_point_size*self.config.prevnum, out_features=self.channel_dimension)
        decoder_norm=nn.LayerNorm(self.channel_dimension)
        transformer_encoder_layer=nn.TransformerEncoderLayer(d_model=self.channel_dimension, nhead=self.config.transformer_heads,batch_first=True)
        self.transformer_encoder=nn.TransformerEncoder(encoder_layer=transformer_encoder_layer,num_layers=self.config.numtransformerlayers, norm=decoder_norm)

        transformer_decoder_layer=nn.TransformerDecoderLayer(d_model=self.channel_dimension, nhead=self.config.transformer_heads,batch_first=True)
        self.transformer_decoder=nn.TransformerDecoder(decoder_layer=transformer_decoder_layer, num_layers=self.config.numtransformerlayers, norm=decoder_norm)
        self.wp_query = nn.Parameter(
                        torch.zeros(
                            self.config.pred_len,
                            self.channel_dimension,
                        )
                    )
        # positional embeddings with respect to themselves (between individual tokens of the same type)
        self.output_token_pos_embedding = nn.Parameter(torch.zeros(self.config.num_bev_query**2+self.config.pred_len, self.channel_dimension))
     

        self.wp_gru = GRUWaypointsPredictorTransFuser(config, pred_len=self.config.pred_len, hidden_size=self.config.gru_hidden_size,target_point_size=self.config.target_point_size)
        if self.config.bev:
            self.bev_query=nn.Parameter(
                            torch.zeros(
                                self.config.num_bev_query**2,
                                self.flattened_channel_dimension,
                            )
                        )
        # positional embeddings with respect to themselves (between individual tokens of the same type)
        self.output_token_pos_embedding_wp = nn.Parameter(torch.zeros(self.config.pred_len, self.flattened_channel_dimension))
        self.output_token_pos_embedding_bev = nn.Parameter(torch.zeros(self.config.num_bev_query**2, self.flattened_channel_dimension))
     

        self.wp_gru = GRUWaypointsPredictorTransFuser(config, pred_len=self.config.pred_len, hidden_size=self.config.gru_hidden_size,target_point_size=self.config.target_point_size)
        if self.config.bev:
            # Computes which pixels are visible in the camera. We mask the others.
            _, valid_voxels = t_u.create_projection_grid(self.config)
            valid_bev_pixels = torch.max(valid_voxels, dim=3, keepdim=False)[0].unsqueeze(1)
            # Conversion from CARLA coordinates x depth, y width to image coordinates x width, y depth.
            # Analogous to transpose after the LiDAR histogram
            valid_bev_pixels = torch.transpose(valid_bev_pixels, 2, 3).contiguous()
            # Register as parameter so that it will automatically be moved to the correct GPU with the rest of the network
            self.valid_bev_pixels = nn.Parameter(valid_bev_pixels, requires_grad=False)
           
            self.bev_semantic_decoder = nn.Sequential(
                nn.Conv2d(
                    self.flattened_channel_dimension,
                    self.flattened_channel_dimension,
                    kernel_size=(3, 3),
                    stride=1,
                    padding=(1, 1),
                    bias=True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    self.flattened_channel_dimension,
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

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x, speed=None, target_point=None, prev_wp=None, memory_to_fuse=None):
        pred_dict={}
        bs=x.shape[0]
        if self.config.backbone=="stacking":
            x=torch.cat([x[:, i,...] for i in range(self.img_token_len)], axis=1)
            x= self.image_encoder(x)
            x=self.change_channel(x)
            x=x.unsqueeze(1)
        else:
            encodings=[]
            for image_index in range(self.img_token_len):
                #this is done to save gpu memory for gradients
                if image_index!=self.img_token_len-1:
                    with torch.no_grad():
                        single_frame_encoding=self.image_encoder(x[:,image_index,...])
                        single_frame_encoding=self.change_channel(single_frame_encoding)
                else:
                    single_frame_encoding=self.image_encoder(x[:,image_index,...])
                    single_frame_encoding=self.change_channel(single_frame_encoding)
                encodings.append(single_frame_encoding)
            x=torch.stack(encodings, dim=1)
       
        if self.config.speed:
            measurement_enc = self.speed_layer(speed).unsqueeze(1) #we add the token dimension here
        else:
            measurement_enc = None
        if self.config.prevnum>0 and self.name!="arp-policy":
            prev_wp_enc = self.previous_wp_layer(prev_wp).unsqueeze(1)
        else:
            prev_wp_enc=None
        x_clone=x.clone()
        for image_index in range(self.img_token_len):
            x_clone[:,image_index,...]=x[:,image_index,...]+self.spatial_position_embedding_per_image(x[:,image_index,...])
            x_clone[:,image_index,...]=x[:,image_index,...]+self.time_position_embedding[image_index,...].repeat(bs, 1, 1, 1)
        x=x_clone
        if self.name=="arp-memory":
            #we (positionally) embed the memory and flatten it to use it directly in the forwardpass of arp-policy
            generated_memory=x.flatten(start_dim=2)
            pred_dict.update({"memory": generated_memory})
        additional_inputs=[input for input in [memory_to_fuse, measurement_enc, prev_wp_enc] if input is not None]
        if additional_inputs:
            additional_inputs=torch.cat(additional_inputs, dim=1)
            x=torch.cat((x.permute(0, 1, 3,4,2).flatten(start_dim=1, end_dim=3),additional_inputs), axis=1)
        else:
            x=x.permute(0, 1, 3,4,2).flatten(start_dim=1, end_dim=3)
        x=self.transformer_encoder(x)
        if self.config.bev:
            queries=torch.cat((self.wp_query, self.bev_query), axis=0).repeat(bs,1,1)+self.output_token_pos_embedding.repeat(bs, 1,1)
        else:
            queries=self.wp_query.repeat(bs,1,1)
        all_tokens_output=self.transformer_decoder(queries, x)
        wp_tokens=self.wp_gru(all_tokens_output[:, :self.wp_query.shape[0],...], target_point)
        pred_dict.update({"wp_predictions": wp_tokens})
        if self.config.bev:
            bev_tokens=all_tokens_output[:, self.wp_query.shape[0]:,...]
            bev_tokens=bev_tokens.permute(0,2,1).reshape(bs, self.channel_dimension, self.config.num_bev_query, self.config.num_bev_query).contiguous()
            pred_bev_grid=self.bev_semantic_decoder(bev_tokens)
            pred_bev_semantic = pred_bev_grid * self.valid_bev_pixels
            pred_dict.update({"pred_bev_semantic": pred_bev_semantic})
            pred_dict.update({"valid_bev_pixels":self.valid_bev_pixels})
        return pred_dict

    def create_resnet_backbone(self, config, number_first_layer_channels):
        self.image_encoder=AIMBackbone(config)
        old_conv_1=self.image_encoder.image_encoder.stem.conv
        new_conv_1=nn.Conv2d(in_channels=number_first_layer_channels, out_channels=old_conv_1.out_channels, kernel_size=old_conv_1.kernel_size, stride=old_conv_1.stride, padding=old_conv_1.padding, bias=old_conv_1.bias)
        if number_first_layer_channels == old_conv_1.in_channels:
            new_conv_1.weight.data = old_conv_1.weight.data
        else:
            new_conv_1.weight.data = old_conv_1.weight.data.mean(dim=1, keepdim=True).repeat(1, number_first_layer_channels, 1, 1)

        # Replace the first convolutional layer
        self.image_encoder.image_encoder.stem.conv = new_conv_1

    def set_img_token_len(self):
        self.total_steps_considered=self.config.max_img_seq_len_baselines+1
        if self.config.backbone=="stacking":
            self.img_token_len=1
        else:
            if self.name=="arp-memory":
                self.img_token_len=self.config.max_img_seq_len_baselines
            elif self.name=="bcoh" or self.name=="keyframes":
                self.img_token_len=self.config.max_img_seq_len_baselines+1
            else:
                self.img_token_len=1

    def set_required_input_channels(self):
        if self.config.backbone=="stacking":
            self.input_channels=self.img_token_len*self.config.rgb_input_channels
        else:
            self.input_channels=self.config.rgb_input_channels