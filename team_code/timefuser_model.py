import torch.nn as nn
import torch
from team_code.aim import AIMBackbone
import team_code.transfuser_utils as t_u
from torch.nn.functional import l1_loss
from torch.nn import CrossEntropyLoss
from team_code.model import PositionEmbeddingSine, GRUWaypointsPredictorTransFuser, get_sinusoidal_positional_embedding_image_order
from team_code.center_net import LidarCenterNetHead
from team_code.video_swin_transformer import SwinTransformer3D
from team_code.x3d import X3D
from team_code.video_resnet import VideoResNet
import os
from coil_utils.baseline_helpers import download_file
import random
def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.constant_(layer.weight, 0.01)  # Initialize weights with 0.01
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)  # Initialize biases with 0
class TimeFuser(nn.Module):
    def __init__(self, name,config, rank=0, training=True):
        super().__init__()
        self.config = config
        self.name=name
        measurement_contribution=1 if self.config.speed else 0
        previous_wp_contribution=1 if self.config.prevnum>0 else 0
        memory_contribution=1 if self.name=="arp-policy" else 0
        self.extra_sensor_memory_contribution=sum([measurement_contribution, previous_wp_contribution,memory_contribution])
        self.set_img_token_len_and_channels_and_seq_len()
        
        if self.config.backbone=="swin":
            if training and config.pretrained==1:
                pretrained_path=os.path.join(os.environ.get("WORK_DIR"), "swin_pretrain", "swin_tiny_patch244_window877_kinetics400_1k.pth")
                download_file("https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth", pretrained_path)
            else:
                pretrained_path=None
            self.image_encoder=SwinTransformer3D(depths=(2, 2, 6,2),
        num_heads=(3, 6, 12, 24), pretrained=pretrained_path)
            
            self.channel_dimension=self.image_encoder.num_features
        if self.config.backbone=="videoresnet":
            self.image_encoder=VideoResNet(in_channels=3, pretrained="R2Plus1D_18_Weights.KINETICS400_V1" if self.config.pretrained else None)
            self.channel_dimension=self.image_encoder.feature_info.info[-1]["num_chs"]
        if self.config.backbone=="resnet":
            self.image_encoder=AIMBackbone(config, channels=self.input_channels, pretrained=True if self.config.pretrained else False)

            original_channel_dimension = self.image_encoder.image_encoder.feature_info[-1]["num_chs"]
            self.channel_dimension=self.config.reduced_channel_dimension
            self.change_channel = nn.Conv2d(
                        original_channel_dimension,
                        self.channel_dimension,
                        kernel_size=1,
                    )
            #self.time_position_embedding = nn.Parameter(torch.zeros(self.img_token_len, self.channel_dimension,self.config.img_encoding_remaining_spatial_dim[0],self.config.img_encoding_remaining_spatial_dim[1]))
            self.time_position_embedding=get_sinusoidal_positional_embedding_image_order
            self.spatial_position_embedding_per_image=PositionEmbeddingSine(num_pos_feats=self.channel_dimension//2, normalize=True)
        if self.config.backbone.startswith("x3d"):
            self.image_encoder=X3D(model_name=self.config.backbone)
            self.channel_dimension=self.image_encoder.output_channels
        if self.config.speed:
            if self.name=="arp-policy":
                # 1 time the velocity (current timestep only)
                self.speed_layer = nn.Linear(in_features=1, out_features=self.channel_dimension)
            elif self.name=="arp-memory":
                # 6 times the velocity (of previous timesteps only)
                self.speed_layer = nn.Linear(in_features=self.total_steps_considered-1, out_features=self.channel_dimension)
            else:
                self.speed_layer = nn.Linear(in_features=self.total_steps_considered, out_features=self.channel_dimension)
        if self.config.prevnum>0 and self.name!="arp-policy":
            #we input the previous waypoints in our ablations only in the memory stream of arp
            self.previous_wp_layer = nn.Linear(in_features=self.config.target_point_size*self.config.prevnum, out_features=self.channel_dimension)
        decoder_norm=nn.LayerNorm(self.channel_dimension)
        if self.config.backbone=="resnet":
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
        if self.config.bev or self.config.detectboxes:
            self.output_token_pos_embedding = nn.Parameter(torch.zeros(self.config.num_bev_query**2+self.config.pred_len, self.channel_dimension))
     

        self.wp_gru = GRUWaypointsPredictorTransFuser(config, pred_len=self.config.pred_len, hidden_size=self.config.gru_hidden_size,target_point_size=self.config.target_point_size)
        if self.config.bev or self.config.detectboxes:
            self.bev_query=nn.Parameter(
                            torch.zeros(
                                self.config.num_bev_query**2,
                                self.channel_dimension,
                            )
                        )
            if self.config.bev:
                # Computes which pixels are visible in the camera. We mask the others.
                _, valid_voxels = t_u.create_projection_grid(self.config)
                valid_bev_pixels = torch.max(valid_voxels, dim=3, keepdim=False)[0].unsqueeze(1)
                # Conversion from CARLA coordinates x depth, y width to image coordinates x width, y depth.
                # Analogous to transpose after the LiDAR histogram
                valid_bev_pixels = torch.transpose(valid_bev_pixels, 2, 3)
                # Register as parameter so that it will automatically be moved to the correct GPU with the rest of the network
                self.valid_bev_pixels = nn.Parameter(valid_bev_pixels, requires_grad=False)
            
                self.bev_semantic_decoder = nn.Sequential(
                    nn.Conv2d(
                        self.channel_dimension,
                        self.channel_dimension,
                        kernel_size=(3, 3),
                        stride=1,
                        padding=(1, 1),
                        bias=True,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        self.channel_dimension,
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
            if self.config.detectboxes:
                self.head=LidarCenterNetHead(config=config)
                self.change_channel_bev_to_bb_and_upscale= nn.Sequential( nn.Conv2d(
                    self.channel_dimension,
                    self.channel_dimension,
                    kernel_size=(3, 3),
                    stride=1,
                    padding=(1, 1),
                    bias=True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    self.channel_dimension,
                    self.config.bb_feature_channel,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=0,
                    bias=True,
                ),
                nn.Upsample(
                    size=(
                        self.config.bb_input_channel,
                        self.config.bb_input_channel,
                    ),
                    mode="bilinear",
                    align_corners=False,
                ),
                )
        if self.config.ego_velocity_prediction:
            self.downsample_to_ego_velocity= nn.Sequential( nn.Conv2d(
                    self.config.bb_feature_channel,
                    self.config.bb_feature_channel,
                    kernel_size=(3, 3),
                    stride=1,
                    padding=(1, 1),
                    bias=True,
                ),
                nn.AdaptiveAvgPool2d((1, 1)))
            
            self.ego_velocity_predictor=nn.Sequential(nn.Linear(in_features=self.config.bb_feature_channel, out_features=self.config.hidden_ego_velocity_head),
                nn.Linear(in_features=self.config.hidden_ego_velocity_head, out_features=1))
        #self.apply(init_weights)
    
    def forward(self, x, speed=None, target_point=None, prev_wp=None, memory_to_fuse=None):
        pred_dict={}
        #custom code for summary as a workaround, because torchinfo doesnt allow None type input args:
        if torch.all(x==1):
            speed=None
            prev_wp=None
            memory_to_fuse=None
        bs, time, channel,height, width=x.shape
        if self.config.framehandling=="stacking":
            x=torch.cat([x[:,i,:,:] for i in range(len(x[0,:,0,0]))], axis=1)
            x= self.image_encoder(x)
            x=self.change_channel(x)
            x=x.unsqueeze(1)
        else:
            if self.config.backbone=="resnet":
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
            if self.config.backbone=="swin":
                x=x.permute(0,2,1,3,4)
                x=self.image_encoder.patch_embed(x)
                for layer in self.image_encoder.layers.values():
                    x=layer(x)
                x=x.flatten(start_dim=2, end_dim=4).permute(0,2,1).contiguous()
            if self.config.backbone=="videoresnet" or self.config.backbone.startswith("x3d"):
                x=x.permute(0,2,1,3,4)
                for _,layer in self.image_encoder.items():
                    x=layer(x)
                x=x.flatten(start_dim=2, end_dim=4).permute(0,2,1).contiguous()
        if self.config.speed:
            measurement_enc = self.speed_layer(speed).unsqueeze(0).unsqueeze(0) #we add the token dimension here
        else:
            measurement_enc = None
        if self.config.prevnum>0 and self.name!="arp-policy":
            prev_wp_enc = self.previous_wp_layer(prev_wp).unsqueeze(1)
        else:
            prev_wp_enc=None
        if self.config.backbone=="resnet":
            for image_index in range(self.img_token_len):
                x[:,image_index,...]=x[:,image_index,...]+self.spatial_position_embedding_per_image(x[:,image_index,...])
            
            time_positional_embeddung=self.time_position_embedding(x).unsqueeze(0).unsqueeze(3).unsqueeze(4)
            time_positional_embeddung=time_positional_embeddung.expand(*x.shape)
            x=x+time_positional_embeddung
        if self.name=="arp-memory":
            #we (positionally) embed the memory and flatten it to use it directly in the forwardpass of arp-policy
            if self.config.backbone!="swin":
                generated_memory=x.permute(0, 1, 3,4,2).flatten(start_dim=1, end_dim=3).contiguous()
            else:
                generated_memory=x
            pred_dict.update({"memory": generated_memory})
        additional_inputs=[input for input in [memory_to_fuse, measurement_enc, prev_wp_enc] if input is not None]
        if len(additional_inputs)!=0 and self.config.backbone=="resnet":
            additional_inputs=torch.cat(additional_inputs, dim=1)
            x=torch.cat((x.permute(0, 1, 3,4,2).flatten(start_dim=1, end_dim=3),additional_inputs), axis=1).contiguous()
        if len(additional_inputs)==0 and self.config.backbone=="resnet":
            x=x.permute(0, 1, 3,4,2).flatten(start_dim=1, end_dim=3).contiguous()
        if len(additional_inputs)!=0 and self.config.backbone=="swin":
            additional_inputs=torch.cat(additional_inputs, dim=1)
            x=torch.cat((x,additional_inputs), axis=1).contiguous()
        if self.config.backbone=="resnet":
            x=self.transformer_encoder(x)
        if self.config.bev or self.config.detectboxes:
            queries=torch.cat((self.wp_query, self.bev_query), axis=0).repeat(bs,1,1)+self.output_token_pos_embedding.repeat(bs, 1,1)
        else:
            queries=self.wp_query.repeat(bs,1,1)
        all_tokens_output=self.transformer_decoder(queries, x)
        wp_tokens=self.wp_gru(all_tokens_output[:, :self.wp_query.shape[0],...], target_point)
        pred_dict.update({"wp_predictions": wp_tokens})
        if self.config.bev or self.config.detectboxes:
            bev_tokens=all_tokens_output[:, self.wp_query.shape[0]:,...]
            bev_tokens=bev_tokens.permute(0,2,1).reshape(bs, self.channel_dimension, self.config.num_bev_query, self.config.num_bev_query).contiguous()
        if self.config.bev:
            pred_bev_grid=self.bev_semantic_decoder(bev_tokens)
            pred_bev_semantic = pred_bev_grid * self.valid_bev_pixels
            pred_dict.update({"pred_bev_semantic": pred_bev_semantic})
            pred_dict.update({"valid_bev_pixels":self.valid_bev_pixels})
        if self.config.detectboxes:
            bev_tokens=self.change_channel_bev_to_bb_and_upscale(bev_tokens)
            pred_bb=self.head(bev_tokens)
            pred_dict.update({"pred_bb": pred_bb})
        if self.config.ego_velocity_prediction:
            bev_tokens=self.downsample_to_ego_velocity(bev_tokens)
            ego_velocity_prediction=self.ego_velocity_predictor(bev_tokens.flatten())
            pred_dict.update({"pred_ego_velocity": ego_velocity_prediction})
        return pred_dict

    def set_img_token_len_and_channels_and_seq_len(self):
        self.total_steps_considered=self.config.img_seq_len
        if self.config.backbone=="stacking":
            self.img_token_len=1
            if self.name=="arp-policy" or self.name=="bcso":
                self.img_seq_len=1
                self.input_channels=1*self.config.rgb_input_channels
            elif self.name=="arp-memory":
                self.img_seq_len=self.config.img_seq_len-1
                self.input_channels=(self.config.img_seq_len-1)*self.config.rgb_input_channels
            else:
                self.img_seq_len=self.config.img_seq_len
                self.input_channels=(self.config.img_seq_len)*self.config.rgb_input_channels
        else:
            self.input_channels=self.config.rgb_input_channels
            if self.name=="arp-policy" or self.name=="bcso":
                self.img_token_len=1
                self.img_seq_len=1
            elif self.name=="arp-memory":
                self.img_token_len=self.config.img_seq_len-1
                self.img_seq_len=self.config.img_seq_len-1
            else:
                self.img_seq_len=self.config.img_seq_len
                self.img_token_len=self.config.img_seq_len
                
    def compute_loss(self,params, keyframes=False):
        losses={}
        main_loss = l1_loss(params["wp_predictions"], params["targets"])
        losses["wp_loss"]=main_loss

        if (self.config.detectboxes and not self.config.freeze) or (self.config.freeze and (params["epoch"]>self.config.epochs_baselines-self.config.epochs_after_freeze)):
            self.detailed_loss_weights={}
            factor = 1.0 / sum(self.config.detailed_loss_weights.values())
            for k in self.config.detailed_loss_weights:
                self.detailed_loss_weights[k] = self.config.detailed_loss_weights[k] * factor
        
        
        if (self.config.bev and not self.config.freeze) or (self.config.freeze and (params["epoch"]>self.config.epochs_baselines-self.config.epochs_after_freeze)):
            if self.config.use_label_smoothing:
                label_smoothing =self.config.label_smoothing_alpha
            else:
                label_smoothing = 0.0
            loss_bev_semantic = CrossEntropyLoss(
                    weight=torch.tensor(self.config.bev_semantic_weights, dtype=torch.float32, device=params["device_id"]),
                    label_smoothing=label_smoothing,
                    ignore_index=-1,
                )
            visible_bev_semantic_label = params["valid_bev_pixels"].squeeze(1).int() * params["bev_targets"]
                # Set 0 class to ignore index -1
            visible_bev_semantic_label = (params["valid_bev_pixels"].squeeze(1).int() - 1) + visible_bev_semantic_label
            loss_bev=loss_bev_semantic(params["pred_bev_semantic"], visible_bev_semantic_label)
            losses["bev_loss"]=loss_bev
        if self.config.detectboxes:
            head_loss=self.head.loss(*params["pred_bb"], *params["targets_bb"])
            sub_loss=torch.zeros((1,),dtype=torch.float32, device=params["device_id"])
            for key, value in head_loss.items():
                sub_loss += self.detailed_loss_weights[key] * value
            losses["detect_loss"]=sub_loss.squeeze()
        else:
            head_loss=None
        if self.config.ego_velocity_prediction:
            ego_velocity_loss=l1_loss(params["pred_ego_velocity"], params["ego_velocity"])
            losses["ego_velocity_loss"]=ego_velocity_loss
        #watch out with order of losses
        final_loss= torch.sum(torch.stack(list(losses.values()))*torch.tensor(self.config.lossweights, device=params["device_id"], dtype=torch.float32)/(len(self.config.lossweights)))

        if keyframes:
            importance_sampling_method = params["importance_sampling_method"]
            if importance_sampling_method == "mean":
                final_loss = torch.mean(final_loss)
            else:
                weight_importance_sampling = params["action_predict_loss"]
                if importance_sampling_method == "softmax":
                    weighted_loss_function = final_loss * nn.functional.softmax(
                        weight_importance_sampling / params["importance_sampling_softmax_temper"],
                        dim=0,
                    )
                elif importance_sampling_method == "threshold":
                    scaled_weight_importance = (weight_importance_sampling > params["importance_sampling_threshold"]).type(
                        torch.float
                    ) * (params["importance_sampling_threshold_weight"] - 1) + 1
                    weighted_loss_function = final_loss * scaled_weight_importance
                else:
                    raise ValueError
                final_loss = torch.mean(weighted_loss_function)
        return final_loss, losses, head_loss
    def convert_features_to_bb_metric(self, bb_predictions):
        bboxes = self.head.get_bboxes(
            bb_predictions[0],
            bb_predictions[1],
            bb_predictions[2],
            bb_predictions[3],
            bb_predictions[4],
            bb_predictions[5],
            bb_predictions[6],
        )[0]

        # filter bbox based on the confidence of the prediction
        bboxes = bboxes[bboxes[:, -1] > self.config.bb_confidence_threshold]

        carla_bboxes = []
        for bbox in bboxes.detach().cpu().numpy():
            bbox = t_u.bb_image_to_vehicle_system(
                bbox, self.config.pixels_per_meter, self.config.min_x, self.config.min_y
            )
            carla_bboxes.append(bbox)

        return carla_bboxes