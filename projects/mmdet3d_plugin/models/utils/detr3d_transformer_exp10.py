
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
import pickle
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud

nusc = NuScenes(version='v1.0-mini', dataroot='/home/pangsu/nuscene_data/NUSCENES_DATASET_ROOT', verbose=True)
#nusc = NuScenes(version='v1.0-trainval', dataroot='/home/pangsu/nuscene_data/NUSCENES_DATASET_ROOT', verbose=True)

a = torch.randn(1,900,3).cuda()
def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER.register_module()
class Detr3DTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 decoder=None,
                 **kwargs):
        super(Detr3DTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.reference_points = nn.Linear(self.embed_dims, 3)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention) or isinstance(m, Detr3DCrossAtten):
                m.init_weight()
        xavier_init(self.reference_points, distribution='uniform', bias=0.)

    def forward(self,
                mlvl_feats,
                query_embed,
                reg_branches=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        assert query_embed is not None
        bs = mlvl_feats[0].size(0)
        ### mlvl_feats : [1,6,256,116,200]
        ### query_embed: [900, 512]
        query_pos, query = torch.split(query_embed, self.embed_dims , dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos) # [1, 900, 3]
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points  # [1,900,3]
        # decoder
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)





        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=mlvl_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            **kwargs)

        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class Detr3DTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, radar_features=None, **kwargs):
        super(Detr3DTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.radar_feature_encoder = nn.Sequential(
            nn.Linear(7, self.embed_dims),
            #nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            #nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.nusc = nusc
        #self.pc_range = pc_range

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                radar_features=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query # [900,1,256]
        intermediate = []
        intermediate_reference_points = []

        ########################################################
        sample_idx = kwargs['img_metas'][0]['sample_idx']
        sample_instance = self.nusc.get('sample', sample_idx)
        #RadarPointCloud.disable_filters()
        point_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        radar_pointcloud_front, timestamps_list = RadarPointCloud.from_file_multisweep(self.nusc, sample_instance, chan="RADAR_FRONT", ref_chan="LIDAR_TOP", nsweeps=5)
        radar_pointcloud_front_left, timestamps_list = RadarPointCloud.from_file_multisweep(self.nusc, sample_instance, chan="RADAR_FRONT_LEFT", ref_chan="LIDAR_TOP", nsweeps=5)
        radar_pointcloud_front_right, timestamps_list = RadarPointCloud.from_file_multisweep(self.nusc, sample_instance, chan="RADAR_FRONT_RIGHT", ref_chan="LIDAR_TOP", nsweeps=5)
        radar_pointcloud_back_left, timestamps_list = RadarPointCloud.from_file_multisweep(self.nusc, sample_instance, chan="RADAR_BACK_LEFT", ref_chan="LIDAR_TOP", nsweeps=5)
        radar_pointcloud_back_right, timestamps_list = RadarPointCloud.from_file_multisweep(self.nusc, sample_instance, chan="RADAR_BACK_RIGHT", ref_chan="LIDAR_TOP", nsweeps=5)
        ####
        #print(radar_pointcloud_front.points.shape)
        radar_front_xyz = radar_pointcloud_front.points.transpose()[:,[0,1,2,6,7,8,9]]  # [0,1,2,8,9] x y z vx_comp vy_comp
        radar_front_left_xyz = radar_pointcloud_front_left.points.transpose()[:,[0,1,2,6,7,8,9]]  # [0,1,2,8,9]
        radar_front_right_xyz = radar_pointcloud_front_right.points.transpose()[:,[0,1,2,6,7,8,9]]  # [0,1,2,8,9]
        radar_back_left_xyz = radar_pointcloud_back_left.points.transpose()[:,[0,1,2,6,7,8,9]]  # [0,1,2,8,9]
        radar_back_right_xyz = radar_pointcloud_back_right.points.transpose()[:,[0,1,2,6,7,8,9]]  # [0,1,2,8,9]
        radar_feat_num = radar_front_xyz.shape[1]

        radar_all_xyz = np.concatenate((radar_front_xyz,radar_front_left_xyz,radar_front_right_xyz,radar_back_left_xyz,radar_back_right_xyz),axis=0)

        points_mask = ((radar_all_xyz[:, 0] > point_range[0])
                          & (radar_all_xyz[:, 1] > point_range[1])
                          & (radar_all_xyz[:, 2] > point_range[2])
                          & (radar_all_xyz[:, 0] < point_range[3])
                          & (radar_all_xyz[:, 1] < point_range[4])
                          & (radar_all_xyz[:, 2] < point_range[5]))

        filtered_radar_points = radar_all_xyz[points_mask]
        radar_points_num = filtered_radar_points.shape[0]  #[n, 7(or 5 or 3)]
        radar_all_xyz_tensor = torch.from_numpy(filtered_radar_points).reshape(1,-1,radar_feat_num).type(torch.float).cuda()


        radar_tokens = torch.zeros(radar_feat_num,1,1500).cuda()  # 625 is a good number for 5 sweeps
        fill_in = min(1500,radar_points_num)
        radar_tokens[:,:,:fill_in] = radar_all_xyz_tensor.permute(2,0,1)[:,:,:fill_in]
        #radar_points = radar_tokens[:3,:,:].clone()
        # normalize the radar points into 0~1
        #radar_points[0:1,...] = (radar_points[0:1,...] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        #radar_points[1:2,...] = (radar_points[1:2,...] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        #radar_points[2:3,...] = (radar_points[2:3,...] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        #print(radar_tokens.shape,radar_points_num)
        radar_features = self.radar_feature_encoder(radar_tokens.permute(1,2,0))  #[1,256,625]
        #####################################################


        for lid, layer in enumerate(self.layers):
            #if lid > 5: break     # This is just for ploting the queries
            # This layer is MultiheadAttention + Detr3DCrossAtten + FFN, FFN is not defined here
            #print(type(layer.ffns[0]),type(layer.attentions[0]))
            reference_points_input = reference_points  # shape: [1, 900, 3] for all 6 layers

            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                radar_features=radar_features,
                filtered_radar_points=radar_tokens,
                **kwargs)
            output = output.permute(1, 0, 2) # shape: [1,900,256], NO, not self.dropout(output) + inp_residual + pos_feat from Detr3DCrossAtten

            if reg_branches is not None:
                tmp = reg_branches[lid](output) # [1,900,10]
                assert reference_points.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[
                    ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@ATTENTION.register_module()
class Detr3DCrossAtten(BaseModule):
    """An attention module used in Detr3d.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=5,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False):
        super(Detr3DCrossAtten, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.nusc = nusc

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.attention_weights = nn.Linear(embed_dims,
                                           num_cams*num_levels*num_points)

        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )


        '''
        self.radar_position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )'''

        #self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dims, nhead = self.num_heads)
        #self.radar_camera_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1)

        ##########################################
        self.rf_multihead_attn = nn.MultiheadAttention(self.embed_dims, self.num_heads, dropout=0.1)
        # Implementation of Feedforward model
        self.rf_linear1 = nn.Linear(self.embed_dims, 512)  # this 512 is the dimension for the feedforward network
        self.rf_dropout = nn.Dropout(0.1)
        self.rf_linear2 = nn.Linear(512, self.embed_dims)

        self.rf_norm1 = nn.LayerNorm(self.embed_dims)
        self.rf_norm2 = nn.LayerNorm(self.embed_dims)
        self.rf_norm3 = nn.LayerNorm(self.embed_dims)
        self.rf_dropout1 = nn.Dropout(0.1)
        self.rf_dropout2 = nn.Dropout(0.1)
        self.rf_dropout3 = nn.Dropout(0.1)
        self.rf_activation = nn.ReLU(inplace=True)


        self.batch_first = batch_first

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                radar_features=None,
                filtered_radar_points=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            #print("no key")# YES
            key = query
        if value is None:
            #print("no value") # NO
            value = key
        '''
        value is the features from the image with the shape of [1, 6, 256, 116, 200] ==> [batch_size, num_cameras, num_channels, width, height]
        '''
        if residual is None:
            #print("no residual")# YES
            inp_residual = query
        if query_pos is not None:
            #print("yes query_pos")# YES
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()
        # (attention_weights): Linear(in_features=256, out_features=24, bias=True)
        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)
        # attention_weights shape: [1, 1, 900, 6, 1, 4]    I guess the num_levels is the number of levels in the FPN (within the image backbone)



        #################################
        '''
        sample_idx = kwargs['img_metas'][0]['sample_idx']
        sample_instance = self.nusc.get('sample', sample_idx)
        #RadarPointCloud.disable_filters()
        #radar_pointcloud1, timestamps_list = RadarPointCloud.from_file_multisweep(self.nusc, sample_instance, chan="RADAR_FRONT", ref_chan="LIDAR_TOP", nsweeps=1)
        radar_pointcloud_front, timestamps_list = RadarPointCloud.from_file_multisweep(self.nusc, sample_instance, chan="RADAR_FRONT", ref_chan="LIDAR_TOP", nsweeps=5)
        radar_pointcloud_front_left, timestamps_list = RadarPointCloud.from_file_multisweep(self.nusc, sample_instance, chan="RADAR_FRONT_LEFT", ref_chan="LIDAR_TOP", nsweeps=5)
        radar_pointcloud_front_right, timestamps_list = RadarPointCloud.from_file_multisweep(self.nusc, sample_instance, chan="RADAR_FRONT_RIGHT", ref_chan="LIDAR_TOP", nsweeps=5)
        radar_pointcloud_back_left, timestamps_list = RadarPointCloud.from_file_multisweep(self.nusc, sample_instance, chan="RADAR_BACK_LEFT", ref_chan="LIDAR_TOP", nsweeps=5)
        radar_pointcloud_back_right, timestamps_list = RadarPointCloud.from_file_multisweep(self.nusc, sample_instance, chan="RADAR_BACK_RIGHT", ref_chan="LIDAR_TOP", nsweeps=5)
        ####
        #print(radar_pointcloud_front.points.shape)
        radar_front_xyz = radar_pointcloud_front.points.transpose()[:,[0,1,2,8,9]]  # [0,1,2,8,9] x y z vx_comp vy_comp
        radar_front_left_xyz = radar_pointcloud_front_left.points.transpose()[:,[0,1,2,8,9]]  # [0,1,2,8,9]
        radar_front_right_xyz = radar_pointcloud_front_right.points.transpose()[:,[0,1,2,8,9]]  # [0,1,2,8,9]
        radar_back_left_xyz = radar_pointcloud_back_left.points.transpose()[:,[0,1,2,8,9]]  # [0,1,2,8,9]
        radar_back_right_xyz = radar_pointcloud_back_right.points.transpose()[:,[0,1,2,8,9]]  # [0,1,2,8,9]
        radar_feat_num = radar_front_xyz.shape[1]
        radar_points_num = radar_front_xyz.shape[0]
        radar_all_xyz = np.concatenate((radar_front_xyz,radar_front_left_xyz,radar_front_right_xyz,radar_back_left_xyz,radar_back_right_xyz),axis=0)
        radar_all_xyz_tensor = torch.from_numpy(radar_all_xyz).reshape(1,-1,radar_feat_num).type(torch.float).cuda()

        #reference_points_clone = reference_points.clone()

        #self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dims, nhead = self.num_heads)
        #self.radar_camera_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1)

        radar_tokens = torch.zeros(5,1,1500).cuda()  # 625 is a good number for 5 sweeps
        fill_in = min(1500,radar_points_num)
        radar_tokens[:,:,:fill_in] = radar_all_xyz_tensor.permute(2,0,1)[:,:,:fill_in]'''
        radar_tokens = filtered_radar_points
        radar_points = radar_tokens[:3,:,:].clone()
        # normalize the radar points into 0~1
        radar_points[0:1,...] = (radar_points[0:1,...] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        radar_points[1:2,...] = (radar_points[1:2,...] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        radar_points[2:3,...] = (radar_points[2:3,...] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        #print(radar_tokens.shape,radar_points_num)
        #radar_features = self.radar_feature_encoder(radar_tokens.permute(1,2,0))  #[1,256,625]
        radar_pos_feat = self.position_encoder(inverse_sigmoid(radar_points).permute(1,2,0))  #[1,256,625]  I think for position encoder, we don't need normalization, just raw data is enough.
        radar_features = radar_features + radar_pos_feat


        '''
        reference_points_clone[..., 0:1] = reference_points_clone[..., 0:1]*(self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        reference_points_clone[..., 1:2] = reference_points_clone[..., 1:2]*(self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        reference_points_clone[..., 2:3] = reference_points_clone[..., 2:3]*(self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        #print(type(radar_all_xyz_tensor),type(reference_points_3d))
        #print(reference_points_clone[0,:5,:])
        #print(radar_all_xyz_tensor.shape)
        dist = torch.cdist(radar_all_xyz_tensor[:,:,:2],reference_points_clone[:,:,:2],p=2.0) # [1,N1.N2]
        dist_min_value,dist_min_ind = torch.min(dist,1)
        #dist_min_value_ind = dist_min_value < 1.0
        #print(dist_min_value_ind.shape)
        #print("before:",reference_points_clone[dist_min_value<1.2][:,:2])
        reference_points_clone[0,(dist_min_value<0.5).reshape(-1),:2] = radar_all_xyz_tensor[0,dist_min_ind[dist_min_value<0.5],:2]
        #print("after",reference_points_clone[dist_min_value<1.2][:,:2])
        #reference_points_clone[dist_min_value<0.5] = radar_all_xyz_tensor[0,dist_min_ind[dist_min_value<0.5]]
        reference_points_clone[..., 0:1] = (reference_points_clone[..., 0:1] - self.pc_range[0])/(self.pc_range[3] - self.pc_range[0])
        reference_points_clone[..., 1:2] = (reference_points_clone[..., 1:2] - self.pc_range[1])/(self.pc_range[4] - self.pc_range[1])
        reference_points_clone[..., 2:3] = (reference_points_clone[..., 2:3] - self.pc_range[2])/(self.pc_range[5] - self.pc_range[2])'''
        ##############################################
        reference_points_3d, output, mask = feature_sampling(
            value, reference_points, self.pc_range, kwargs['img_metas'])
        # the output above is in shape: [1, 256, 900, 6, 1, 4]
        # the mask above is in shapee: [1, 1, 900, 6, 1, 1]
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)


        # attention_weights shape: [1, 1, 900, 6, 1, 4]
        attention_weights = attention_weights.sigmoid() * mask  # I guess this is the attention score ???? I think this is used to erase the ones that are out of the field of view

        output = output * attention_weights  #
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1) # shape: [900, 1, 256]


        # (output_proj): Linear(in_features=256, out_features=256, bias=True)
        output = self.output_proj(output)
        #print(output.shape,radar_features.shape)

        #output = self.radar_camera_decoder(output,radar_features.permute(1,0,2))


        ################################
        tgt2,att_score = self.rf_multihead_attn(output, radar_features.permute(1,0,2), radar_features.permute(1,0,2))#[0]
        output = output + self.rf_dropout2(tgt2)
        output = self.rf_norm2(output)
        tgt2 = self.rf_linear2(self.rf_dropout(self.rf_activation(self.rf_linear1(output))))
        output = output + self.rf_dropout3(tgt2)
        output = self.rf_norm3(output)

        #### save to pickle file for analysis
        '''
        save_dict = {}
        radar_points_for_plot = radar_tokens[:3,:,:].clone()
        save_dict['radar_points'] = radar_points_for_plot
        save_dict['att_score'] = att_score
        reference_points_clone = reference_points.clone()
        reference_points_clone[..., 0:1] = reference_points_clone[..., 0:1]*(self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        reference_points_clone[..., 1:2] = reference_points_clone[..., 1:2]*(self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        reference_points_clone[..., 2:3] = reference_points_clone[..., 2:3]*(self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        save_dict['referece_points'] = reference_points_clone
        filename = '/home/pangsu/Downloads/detr3d/att_plot/raw_output/' + kwargs['img_metas'][0]['sample_idx'] + '.pkl'
        pkl_output = open(filename,'wb')
        pickle.dump(save_dict,pkl_output)
        pkl_output.close()'''

        # (num_query, bs, embed_dims)
        #pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)
        #print(reference_points_clone.shape)
        #pos_feat_radar = self.radar_position_encoder(inverse_sigmoid(reference_points_clone)).permute(1, 0, 2)
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)  # reference_points_3d [1,900,3]
        #pos_feat = pos_feat + pos_feat_radar
        ################################################
        '''
        radar_feat = self.radar_position_encoder(radar_all_xyz_tensor).permute(1, 0, 2)
        combined_feat = self.radar_camera_attention(inp_residual,pos_feat,pos_feat)  #multihead_attn(query, key, value)
        print(combined_feat.shape)'''

        ################################################
        # all the tensros below are in shape [900,1,256]
        # output is the image feature, inp_residual is the original query, pos_feat if the new positional encoding
        #global a
        #a = reference_points_clone
        return self.dropout(output) + inp_residual + pos_feat


def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
    '''
    img_metas is a list with only one element (maybe because the batch size is one), img_metas[0] is a dictionary, the keys are as follows:
    'filename', 'ori_shape', 'img_shape', 'lidar2img', 'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip',
    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'sample_idx', 'pcd_scale_factor', 'pts_filename', 'input_shape']
    '''
    lidar2img = []
    #print("#########",img_metas[0].keys())
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    # reference_points (B, num_queries, 4)
    #print(reference_points[0,:10,:])
    #print("done!!!", img_metas[0].keys())
    #print("done!")
    '''
    Below is used to save the object queries as pkl files.

    filename = '/home/pangsu/Downloads/detr3d/query_plot/layer_5/' + img_metas[0]['sample_idx'] + '.pkl'
    output = open(filename,'wb')
    pickle.dump(reference_points,output)
    output.close()'''

    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1)
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0)
                 & (reference_points_cam[..., 0:1] < 1.0)
                 & (reference_points_cam[..., 1:2] > -1.0)
                 & (reference_points_cam[..., 1:2] < 1.0))
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    mask = torch.nan_to_num(mask)
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B*N, C, H, W)
        reference_points_cam_lvl = reference_points_cam.view(B*N, num_query, 1, 2)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam,  1, len(mlvl_feats))
    return reference_points_3d, sampled_feats, mask
