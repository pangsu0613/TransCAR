import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import force_fp32

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from pyquaternion import Quaternion


import numpy as np
import pickle
#--------------------- for pointnet2
import importlib
import os
import sys
# -------------------------------
#nusc = NuScenes(version='v1.0-mini', dataroot='/home/pangsu/nuscene_data/NUSCENES_DATASET_ROOT', verbose=True)
nusc = NuScenes(version='v1.0-trainval', dataroot='/home/pangsu/nuscene_data/NUSCENES_DATASET_ROOT', verbose=True)


@HEADS.register_module()
class Detr3DHead(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """
    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.num_cls_fcs = num_cls_fcs - 1
        super(Detr3DHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)


        self.final_cls = nn.Sequential(
            nn.Linear(self.embed_dims,self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims,self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.cls_out_channels),
        )

        self.final_reg = nn.Sequential(
            nn.Linear(self.embed_dims,self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims,self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.code_size),
        )

        self.final_cls2 = nn.Sequential(
            nn.Linear(self.embed_dims,self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims,self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.cls_out_channels),
        )

        self.final_reg2 = nn.Sequential(
            nn.Linear(self.embed_dims,self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims,self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.code_size),
        )

        self.rf_multihead_attn = nn.MultiheadAttention(self.embed_dims, 8, dropout=0.1)
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


        self.rf_multihead_attn2 = nn.MultiheadAttention(self.embed_dims, 8, dropout=0.1)
        # Implementation of Feedforward model
        self.rf_linear1_2 = nn.Linear(self.embed_dims, 512)  # this 512 is the dimension for the feedforward network
        self.rf_dropout_2 = nn.Dropout(0.1)
        self.rf_linear2_2 = nn.Linear(512, self.embed_dims)

        self.rf_norm1_2 = nn.LayerNorm(self.embed_dims)
        self.rf_norm2_2 = nn.LayerNorm(self.embed_dims)
        self.rf_norm3_2 = nn.LayerNorm(self.embed_dims)
        self.rf_dropout1_2 = nn.Dropout(0.1)
        self.rf_dropout2_2 = nn.Dropout(0.1)
        self.rf_dropout3_2 = nn.Dropout(0.1)
        self.rf_activation_2 = nn.ReLU(inplace=True)


        self.radar_position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        '''
        self.radar_feat_encoder = nn.Sequential(
            nn.Linear(36, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.embed_dims),
            nn.ReLU(inplace=True),
        )'''

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ROOT_DIR = BASE_DIR
        sys.path.append(os.path.join(ROOT_DIR))
        #self.pm = importlib.import_module('pointnet2_sem_seg_msg')
        self.pm = importlib.import_module('pointnet2_sem_seg')
        self.radar_pointnet = self.pm.get_model().cuda()

        self.attention_weights2 = nn.Linear(self.embed_dims,6*4)

        self.output_proj2 = nn.Linear(self.embed_dims, self.embed_dims)
        self.nusc = nusc

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            '''
            This part is running in Detr3D because as_two_stage is set to False
            Remember the meaning of Embedding, there are some good examples here: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
            '''
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def forward(self, mlvl_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        query_embeds = self.query_embedding.weight  # shape: [900, 512]
        # The below transformer is Detr3DTransformer (NOT Detr3DTransformerDecoder, the Detr3DTransformerDecoder is just part of Detr3DTransformer)
        # in projects/mmdet3d_plugin/models/utils/detr3d_transformer.py
        # mlvl_feats is the output from a feature pyramid network, it is a list with 4 elements, each one represents a feature map
        # from a certain level. for each feature map the shape is [1, 6, 256, 116, 200], 1 is the batch size, 6 is the number of cameras.
        hs, init_reference, inter_references = self.transformer(
            mlvl_feats,
            query_embeds,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
            img_metas=img_metas,
        )
        #print("head:",mlvl_feats[0].shape,mlvl_feats[1].shape)
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        #print(inter_references.shape) # [6,1,900,3]
        #print("00000000000000000000",torch.isnan(hs[5]).any())
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)


        sample_idx = img_metas[0]['sample_idx']
        sample_instance = self.nusc.get('sample', sample_idx)
        #RadarPointCloud.disable_filters()
        point_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        radar_pointcloud_front, timestamps_list_f = RadarPointCloud.from_file_multisweep(self.nusc, sample_instance, chan="RADAR_FRONT", ref_chan="LIDAR_TOP", nsweeps=5)
        radar_pointcloud_front_left, timestamps_list_fl = RadarPointCloud.from_file_multisweep(self.nusc, sample_instance, chan="RADAR_FRONT_LEFT", ref_chan="LIDAR_TOP", nsweeps=5)
        radar_pointcloud_front_right, timestamps_list_fr = RadarPointCloud.from_file_multisweep(self.nusc, sample_instance, chan="RADAR_FRONT_RIGHT", ref_chan="LIDAR_TOP", nsweeps=5)
        radar_pointcloud_back_left, timestamps_list_bl = RadarPointCloud.from_file_multisweep(self.nusc, sample_instance, chan="RADAR_BACK_LEFT", ref_chan="LIDAR_TOP", nsweeps=5)
        radar_pointcloud_back_right, timestamps_list_br = RadarPointCloud.from_file_multisweep(self.nusc, sample_instance, chan="RADAR_BACK_RIGHT", ref_chan="LIDAR_TOP", nsweeps=5)

        ref_sd_record = self.nusc.get('sample_data', sample_instance['data']['LIDAR_TOP'])
        for radar_name in ['RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']:
            sd_record = self.nusc.get('sample_data', sample_instance['data'][radar_name])
            radar_cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            ref_cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
            if radar_name == 'RADAR_FRONT':
                velocities_f = radar_pointcloud_front.points[8:10, :]  # Compensated velocity
                velocities_f = np.vstack((velocities_f, np.zeros(radar_pointcloud_front.points.shape[1])))
                velocities_f = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities_f)
                velocities_f = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities_f)
                velocities_f[2, :] = np.zeros(radar_pointcloud_front.points.shape[1]) # [3,N]

                velocities_f2 = radar_pointcloud_front.points[6:8, :]  # Compensated velocity
                velocities_f2 = np.vstack((velocities_f2, np.zeros(radar_pointcloud_front.points.shape[1])))
                velocities_f2 = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities_f2)
                velocities_f2 = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities_f2)
                velocities_f2[2, :] = np.zeros(radar_pointcloud_front.points.shape[1]) # [3,N]
            if radar_name == 'RADAR_FRONT_LEFT':
                velocities_fl = radar_pointcloud_front_left.points[8:10, :]  # Compensated velocity
                velocities_fl = np.vstack((velocities_fl, np.zeros(radar_pointcloud_front_left.points.shape[1])))
                velocities_fl = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities_fl)
                velocities_fl = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities_fl)
                velocities_fl[2, :] = np.zeros(radar_pointcloud_front_left.points.shape[1]) # [3,N]

                velocities_fl2 = radar_pointcloud_front_left.points[6:8, :]  # Compensated velocity
                velocities_fl2 = np.vstack((velocities_fl2, np.zeros(radar_pointcloud_front_left.points.shape[1])))
                velocities_fl2 = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities_fl2)
                velocities_fl2 = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities_fl2)
                velocities_fl2[2, :] = np.zeros(radar_pointcloud_front_left.points.shape[1]) # [3,N]
            if radar_name == 'RADAR_FRONT_RIGHT':
                velocities_fr = radar_pointcloud_front_right.points[8:10, :]  # Compensated velocity
                velocities_fr = np.vstack((velocities_fr, np.zeros(radar_pointcloud_front_right.points.shape[1])))
                velocities_fr = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities_fr)
                velocities_fr = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities_fr)
                velocities_fr[2, :] = np.zeros(radar_pointcloud_front_right.points.shape[1]) # [3,N]

                velocities_fr2 = radar_pointcloud_front_right.points[6:8, :]  # Compensated velocity
                velocities_fr2 = np.vstack((velocities_fr2, np.zeros(radar_pointcloud_front_right.points.shape[1])))
                velocities_fr2 = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities_fr2)
                velocities_fr2 = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities_fr2)
                velocities_fr2[2, :] = np.zeros(radar_pointcloud_front_right.points.shape[1]) # [3,N]
            if radar_name == 'RADAR_BACK_LEFT':
                velocities_bl = radar_pointcloud_back_left.points[8:10, :]  # Compensated velocity
                velocities_bl = np.vstack((velocities_bl, np.zeros(radar_pointcloud_back_left.points.shape[1])))
                velocities_bl = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities_bl)
                velocities_bl = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities_bl)
                velocities_bl[2, :] = np.zeros(radar_pointcloud_back_left.points.shape[1]) # [3,N]

                velocities_bl2 = radar_pointcloud_back_left.points[6:8, :]  # Compensated velocity
                velocities_bl2 = np.vstack((velocities_bl2, np.zeros(radar_pointcloud_back_left.points.shape[1])))
                velocities_bl2 = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities_bl2)
                velocities_bl2 = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities_bl2)
                velocities_bl2[2, :] = np.zeros(radar_pointcloud_back_left.points.shape[1]) # [3,N]
            if radar_name == 'RADAR_BACK_RIGHT':
                velocities_br = radar_pointcloud_back_right.points[8:10, :]  # Compensated velocity
                velocities_br = np.vstack((velocities_br, np.zeros(radar_pointcloud_back_right.points.shape[1])))
                velocities_br = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities_br)
                velocities_br = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities_br)
                velocities_br[2, :] = np.zeros(radar_pointcloud_back_right.points.shape[1]) # [3,N]

                velocities_br2 = radar_pointcloud_back_right.points[6:8, :]  # Compensated velocity
                velocities_br2 = np.vstack((velocities_br2, np.zeros(radar_pointcloud_back_right.points.shape[1])))
                velocities_br2 = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities_br2)
                velocities_br2 = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities_br2)
                velocities_br2[2, :] = np.zeros(radar_pointcloud_back_right.points.shape[1]) # [3,N]


        #######################################################  Radar front
        num_f = radar_pointcloud_front.points.shape[1]
        dyn_prop_f = radar_pointcloud_front.points.transpose()[:,3].astype(int)
        ambig_state_f = radar_pointcloud_front.points.transpose()[:,11].astype(int)
        pdh0_f = radar_pointcloud_front.points.transpose()[:,15].astype(int)

        dyn_prop_f_1h = np.zeros([num_f,8])
        dyn_prop_f_1h[list(range(num_f)),dyn_prop_f] = 1.0

        ambig_state_f_1h = np.zeros([num_f,5])
        ambig_state_f_1h[list(range(num_f)),ambig_state_f] = 1.0

        pdh0_f_1h = np.zeros([num_f,8])
        pdh0_f_1h[list(range(num_f)),pdh0_f] = 1.0
        ##################################################### Radar front left
        num_fl = radar_pointcloud_front_left.points.shape[1]
        dyn_prop_fl = radar_pointcloud_front_left.points.transpose()[:,3].astype(int)
        ambig_state_fl = radar_pointcloud_front_left.points.transpose()[:,11].astype(int)
        pdh0_fl = radar_pointcloud_front_left.points.transpose()[:,15].astype(int)

        dyn_prop_fl_1h = np.zeros([num_fl,8])
        dyn_prop_fl_1h[list(range(num_fl)),dyn_prop_fl] = 1.0

        ambig_state_fl_1h = np.zeros([num_fl,5])
        ambig_state_fl_1h[list(range(num_fl)),ambig_state_fl] = 1.0

        pdh0_fl_1h = np.zeros([num_fl,8])
        pdh0_fl_1h[list(range(num_fl)),pdh0_fl] = 1.0
        ###################################################### Radar front right
        num_fr = radar_pointcloud_front_right.points.shape[1]
        dyn_prop_fr = radar_pointcloud_front_right.points.transpose()[:,3].astype(int)
        ambig_state_fr = radar_pointcloud_front_right.points.transpose()[:,11].astype(int)
        pdh0_fr = radar_pointcloud_front_right.points.transpose()[:,15].astype(int)

        dyn_prop_fr_1h = np.zeros([num_fr,8])
        dyn_prop_fr_1h[list(range(num_fr)),dyn_prop_fr] = 1.0

        ambig_state_fr_1h = np.zeros([num_fr,5])
        ambig_state_fr_1h[list(range(num_fr)),ambig_state_fr] = 1.0

        pdh0_fr_1h = np.zeros([num_fr,8])
        pdh0_fr_1h[list(range(num_fr)),pdh0_fr] = 1.0
        ##################################################### Radar back left
        num_bl = radar_pointcloud_back_left.points.shape[1]
        dyn_prop_bl = radar_pointcloud_back_left.points.transpose()[:,3].astype(int)
        ambig_state_bl = radar_pointcloud_back_left.points.transpose()[:,11].astype(int)
        pdh0_bl = radar_pointcloud_back_left.points.transpose()[:,15].astype(int)

        dyn_prop_bl_1h = np.zeros([num_bl,8])
        dyn_prop_bl_1h[list(range(num_bl)),dyn_prop_bl] = 1.0

        ambig_state_bl_1h = np.zeros([num_bl,5])
        ambig_state_bl_1h[list(range(num_bl)),ambig_state_bl] = 1.0

        pdh0_bl_1h = np.zeros([num_bl,8])
        pdh0_bl_1h[list(range(num_bl)),pdh0_bl] = 1.0
        ###################################################### Radar back right
        num_br = radar_pointcloud_back_right.points.shape[1]
        dyn_prop_br = radar_pointcloud_back_right.points.transpose()[:,3].astype(int)
        ambig_state_br = radar_pointcloud_back_right.points.transpose()[:,11].astype(int)
        pdh0_br = radar_pointcloud_back_right.points.transpose()[:,15].astype(int)

        dyn_prop_br_1h = np.zeros([num_br,8])
        dyn_prop_br_1h[list(range(num_br)),dyn_prop_br] = 1.0

        ambig_state_br_1h = np.zeros([num_br,5])
        ambig_state_br_1h[list(range(num_br)),ambig_state_br] = 1.0

        pdh0_br_1h = np.zeros([num_br,8])
        pdh0_br_1h[list(range(num_br)),pdh0_br] = 1.0
        ########################################################

        #------------------------------------------------------------------------------------------
        # adjusting time, becuase the default time is using sample frame time as reference
        # the timestamp is in seconds
        #print(timestamps_list_f.shape)
        if timestamps_list_f.shape[1] != 0:
            start_time_f =  np.max(timestamps_list_f)
            timestamps_list_f = timestamps_list_f - start_time_f
            timestamps_list_f = np.repeat(timestamps_list_f.transpose(),2,axis=1)
            offset_f = velocities_f.transpose()[:,:2] * timestamps_list_f
        else:
            timestamps_list_f = np.repeat(timestamps_list_f.transpose(),2,axis=1)
            offset_f = velocities_f.transpose()[:,:2] * timestamps_list_f

        if timestamps_list_fl.shape[1] != 0:
            start_time_fl =  np.max(timestamps_list_fl)
            timestamps_list_fl = timestamps_list_fl - start_time_fl
            timestamps_list_fl = np.repeat(timestamps_list_fl.transpose(),2,axis=1)
            offset_fl = velocities_fl.transpose()[:,:2] * timestamps_list_fl
        else:
            timestamps_list_fl = np.repeat(timestamps_list_fl.transpose(),2,axis=1)
            offset_fl = velocities_fl.transpose()[:,:2] * timestamps_list_fl

        if timestamps_list_fr.shape[1] != 0:
            start_time_fr =  np.max(timestamps_list_fr)
            timestamps_list_fr = timestamps_list_fr - start_time_fr
            timestamps_list_fr = np.repeat(timestamps_list_fr.transpose(),2,axis=1)
            offset_fr = velocities_fr.transpose()[:,:2] * timestamps_list_fr
        else:
            timestamps_list_fr = np.repeat(timestamps_list_fr.transpose(),2,axis=1)
            offset_fr = velocities_fr.transpose()[:,:2] * timestamps_list_fr

        if timestamps_list_bl.shape[1] != 0:
            start_time_bl =  np.max(timestamps_list_bl)
            timestamps_list_bl = timestamps_list_bl - start_time_bl
            timestamps_list_bl = np.repeat(timestamps_list_bl.transpose(),2,axis=1)
            offset_bl = velocities_bl.transpose()[:,:2] * timestamps_list_bl
        else:
            timestamps_list_bl = np.repeat(timestamps_list_bl.transpose(),2,axis=1)
            offset_bl = velocities_bl.transpose()[:,:2] * timestamps_list_bl

        if timestamps_list_br.shape[1] != 0:
            start_time_br =  np.max(timestamps_list_br)
            timestamps_list_br = timestamps_list_br - start_time_br
            timestamps_list_br = np.repeat(timestamps_list_br.transpose(),2,axis=1)
            offset_br = velocities_br.transpose()[:,:2] * timestamps_list_br
        else:
            timestamps_list_br = np.repeat(timestamps_list_br.transpose(),2,axis=1)
            offset_br = velocities_br.transpose()[:,:2] * timestamps_list_br
        '''
        start_time_f =  np.max(timestamps_list_f)
        start_time_fl =  np.max(timestamps_list_fl)
        start_time_fr =  np.max(timestamps_list_fr)
        start_time_bl =  np.max(timestamps_list_bl)
        start_time_br =  np.max(timestamps_list_br)
        timestamps_list_f = timestamps_list_f - start_time_f
        timestamps_list_fl = timestamps_list_fl - start_time_fl
        timestamps_list_fr = timestamps_list_fr - start_time_fr
        timestamps_list_bl = timestamps_list_bl - start_time_bl
        timestamps_list_br = timestamps_list_br - start_time_br
        timestamps_list_f = np.repeat(timestamps_list_f.transpose(),2,axis=1)
        timestamps_list_fl = np.repeat(timestamps_list_fl.transpose(),2,axis=1)
        timestamps_list_fr = np.repeat(timestamps_list_fr.transpose(),2,axis=1)
        timestamps_list_bl = np.repeat(timestamps_list_bl.transpose(),2,axis=1)
        timestamps_list_br = np.repeat(timestamps_list_br.transpose(),2,axis=1)
        #print(timestamps_list_f.shape)
        offset_f = velocities_f.transpose()[:,:2] * timestamps_list_f
        offset_fl = velocities_fl.transpose()[:,:2] * timestamps_list_fl
        offset_fr = velocities_fr.transpose()[:,:2] * timestamps_list_fr
        offset_bl = velocities_bl.transpose()[:,:2] * timestamps_list_bl
        offset_br = velocities_br.transpose()[:,:2] * timestamps_list_br'''

        #print(offset_f.shape)


        # (0)x (1)y (2)z (3)dyn_prop (4)id (5)rcs (6)vx (7)vy (8)vx_comp (9)vy_comp (10)is_quality_valid (11)ambig_state (12)x_rms (13)y_rms (14)invalid_state (15)pdh0 (16)vx_rms (17)vy_rms
        radar_front_xyz = radar_pointcloud_front.points.transpose()[:,[0,1,2,4,5,10,14]]  # [0,1,2,8,9] x y z vx_comp vy_comp
        radar_front_left_xyz = radar_pointcloud_front_left.points.transpose()[:,[0,1,2,4,5,10,14]]  # [0,1,2,8,9]
        radar_front_right_xyz = radar_pointcloud_front_right.points.transpose()[:,[0,1,2,4,5,10,14]]  # [0,1,2,8,9]
        radar_back_left_xyz = radar_pointcloud_back_left.points.transpose()[:,[0,1,2,4,5,10,14]]  # [0,1,2,8,9]
        radar_back_right_xyz = radar_pointcloud_back_right.points.transpose()[:,[0,1,2,4,5,10,14]]  # [0,1,2,8,9]
        #radar_feat_num = radar_front_xyz.shape[1]
        #print(timestamps_list_f.shape, radar_front_xyz.shape, offset_f.shape)
        #print(radar_front_right_xyz.shape, timestamps_list_fr.shape)
        radar_front_xyz = np.concatenate((radar_front_xyz, timestamps_list_f, offset_f, velocities_f.transpose()[:,:2], velocities_f2.transpose()[:,:2],dyn_prop_f_1h, ambig_state_f_1h, pdh0_f_1h),axis=1)
        radar_front_left_xyz = np.concatenate((radar_front_left_xyz, timestamps_list_fl, offset_fl, velocities_fl.transpose()[:,:2], velocities_fl2.transpose()[:,:2],dyn_prop_fl_1h, ambig_state_fl_1h, pdh0_fl_1h),axis=1)
        radar_front_right_xyz = np.concatenate((radar_front_right_xyz, timestamps_list_fr, offset_fr, velocities_fr.transpose()[:,:2], velocities_fr2.transpose()[:,:2],dyn_prop_fr_1h, ambig_state_fr_1h, pdh0_fr_1h),axis=1)
        radar_back_left_xyz = np.concatenate((radar_back_left_xyz, timestamps_list_bl, offset_bl, velocities_bl.transpose()[:,:2], velocities_bl2.transpose()[:,:2],dyn_prop_bl_1h, ambig_state_bl_1h, pdh0_bl_1h),axis=1)
        radar_back_right_xyz = np.concatenate((radar_back_right_xyz, timestamps_list_br, offset_br, velocities_br.transpose()[:,:2], velocities_br2.transpose()[:,:2],dyn_prop_br_1h, ambig_state_br_1h, pdh0_br_1h),axis=1)
        radar_feat_num = radar_front_xyz.shape[1]
        #print(radar_front_xyz.shape)


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


        radar_tokens = torch.zeros(radar_feat_num,1,1500).cuda()  #
        radar_tokens[:,0,:] = 500
        fill_in = min(1500,radar_points_num)
        radar_all_xyz_tensor = radar_all_xyz_tensor.permute(2,0,1)
        radar_tokens[:,:,:fill_in] = radar_all_xyz_tensor[:,:,:fill_in]
        radar_points = radar_tokens[:3,:,:]   #radar_token shape: [7,1,1500]
        #print("$$$$$",radar_points.shape)
        radar_points = radar_points.permute(1,2,0)
        radar_pos_feat = self.radar_position_encoder(radar_points)
        #print("before:",radar_tokens.shape)
        #--------------------------------- Radar PointNet2 Su Pang
        radar_tokens = radar_tokens.permute(1,2,0)
        #print("radar_tokens",radar_tokens.shape)
        #radar_feat = self.radar_feat_encoder(radar_tokens)
        radar_feat = self.radar_pointnet(radar_tokens.permute(0,2,1))
        #print(radar_feat.shape)
        radar_feat = radar_feat.permute(0,2,1)
        combined_radar_feat = radar_pos_feat + radar_feat

        #print(outputs_coords[0].shape)
        # the shape of the output feature map: [6, 1, 900, 256]
        #print(hs[5].shape)
        query_feat = hs[5].permute(1,0,2).clone()
        k_padding_mask = torch.zeros(1,1500,dtype=torch.bool).cuda()
        k_padding_mask[0,fill_in:] = True ### True means ignored

        reference = inter_references[-1]
        reference_points_clone = reference.clone()
        reference_points_clone[..., 0:1] = reference_points_clone[..., 0:1]*(self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        reference_points_clone[..., 1:2] = reference_points_clone[..., 1:2]*(self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        reference_points_clone[..., 2:3] = reference_points_clone[..., 2:3]*(self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        #radar_tokens = radar_tokens.permute(1,2,0)
        dist = torch.cdist(reference_points_clone[:,:,:2],radar_tokens[:,:,:2],p=2.0) # [1,N1.N2] [1,900,1500]
        #dist_min_value,dist_min_ind = torch.min(dist,1)
        attention_mask = dist[0]>2.0
        #print(attention_mask.shape)
        nan_row_index = torch.where((attention_mask == False).any(dim=1))[0]
        new_attention_mask = attention_mask[nan_row_index]
        #print(new_attention_mask.shape)
        #print((new_attention_mask==True).all(dim=1)==True)
        query_feat_input = query_feat[nan_row_index]

        combined_radar_feat = combined_radar_feat.permute(1,0,2)
        tgt2,att_score = self.rf_multihead_attn(query_feat_input, combined_radar_feat, combined_radar_feat,
                                                #key_padding_mask=k_padding_mask, attn_mask = new_attention_mask)#[0]
                                                attn_mask = new_attention_mask)

        query_feat[nan_row_index] = query_feat[nan_row_index] + self.rf_dropout2(tgt2)
        #query_feat_input = query_feat_input + self.rf_dropout2(tgt2)
        query_feat = self.rf_norm2(query_feat)
        ffn_out = self.rf_linear2(self.rf_dropout(self.rf_activation(self.rf_linear1(query_feat))))
        query_feat = query_feat + self.rf_dropout3(ffn_out)
        query_feat = self.rf_norm3(query_feat)

        #query_feat[nan_row_index] = query_feat_input

        query_feat = query_feat.permute(1,0,2)

        output_final_class = self.final_cls(query_feat)
        tmp = self.final_reg(query_feat)
        #print("333333333333333333333",torch.isnan(tgt2).any(),torch.isnan(att_score).any(),torch.isnan(query_feat).any(),torch.isnan(reference).any(),torch.isnan(radar_pos_feat).any())
        #reference = inverse_sigmoid(reference)
        #print("444444444444444444444",torch.isnan(reference).any())
        assert reference.shape[-1] == 3
        reference[..., 0:1] = (reference[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        reference[..., 1:2] = (reference[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        reference[..., 4:5] = (reference[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
        tmp[..., 0:2] += reference[..., 0:2]
        #tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
        tmp[..., 4:5] += reference[..., 2:3]
        #tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
        #tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        #tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        #tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])



        # TODO: check if using sigmoid
        output_final_coord = tmp
        outputs_classes = []
        outputs_coords = []
        outputs_classes.append(output_final_class)
        outputs_coords.append(output_final_coord)

        #===========================================================================================================================
        # new reference points
        new_reference_3d = torch.zeros_like(reference)
        new_reference_3d[..., :2] = tmp[..., :2]
        new_reference_3d[..., 2:3] = tmp[..., 4:5]

        dist2 = torch.cdist(new_reference_3d[:,:,:2],radar_tokens[:,:,:2],p=2.0) # [1,N1.N2] [1,900,1500]
        #dist_min_value,dist_min_ind = torch.min(dist,1)
        #print(torch.min(new_reference_3d),torch.max(new_reference_3d),torch.min(radar_tokens),torch.max(radar_tokens))
        attention_mask2 = dist2[0]>1.0

        query_feat = query_feat.clone().permute(1,0,2)
        nan_row_index2 = torch.where((attention_mask2 == False).any(dim=1))[0]
        new_attention_mask2 = attention_mask2[nan_row_index2]

        # re-sample image features
        #print("before:",new_reference_3d[0,:5,:])
        reference_points_3d_sampling, img_feat, mask = feature_sampling(
            mlvl_feats, new_reference_3d, self.pc_range, img_metas)
        #print("after:",new_reference_3d[0,:5,:])
        img_feat[torch.isnan(img_feat)]=0
        mask[torch.isnan(mask)]=0
        attention_weights = self.attention_weights2(query_feat).view(
            1, 1, 900, 6, 1, 4)
        attention_weights = attention_weights.sigmoid() * mask  # I guess this is the attention score ???? I think this is used to erase the ones that are out of the field of view
        img_feat = img_feat * attention_weights  #
        img_feat = img_feat.sum(-1).sum(-1).sum(-1)
        img_feat = img_feat.permute(2, 0, 1) # shape: [900, 1, 256]
        img_feat = self.output_proj2(img_feat)
        query_feat = query_feat + img_feat

        query_feat_input2 = query_feat[nan_row_index2]

        #combined_radar_feat = combined_radar_feat.permute(1,0,2)
        tgt2_2,att_score2 = self.rf_multihead_attn2(query_feat_input2, combined_radar_feat, combined_radar_feat,
                                                #key_padding_mask=k_padding_mask, attn_mask = new_attention_mask)#[0]
                                                attn_mask = new_attention_mask2)


        query_feat[nan_row_index2] = query_feat[nan_row_index2] + self.rf_dropout2_2(tgt2_2)
        #query_feat_input = query_feat_input + self.rf_dropout2(tgt2)
        query_feat = self.rf_norm2_2(query_feat)
        ffn_out2 = self.rf_linear2_2(self.rf_dropout_2(self.rf_activation_2(self.rf_linear1_2(query_feat))))
        query_feat = query_feat + self.rf_dropout3_2(ffn_out2)
        query_feat = self.rf_norm3_2(query_feat)

        query_feat = query_feat.permute(1,0,2)






        output_final_class2 = self.final_cls2(query_feat)
        tmp2 = self.final_reg2(query_feat)
        assert new_reference_3d.shape[-1] == 3
        tmp2[..., 0:2] += new_reference_3d[..., 0:2]
        #tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
        tmp2[..., 4:5] += new_reference_3d[..., 2:3]
        output_final_coord2 = tmp2

        outputs_classes.append(output_final_class2)
        outputs_coords.append(output_final_coord2)


        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        #print(outputs_classes.shape)
        ###### save to pickle file for analysis
        '''
        save_dict = {}
        ##### Note the shape of radar_points, then shape may have problem !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        radar_points_for_plot = radar_points#radar_tokens[:3,:,:].clone()
        save_dict['radar_points'] = radar_points_for_plot
        save_dict['att_score'] = att_score
        reference_points_clone = reference.clone().sigmoid()
        reference_points_clone[..., 0:1] = reference_points_clone[..., 0:1]*(self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        reference_points_clone[..., 1:2] = reference_points_clone[..., 1:2]*(self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        reference_points_clone[..., 2:3] = reference_points_clone[..., 2:3]*(self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        save_dict['referece_points'] = reference_points_clone
        filename = '/home/pangsu/Downloads/detr3d/att_plot/raw_output_exp11/' + sample_idx + '.pkl'
        pkl_output = open(filename,'wb')
        pickle.dump(save_dict,pkl_output)
        pkl_output.close()'''

        #print("55555555555555555555555555",torch.isnan(hs[5]).any())
        #print("55555555555555555555555",torch.isnan(outputs_classes).any(),torch.isnan(outputs_coords).any())
        outs = {
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :9]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        #loss_cls[torch.isnan(loss_cls)]=0
        #loss_bbox[torch.isnan(loss_bbox)]=0
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        #print(num_dec_layers)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)
        #print(len(losses_cls))
        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, 9)
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list

def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    #reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    #reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    #reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    # reference_points (B, num_queries, 4)
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
    #mask[torch.isnan(mask)]=0
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
