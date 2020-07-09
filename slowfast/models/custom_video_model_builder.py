
"""Compressed Video models."""

import torch
import torch.nn as nn

import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.batchnorm_helper import get_norm

from . import head_helper, resnet_helper, stem_helper
from .build import MODEL_REGISTRY

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {18: (2, 2, 2, 2), 50: (3, 4, 6, 3), 101: (3, 4, 23, 3))}

@MODEL_REGISTRY.register()
class ComNet(nn.Module):
    """
    ComNet model builder for Compressed Video Recognition network.

    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ComNet, self).__init__()
        self.norm_module = get_norm(cfg)
        self._construct_network(cfg)
        
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a ComNet model. 
        
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """

        (d2_i, d3_i, d4_i, d5_i) = _MODEL_STAGE_DEPTH[cfg.RESNET_I.DEPTH] #resnet depth for i-frame blocks
        (d2_p, d3_p, d4_p, d5_p) = _MODEL_STAGE_DEPTH[cfg.RESNET_P.DEPTH] #resnet depth for p-frame blocks

        # Build stem for i-frames
        self.s1_i = stem_helper.VideoModelStem(
            dim_in= [3] ,
            dim_out=[64],
            kernel=[[1, 7, 7]],
            stride=[[1, 2, 2]],
            padding=[[1 // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        # Build stem for motion vectors
        self.s1_m = stem_helper.VideoModelStem(
            dim_in= [2] ,
            dim_out=[64],
            kernel=[[1, 7, 7]],
            stride=[[1, 2, 2]],
            padding=[[1 // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        # Build stem for residuals
        self.s1_r = stem_helper.VideoModelStem(
            dim_in= [3] ,
            dim_out=[64],
            kernel=[[1, 7, 7]],
            stride=[[1, 2, 2]],
            padding=[[1 // 2, 3, 3]],
            norm_module=self.norm_module,
        )



        self.s2_i = resnet_helper.ResStage(
            dim_in=[64],
            dim_out=[64*4],
            dim_inner=[64],
            temp_kernel_sizes=[[1]],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2_i],
            num_groups=[cfg.RESNET.NUM_GROUPS],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )


        self.s2_m = resnet_helper.ResStage(
            dim_in=[64],
            dim_out=[64],
            dim_inner=[64],
            temp_kernel_sizes=[[1]],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2_p],
            num_groups=[cfg.RESNET.NUM_GROUPS],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        self.s2_r = resnet_helper.ResStage(
            dim_in=[64],
            dim_out=[64],
            dim_inner=[64],
            temp_kernel_sizes=[[1]],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2_p],
            num_groups=[cfg.RESNET.NUM_GROUPS],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )
        
        self.s2_fuse = FuseIframeToPframe(
            dim_in = 256,
            fusion_conv_channel_ratio = 2,
            norm_module=self.norm_module,
        )

        self.s3_i = resnet_helper.ResStage(
            dim_in=[256],
            dim_out=[512],
            dim_inner=[128],
            temp_kernel_sizes=[[3]],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3_i],
            num_groups=[cfg.RESNET.NUM_GROUPS],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s3_m = resnet_helper.ResStage(
            dim_in=[128],
            dim_out=[128],
            dim_inner=[128],
            temp_kernel_sizes=[[3]],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3_p],
            num_groups=[cfg.RESNET.NUM_GROUPS],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_r = resnet_helper.ResStage(
            dim_in=[128],
            dim_out=[128],
            dim_inner=[128],
            temp_kernel_sizes=[[3]],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3_p],
            num_groups=[cfg.RESNET.NUM_GROUPS],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s3_fuse = FuseIframeToPframe(
            dim_in = 512,
            fusion_conv_channel_ratio = 2,
            norm_module=self.norm_module,
        )

        width_per_group = 64
        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=[[3]],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4_i],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=[[3]],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5_i],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        self.head = head_helper.ResNetBasicHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )


    def forward(self, x):
        x_i = x[0]
        x_m = x[1]
        x_r = x[2]

        x_i = self.s1_i(x_i)
        x_m = self.s1_m(x_m)
        x_r = self.s1_r(x_r)

        x_m_1_3 = x_m[:,0:2]
        x_r_1_3 = x_r[:,0:2]
        x_m_4_7 = x_m[:,3:6]
        x_r_4_7 = x_r[:,3:6]
        x_m_8_11 = x_m[:,7:10]
        x_r_8_11 = x_r[:,7:10]

        x_i = self.s2_fuse(x_i, x_m_1_3, x_r_1_3)

        x_i = self.s2_i(x_i)
        x_m_group_2 = self.s2_m(x_m_4_7)
        x_m_group_3 = self.s2_m(x_m_8_11)
        x_r_group_2 = self.s2_r(x_r_4_7)
        x_r_group_3 = self.s2_r(x_r_8_11)

        x_i = self.s3_fuse(x_i, [x_m_group_2, x_m_group_3], [x_r_group_2, x_r_group_3])

        x = self.s4(x_i)
        x = self.s4_fuse(x)
        x = self.s5(x)
        x = self.head(x)
        
        return x



class FuseIframeToPframe(nn.Module):
    """
    Fuses the information from I-frame to the P-frame. Given the
    feature tensors from I-frame, Motion Vector, and Residual, fuse information from I-frame to MV/Res, then return the concatenated tensors in temporal dimension.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the I-frame input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from I-frame to P-frame.
            fusion_kernel (int): kernel size of the convolution used to fuse 
                from I-frame to P-frame.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseIframeToPframe, self).__init__()
        self.conv_i2p = nn.Conv3d(
            dim_in,
            dim_in // fusion_conv_channel_ratio,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in // fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_i = x[0] #i-frame input
        x_m = x[1] #mv input
        x_r = x[2] #residual input
        fuse = self.conv_i2p(x_i)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)

        x_fuse = torch.cat([fuse, x_m, x_r], 1)
        #cat in channel axis.
        #TODO: add a channel shuffle conv. or all p frame aggregated tensor is the same in the half part of the fuse. 
        return [x_fuse]  



