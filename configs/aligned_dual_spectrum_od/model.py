_base_ = '../deformable_detr/deformable-detr-refine_r50_16xb2-50e_coco.py'

model = dict(
    type='DeformableDETR',
    data_preprocessor=dict(
        type='AlignedDualSpectrumDataPreprocessor',
        # 定制的数据预处理器 详见：mmdet.models.data_preprocessors.data_preprocessor.AlignedDualChannelDataPreprocessor
        mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],  # 将RGB和IR的均值和方差分别填入
        std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        pad_size_divisor=1
    ),
    backbone=dict(
        _delete_=True,
        type='TSBackbone',  # 定制了一个简单的双骨干网络 详见：mmdet.models.backbones.ts_backbone.TSBackbone
        b1=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        ),
        b2=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        ),
        fusion=dict(
            type='ExampleFusion',  # 定制的融合模块 详见：mmdet.models.backbones.ts_backbone.ExampleFusion
            dims=[256, 512, 1024, 2048],  # 每个阶段来自特征提取器输出的通道数，这将用于融合模块的通道数设置
        )
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4
    ),
)

# 预训练的 Deformable DETR 模型权重
load_from = '/home/ubuntu/workspace/mmdetection310/pretrained_weight/deformable-detr-refine-twostage_r50_16xb2-50e_coco_20221021_184714-acc8a5ff.pth'

detect_anomalous_params=True
find_unused_parameters=True
