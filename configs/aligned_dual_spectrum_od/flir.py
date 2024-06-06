_base_ = 'model.py'  # 模型配置文件

data_root = 'you_flir_dataset_path'  # 数据集的根目录

model = dict(
    bbox_head=dict(num_classes=3)
)

dataset_type = 'FLIRDataset'
backend_args = None
train_pipeline = [
    dict(type='LoadAlignedDualSpectrumImageFormImage', backend_args=backend_args,
         color_type='color',
         # 默认使用color,在处理单通道灰度图时，会自动转换为3通道
         # 详见：mmdet.datasets.transforms.loading.LoadAlignedDualChannelImageFormImage
         ),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadAlignedDualSpectrumImageFormImage', backend_args=backend_args, color_type='color'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='coco_annotations/train_new.json',  # LLVIP 数据集的训练集的注释
        data_prefix=dict(
            _delete_=True,
            img_path_rgb='visible/train/',  # 训练集的可见光 RGB 图像路径
            img_path_ir='thermal/train/'  # 训练集的红外图 IR 像路径
        ),
        pipeline=train_pipeline,  # 替换为 对齐双通道数据的pipeline
    )
)

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='coco_annotations/test_new.json',  # LLVIP 数据集的测试集的注释，因为没有 val 部分，所以使用 test.json
        data_prefix=dict(
            _delete_=True,
            img_path_rgb='visible/test/',  # 测试集的可见光 RGB 图像路径
            img_path_ir='thermal/test/'  # 测试集的红外图 IR 像路径
        ),
        test_mode=True,
        pipeline=test_pipeline,  # 替换为 对齐双通道数据的pipeline
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/coco_annotations/test_new.json',  # LLVIP 数据集的测试集的注释，因为没有 val 部分，所以使用 test.json
    metric='bbox',
    format_only=False,
    backend_args=backend_args
)
test_evaluator = val_evaluator

# learning policy
max_epochs =30
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[20,],
        gamma=0.1)
]
