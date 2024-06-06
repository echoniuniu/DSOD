import torch
from mmengine.model import BaseModule
from torch import nn
from torch.nn import ModuleList

from .resnet import ResNet
from mmdet.registry import MODELS


@MODELS.register_module()
class _ResNet(ResNet):
    """
    ResNet backbone
    """

    def forward_stage(self, idx, x):
        #  ResNet 具有相同的第一(Stage0)阶段， 和 后续的四个阶段
        #  stage_index 为 -1 时，执行stem 阶段， 为 1-4 时，执行后续的四个阶段
        if idx == -1:
            if self.deep_stem:
                x = self.stem(x)
            else:
                x = self.conv1(x)
                x = self.norm1(x)
                x = self.relu(x)
            x = self.maxpool(x)
        else:
            layer_name = self.res_layers[idx]
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
        return x


@MODELS.register_module()
class ExampleFusionBlock(BaseModule):
    """
    Fusion Block
    """

    def __init__(self, dim, **kwargs):
        super(ExampleFusionBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(dim * 2),
            nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1),
        )

    def forward(self, x):
        # 这里只是一个简单的示例，实际上可以根据需求进行修改
        x_cat = torch.cat(x, dim=1)
        x = self.layer(x_cat)
        return x


@MODELS.register_module()
class ExampleFusion(BaseModule):
    """
    Example Fusion
    """

    def __init__(self, stages, dims, **kwargs):
        super(ExampleFusion, self).__init__()
        self.stages = stages
        self.fusion_layers = ModuleList()
        for idx in stages:
            dim = dims[idx]
            self.fusion_layers.append(ExampleFusionBlock(dim=dim))

    def forward_stage(self, idx, x):
        x = self.fusion_layers[idx](x)
        return x


@MODELS.register_module()
class TSBackbone(BaseModule):
    """
    Two Stream backbone

    """

    def __init__(self, b1, b2, **kwargs):
        super(TSBackbone, self).__init__()
        if b1.type == 'ResNet' and b2.type == 'ResNet':
            b1.type = '_ResNet'
            b2.type = '_ResNet'
        self.fe1 = MODELS.build(b1)  # Feature Extractor1 (fe1)
        self.fe2 = MODELS.build(b2)  # Feature Extractor2 (fe2)

        # 如果没有指定 out_indices， 则使用 fe1 的 out_indices 作为默认值，
        # 一般来说 两个特征提取器的结构是一样的，所以 out_indices 也是一样的
        self.out_indices = kwargs.get('out_indices', None) or self.fe1.out_indices

        # 需要注意， stages 和out_indices的对应关系
        # 一般来说 特征提取器分为 stem 和 res_layers 两部分
        # 在分阶段提取特征时，为了兼容 out_indices，需要把 stem 索引设置为 -1
        ef_stages = list(range(len(self.fe1.res_layers)))
        self.stages = [-1] + ef_stages

        # 默认融合阶段和 特征提取器的阶段一致，如果需要自定义，可以在参数中指定
        self.fusion_stages = kwargs.get('fusion_stages', None) or ef_stages

        if kwargs.get('fusion', None):
            fusion = kwargs['fusion']
            fusion.stages = self.fusion_stages
            self.fusion = MODELS.build(fusion)
        print("backbone build success!")

    def init_weights(self):
        self.fe1.init_weights()
        self.fe2.init_weights()
        if hasattr(self, 'fusion'):
            self.fusion.init_weights()

    def forward(self, x):
        # 重新分离出 Rgb 和 IR 两个通道的数据
        rgb_x = x[:, 0:3, :, :]
        ir_x = x[:, 3:6, :, :]
        #  一般对特征提取器不同阶段对特征进行融合，这需要吧 特征提取器的多个阶段拆开
        #   可以参考：CBNetCBNetV2: https://github.com/VDIGPKU/CBNetV2/blob/main/mmdet/models/backbones/cbnet.py
        #   这里只针对 ResNet 进行定制，其他的需要根据 forward  进行修改

        outs = []

        for idx in self.stages:
            rgb_x = self.fe1.forward_stage(idx, rgb_x)
            ir_x = self.fe2.forward_stage(idx, ir_x)
            if hasattr(self, 'fusion') and idx in self.fusion_stages:
                # 如果有融合模块,并且当前阶段需要融合
                x = self.fusion.forward_stage(idx, (rgb_x, ir_x))
            if idx in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        self.fe1.train(mode)
        self.fe2.train(mode)
        if hasattr(self, 'fusion'):
            self.fusion.train(mode)
