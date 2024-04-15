import os.path as osp
from typing import List, Union

from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class AlignedDualSpectrumDataset(CocoDataset):

    # 严格来说，MultiSpectral 只是双光谱，它一般包括 RGB 和 IR 两个波段的数据
    # 但是学界对双光谱目标检测似乎没有一个很恰当的定义，
    # 可参考：https://paperswithcode.com/task/multispectral-object-detection

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """

        modify from mmdet.datasets.coco.CocoDataset.parse_data_info

        Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = dict()

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        # img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        # 移除了 img_path 的配置，添加了 img_path_rgb 和 img_path_ir 的配置
        img_path_rgb = osp.join(self.data_prefix['img_path_rgb'], img_info['file_name'])  # 多光谱数据的路径
        img_path_ir = osp.join(self.data_prefix['img_path_ir'], img_info['file_name'])  # 多光谱数据的路径

        data_info['img_path_ir'] = img_path_ir
        data_info['img_path_rgb'] = img_path_rgb
        data_info['img_path'] = [img_path_rgb, img_path_ir]
        # 此时,data_info['img_path'] 是一个列表，包含了 RGB 和 IR 两个波段的数据路径
        # 因此可以使用 mmdet.datasets.transforms.loading.LoadMultiChannelImageFromFiles 来加载多光谱数据

        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None

        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info


# 配对的LLVIP 数据集
@DATASETS.register_module()
class LLVIPDataset(AlignedDualSpectrumDataset):
    METAINFO = {
        'classes': ('person',),  # palette is a list of color tuples, which is used for visualization.
        'palette': [(220, 20, 60)]
    }


# 配对的 FLIR 数据集
@DATASETS.register_module()
class FLIRDataset(AlignedDualSpectrumDataset):
    METAINFO = {
        'classes': ("bicycle", "car", "dog", "person"),
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)]
    }

#  添加你自己的数据集
# @DATASETS.register_module()
# class YourCustomDataset(MultiSpectralDataset):
#     # 示例 添加自己的指定名称和类别的数据集
#     # METAINFO 的定义请参考，CocoDataset
#     # mmdetection3.x 舍弃了 classes=() 注册数据集类型的配置
#     METAINFO = {
#         'classes': ("class1", "class2", "class3"),
#         'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142)]
#     }
