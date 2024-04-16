import os
import os.path as osp
import shutil
import subprocess
import time

import typer
from tqdm import tqdm
from typing_extensions import Annotated

app = typer.Typer()

import json


def remove_classes(all_datas, rcs):
    """
    all_datas: list of dict, 每个元素是一个coco数据集标注的字典
    rcs: list of str, 要移除的类别的名称
    """

    def generate_new_categories(old_categories, rc_ids, id_maps):
        """
        old_categories: 原始的类别列表
        rc_ids: 要移除的类别的id
        id_maps: 旧id -> 新id 的映射
        """
        new_categories = []
        for category in old_categories:
            oid = category['id']
            if oid in rc_ids:
                continue
            new_category = category.copy()
            new_category['id'] = id_maps[category['id']]
            new_categories.append(new_category)
        return new_categories

    def generate_new_annotations(old_annotations: list, rc_ids, id_maps):
        """
        old_annotations: 原始的注释列表
        rc_ids: 要移除的类别的id
        id_maps: 旧id -> 新id 的映射
        """

        idx = 1  # 用于记录新的id
        new_annotations = []
        for annotation in old_annotations:
            if annotation['category_id'] in rc_ids:
                continue
            new_annotation = annotation.copy()
            new_annotation['category_id'] = id_maps[annotation['category_id']]
            new_annotation['id'] = idx
            idx += 1  # 更新id
            new_annotations.append(new_annotation)

        return new_annotations

    def generate_id_map_and_rc_id(coco_data):
        # 然后对所有的类别做一个排序， 如果有 标识 is_rc=True 则排在后面
        categorys = coco_data['categories']

        # 对要移除的类别做一个标记，然后根据标价对所有的类别做一个排序，如果有标记的类别排在后面
        ddc = [dict(name=category['name'], id=category['id'], is_rc=int(category['name'] in rcs)) for category in
               categorys]
        ddc = sorted(ddc, key=lambda x: x['is_rc'])

        # 根据 ddc 排序做出 old_id -> new_id 的映射
        id_map = {category['id']: i for i, category in enumerate(ddc) if category['is_rc'] == 0}
        rc_id = [category['id'] for category in ddc if category['is_rc'] == 1]  # 要移除的类别的 id\
        return id_map, rc_id

    assert len(all_datas) > 0, '没有加载到数据'

    id_map, rc_id = generate_id_map_and_rc_id(all_datas[0])  # 所有的注释共用同一个类别

    new_all_datas = []
    for data in all_datas:
        data['categories'] = generate_new_categories(data['categories'], rc_id, id_map)
        data['annotations'] = generate_new_annotations(data['annotations'], rc_id, id_map)
        new_all_datas.append(data)
    return tuple(new_all_datas)


@app.command()
def coco_rc(
        aps: Annotated[
            str, typer.Option(help="list of str, coco标注文件的路径,支持匹配的多个，如 'a.json', 'b.json' ")],
        rcs: Annotated[str, typer.Option(help="list of str, 要移除的类别的名称，如 'dog', 'cat'")],
        sps: Annotated[str, typer.Option(help="list of str, 保存的路径，于传入路径对应，如 'a_new.json','b_new.json',默认保存在aps 输入路径下，并以_new.json 命名]")]=None
):
    aps = aps.replace(" ", '').split(',')
    rcs = rcs.replace(" ", '').split(',')
    if sps is None:
        sps = [ap.replace('.json', '_new.json') for ap in aps]
    else:
        sps = sps.replace(" ", '').split(',')
    all_ap_data = []
    for ap1 in aps:
        with open(ap1, 'r') as f:
            data = json.load(f)
        all_ap_data.append(data)
    new_all_data = remove_classes(all_ap_data, rcs)
    for i, sp in enumerate(sps):
        json.dump(new_all_data[i], open(sp, 'w'))

    print("处理完成！")


@app.command()
def llvip(
        data_dir: Annotated[str, typer.Option(help="LLVIP 数据集的解压路径，请保持原始的文件夹结构")],
        save_dir: Annotated[str, typer.Option(
            help="保存的COCO 格式的json文件路径,默认保存在 {data_dir}/coco_annotations/ 目录下，"
                 "生成文件 train.json 和 test.json")] = 'coco_annotations'):
    tmp_path = osp.join(data_dir, f'tmp_{hex(int(time.time()))}')
    tmp_path_train = osp.join(tmp_path, 'train')
    tmp_path_test = osp.join(tmp_path, 'test')
    os.makedirs(tmp_path, exist_ok=True)
    os.makedirs(tmp_path_train, exist_ok=True)
    os.makedirs(tmp_path_test, exist_ok=True)

    train_file_names = [_.replace('.jpg', '.xml') for _ in os.listdir(osp.join(data_dir, 'visible/train'))]
    test_file_names = [_.replace('.jpg', '.xml') for _ in os.listdir(osp.join(data_dir, 'visible/test'))]

    ann_path = osp.join(data_dir, 'Annotations')
    for file_name in train_file_names:
        source = osp.join(ann_path, file_name)
        assert osp.exists(source), f'{source} not exists'  # 目标标注必须存在
        os.symlink(source, osp.join(tmp_path_train, file_name))

    for file_name in test_file_names:
        source = osp.join(ann_path, file_name)
        assert osp.exists(source), f'{source} not exists'
        os.symlink(source, osp.join(tmp_path_test, file_name))

    save_train_path = osp.join(save_dir, 'train.json')
    save_test_path = osp.join(save_dir, 'test.json')

    print("正在使用 LLVIP 官方提供的转换文件生成 COCO 标注格式的json文件")

    subprocess.run(['python', 'voc2coco.py', '--annotation_path', tmp_path_train, '--json_save_path', save_train_path])
    subprocess.run(['python', 'voc2coco.py', '--annotation_path', tmp_path_test, '--json_save_path', save_test_path])

    # 删除临时文件夹
    os.system(f'rm -rf {tmp_path}')

    print("任务完成\ntrain.json 保存在: ", save_train_path, "\ntest.json 保存在: ", save_test_path)


@app.command()
def flir(
        data_dir: Annotated[str, typer.Option(help="FLIR数据集的解压路径，请保持原始的文件夹结构")],
        image_save_dir: Annotated[str, typer.Option(help="Zhang等人提供的数据集中 RGB 图像和 IR图像被存放在同一个文件夹中，这样不太方"
                                                         "便被读取，将会把他分开,默认存储到 {data_dir} /visible  和 thermal 中")] = None,
        ann_save_dir: Annotated[str, typer.Option(help="保存的COCO 格式的json文件路径,默认保存在 {data_dir}/coco_annotations/ 目录下，"
                                                       "生成文件 train.json 和 test.json")] = 'coco_annotations'):
    # 创建保存注释的文件夹
    ann_save_dir = osp.join(data_dir, ann_save_dir)
    os.makedirs(ann_save_dir, exist_ok=True)
    #  准备文件夹
    image_save_dir = image_save_dir if image_save_dir else data_dir
    vis_image_save_dir = osp.join(image_save_dir, 'visible')
    vis_train = osp.join(vis_image_save_dir, 'train')
    vis_test = osp.join(vis_image_save_dir, 'test')
    ir_image_save_dir = osp.join(image_save_dir, 'thermal')
    ir_train = osp.join(ir_image_save_dir, 'train')
    ir_test = osp.join(ir_image_save_dir, 'test')

    # 如果存在则删除
    os.system(f'rm -rf {vis_image_save_dir}')
    os.system(f'rm -rf {ir_image_save_dir}')
    os.makedirs(vis_image_save_dir, exist_ok=True)
    os.makedirs(ir_image_save_dir, exist_ok=True)
    os.makedirs(vis_train, exist_ok=True)
    os.makedirs(vis_test, exist_ok=True)
    os.makedirs(ir_test, exist_ok=True)
    os.makedirs(ir_train, exist_ok=True)

    # 创建临时文件夹 ,存储注释
    tmp_path = osp.join(data_dir, f'tmp_{hex(int(time.time()))}')
    tmp_path_train = osp.join(tmp_path, 'train')
    tmp_path_test = osp.join(tmp_path, 'test')
    os.makedirs(tmp_path_test, exist_ok=True)
    os.makedirs(tmp_path_train, exist_ok=True)

    # 读取train.txt 和 test.txt 文件,获取训练集和测试集的图片名
    train_txt = osp.join(data_dir, 'align_train.txt')
    test_txt = osp.join(data_dir, 'align_validation.txt')

    assert osp.exists(test_txt) and osp.exists(train_txt), 'train.txt or test.txt not exists, please check the path'

    with open(train_txt, 'r') as f:
        train_list = [line.strip().replace('_PreviewData', '') for line in f]
    with open(test_txt, 'r') as f:
        test_list = [line.strip().replace('_PreviewData', '') for line in f]

    # 为xml文件创建软链接 , 生成coco格式的json文件

    for file_name in train_list:
        xml_path = osp.join(data_dir, 'Annotations', file_name + '_PreviewData.xml')
        assert osp.exists(xml_path), f'{xml_path} not exists, please check the path'
        os.symlink(xml_path, osp.join(tmp_path_train, f'{file_name}.xml'))

    for file_name in test_list:
        xml_path = osp.join(data_dir, 'Annotations', file_name + '_PreviewData.xml')
        assert osp.exists(xml_path), f'{xml_path} not exists, please check the path'
        os.symlink(xml_path, osp.join(tmp_path_test, f'{file_name}.xml'))

    print("正在生成coco格式的json文件...")
    # 生成coco格式的json文件
    subprocess.run(['python', 'voc2coco.py', '--annotation_path', tmp_path_train, '--json_save_path',
                    osp.join(ann_save_dir, 'train.json')])
    subprocess.run(['python', 'voc2coco.py', '--annotation_path', tmp_path_test, '--json_save_path',
                    osp.join(ann_save_dir, 'test.json')])

    os.system(f'rm -rf {tmp_path}')  # 删除临时文件夹

    print("正在迁移图片(使用拷贝)...")
    # 为图片创建软链接
    for file_name in tqdm(train_list):
        vis_path = osp.join(data_dir, 'JPEGImages', file_name + '_RGB.jpg')  # 可见光图片
        ir_path = osp.join(data_dir, 'JPEGImages', file_name + '_PreviewData.jpeg')  # 红外图片
        assert osp.exists(vis_path) and osp.exists(
            ir_path), f'{vis_path} or {ir_path} not exists, please check the path'
        shutil.copy2(vis_path, osp.join(vis_train, f'{file_name}.jpeg'))
        shutil.copy2(ir_path, osp.join(ir_train, f'{file_name}.jpeg'))

    for file_name in tqdm(test_list):
        vis_path = osp.join(data_dir, 'JPEGImages', file_name + '_RGB.jpg')  # 可见光图片
        ir_path = osp.join(data_dir, 'JPEGImages', file_name + '_PreviewData.jpeg')  # 红外图片
        assert osp.exists(vis_path) and osp.exists(
            ir_path), f'{vis_path} or {ir_path} not exists, please check the path'
        shutil.copy2(vis_path, osp.join(vis_test, f'{file_name}.jpeg'))
        shutil.copy2(ir_path, osp.join(ir_test, f'{file_name}.jpeg'))

    print(f"任务完成\n"
          f"注释文件保存在: {ann_save_dir}，训练集: train.json 测试集: test.json\n"
          f"可见光图片保存在: {vis_image_save_dir} 测试集: train/ 训练集: test/ \n"
          f"红外图片保存在: {ir_image_save_dir} 测试集: train/ 训练集: test/  \n"
          )


if __name__ == "__main__":
    app()
