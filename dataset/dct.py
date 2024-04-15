import os
import os.path as osp
import shutil
import subprocess
import time

import typer
from tqdm import tqdm
from typing_extensions import Annotated

app = typer.Typer()


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

    # 读取


if __name__ == "__main__":
    app()
