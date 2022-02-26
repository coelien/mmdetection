import os.path as osp
import mmcv
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

@DATASETS.register_module()
class BallonDataset(CustomDataset):
    def load_annotations(self, ann_file):
        data_infos = mmcv.load(ann_file)
        print((data_infos.values()))
        # load image list from file
        # image_list = mmcv.list_from_file(ann_file)
        # print(image_list)
        data_informations = []

        image_prefix = self.img_prefix

        annotations = []
        images = []
        obj_count = 0
        for idx, v in enumerate(mmcv.track_iter_progress(data_infos.values())):
            filename = v['filename']
            img_path = osp.join(image_prefix, filename)
            height, width = mmcv.imread(img_path).shape[:2]
            data_info = dict(filename=filename, width=width, height=height)

            images.append(dict(
                id=idx,
                file_name=filename,
                height=height,
                width=width))

            bboxes = []
            labels = []
            masks = []
            for _, obj in v['regions'].items():
                assert not obj['region_attributes']
                obj = obj['shape_attributes']
                px = obj['all_points_x']
                py = obj['all_points_y']
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                x_min, y_min, x_max, y_max = (
                    min(px), min(py), max(px), max(py))


                data_anno = dict(
                    #image_id=idx,
                    #id=obj_count,
                    #category_id=0,
                    bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                    #area=(x_max - x_min) * (y_max - y_min),
                    #segmentation=[poly],
                    #iscrowd=0
                )
                annotations.append(data_anno)
                obj_count += 1

        coco_format_json = dict(
            images=images,
            annotations=annotations,
            categories=[{'id':0, 'name': 'balloon'}])
        # mmcv.dump(coco_format_json, out_file)
        return coco_format_json
from mmcv import Config
cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')

from mmdet.apis import set_random_seed, inference_detector, show_result_pyplot

# Modify dataset type and path
cfg.classes = ('balloon',)
cfg.dataset_type = 'BallonDataset'
cfg.data_root = 'balloon/'

cfg.data.test.type = 'BallonDataset'
cfg.data.test.data_root = 'balloon/val'
cfg.data.test.classes = ('balloon',)
cfg.data.test.ann_file = 'via_region_data.json'
cfg.data.test.img_prefix = ''

cfg.data.train.type = 'BallonDataset'
cfg.data.train.data_root = 'balloon/train'
cfg.data.train.classes = ('balloon',)
cfg.data.train.ann_file = 'via_region_data.json'
cfg.data.train.img_prefix = ''

cfg.data.val.type = 'BallonDataset'
cfg.data.val.data_root = 'balloon/val'
cfg.data.val.classes = ('balloon',)
cfg.data.val.ann_file = 'via_region_data.json'
cfg.data.val.img_prefix = ''

# modify num classes of the model in box head
cfg.model.roi_head.bbox_head.num_classes = 1
# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
# cfg.load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './balloon_exps'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'mAP'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 12
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 12
cfg.load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)


# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')


from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector


# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(
    cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)
