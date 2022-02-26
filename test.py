import mmcv

from mmdet.apis import init_detector, inference_detector,show_result_pyplot
# from test3 import cfg
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# checkpoint_file = 'tutorial_exps/latest.pth'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# model.cfg = cfg
# inference the demo image
# img = 'demo/demo.jpg'
img = 'kitti_tiny/training/image_2/000068.jpeg'
result = inference_detector(model, img)
# model.show_result(img, result)
# Let's plot the result
show_result_pyplot(model, img, result, score_thr=0.3)