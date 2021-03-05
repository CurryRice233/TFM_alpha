import os
import sys
import cv2
import torch
import torch.backends.cudnn as cudnn

from detection.detection import Detection
from detection.yolact.yolact import Yolact
from detection.yolact.utils.augmentations import FastBaseTransform
from detection.yolact.layers.output_utils import postprocess
from detection.yolact.utils import timer
from detection.yolact.data import cfg, set_cfg


class YOLACT(Detection):
    def __init__(self):
        Detection.__init__(self, 'YOLACT')
        self.root_path = os.path.abspath("./")
        sys.path.append(self.root_path)
        self.coco_model_path = os.path.join(self.root_path, "dataset/yolact_resnet50_54_800000.pth")

        set_cfg('yolact_resnet50_config')
        with torch.no_grad():
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        self.model = Yolact()
        self.model.load_weights(self.coco_model_path)
        self.model.eval()
        self.model = self.model.cuda()
        self.model.detect.use_fast_nms = True
        self.model.detect.use_cross_class_nms = False
        cfg.mask_proto_debug = False
        cudnn.benchmark = True

        # self.transform = torch.nn.DataParallel(FastBaseTransform()).cuda()

    def detect(self, image):
        frame = torch.from_numpy(image).cuda().float()
        with torch.no_grad():
            # batch = self.transform(torch.stack([frame], 0))
            batch = FastBaseTransform()(frame.unsqueeze(0))
        result = self.model(batch)
        h, w, _ = image.shape
        with timer.env('Postprocess'):
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            t = postprocess(result, w, h, visualize_lincomb=False, crop_masks=True, score_threshold=0.15)
            cfg.rescore_bbox = save
        with timer.env('Copy'):
            idx = t[1].argsort(0, descending=True)[:5]  # number of predictions
            classes, scores, boxes = [x[idx].cpu().detach().numpy() for x in t[:3]]
        return classes, scores, boxes

    def object_tracking(self, deep_sort, classes, scores, boxes, image):
        bbox_xywh = []
        confs = []
        for idx in range(len(classes)):
            if classes[idx] == 0:
                x1, y1, x2, y2 = boxes[idx]
                obj = [int((x1 + x2) / 2), int((y1 + y2) / 2), x2 - x1, y2 - y1]
                bbox_xywh.append(obj)
                confs.append(scores[idx])
        if len(bbox_xywh) == 0:
            return None
        return deep_sort.update(torch.Tensor(bbox_xywh), torch.Tensor(confs), image)

    def visualize(self, image, result, tracks, show_mask=True, show_bbox=True, show_label=True):
        for value in list(result):
            x1, y1, x2, y2, track_id = value
            direction = tracks[track_id]['direction']
            if direction is not None:
                if show_bbox:
                    color = (255, 0, 0)
                    if direction == 'out':
                        color = (0, 255, 255)
                    elif direction == 'in':
                        color = (0, 0, 255)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
                if show_label:
                    cv2.putText(image, 'ID:{}'.format(track_id), (x1, y1 + 10), 0, 0.5, (255, 255, 255))

                track_list = tracks[track_id]['history']
                for idx, track in enumerate(track_list):
                    cv2.circle(image, track, radius=2, color=(255, 0, 0), thickness=-1)

        return image
