from .detection import Detection
import os
import sys
import numpy as np
import cv2
import random
import colorsys
import warnings
import torch

warnings.simplefilter(action='ignore', category=FutureWarning)


def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255, image[:, :, c])
    return image


class MRCNN(Detection):
    def __init__(self):
        Detection.__init__(self, 'Mask R-CNN')
        self.root_path = os.path.abspath("./")
        # Import Mask RCNN
        sys.path.append(self.root_path)  # To find local version of the library
        from mrcnn import utils
        import mrcnn.model as modellib
        from mrcnn import visualize
        # Import COCO config
        sys.path.append(os.path.join(self.root_path, "coco/"))  # To find local version
        import coco
        # Directory to save logs and trained model
        self.model_dir = os.path.join(self.root_path, "logs")

        # Local path to trained weights file
        self.coco_model_path = os.path.join(self.root_path, "./model/mask_rcnn_coco.h5")

        class InferenceConfig(coco.CocoConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        self.coco_config = InferenceConfig()
        self.coco_config.display()
        self.model = modellib.MaskRCNN(mode="inference", model_dir=self.model_dir, config=self.coco_config)
        self.model.load_weights(self.coco_model_path, by_name=True)
        self.model.keras_model._make_predict_function()

    def detect(self, image):
        results = self.model.detect([image], verbose=0)[0]
        return results['class_ids'], results['scores'], results['rois']

    def object_tracking(self, deep_sort, classes, scores, boxes, image):
        bbox_xywh = []
        confs = []
        for idx in range(len(classes)):
            if classes[idx] == 1:
                y1, x1, y2, x2 = boxes[idx]
                obj = [int((x1 + x2) / 2), int((y1 + y2) / 2), x2 - x1, y2 - y1]
                bbox_xywh.append(obj)
                confs.append(scores[idx])
        if len(bbox_xywh) == 0:
            return None
        return deep_sort.update(torch.Tensor(bbox_xywh), torch.Tensor(confs), image)

    def visualize(self, image, result, tracks, show_mask=True, show_bbox=True, show_label=True):
        for value in list(result):
            x1, y1, x2, y2, track_id = value
            if show_bbox:
                color = (0, 0, 255)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            if show_label:
                cv2.putText(image, 'ID:{}'.format(track_id), (x1, y1 + 10), 0, 0.5, (255, 255, 255))

            track_list = tracks[track_id]['history']
            for idx, track in enumerate(track_list):
                cv2.circle(image, track, radius=2, color=(255, 0, 0), thickness=-1)

        return image
