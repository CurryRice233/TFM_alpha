from .detection import Detection
import os
import sys
import numpy as np
import cv2
import random
import colorsys
import warnings

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
        self.coco_model_path = os.path.join(self.root_path, "mask_rcnn_coco.h5")

        class InferenceConfig(coco.CocoConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        self.coco_config = InferenceConfig()
        self.coco_config.display()
        self.model = modellib.MaskRCNN(mode="inference", model_dir=self.model_dir, config=self.coco_config)
        self.model.load_weights(self.coco_model_path, by_name=True)
        self.model.keras_model._make_predict_function()

    def detect(self, image):
        results = self.model.detect([image], verbose=0)
        return results[0]

    def visualize(self, image, result, show_mask=True, show_bbox=True, show_label=True):
        colors = random_colors(len(result['rois']))
        for idx, class_id in enumerate(result['class_ids']):
            if class_id == 1:
                y1, x1, y2, x2 = result['rois'][idx]
                if show_bbox:
                    color = tuple(map(lambda x: x * 255, colors[idx]))
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
                if show_mask:
                    mask = result['masks'][:, :, idx]
                    image = apply_mask(image, mask, colors[idx])
                if show_label:
                    cv2.putText(image, 'Person {:.3f}'.format(result['scores'][idx]), (x1, y1 + 10), 0, 0.3, (255, 255, 255))
        return image
