import math
from typing import List

import cv2
from detection.algorithm_yolact import YOLACT
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort


class Singleton(object):
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instances[cls]


class DetectionManager(Singleton):
    def __init__(self):
        # Singleton.__init__()
        self.algorithms = [YOLACT()]
        self.cfg = get_config()
        self.cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deep_sort = DeepSort(self.cfg.DEEPSORT.REID_CKPT,
                                  max_dist=self.cfg.DEEPSORT.MAX_DIST, min_confidence=self.cfg.DEEPSORT.MIN_CONFIDENCE,
                                  nms_max_overlap=self.cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                  max_iou_distance=self.cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                  max_age=self.cfg.DEEPSORT.MAX_AGE, n_init=self.cfg.DEEPSORT.N_INIT,
                                  nn_budget=self.cfg.DEEPSORT.NN_BUDGET,
                                  use_cuda=True)

        self.tracks = {}
        self.tracks_length = 10
        self.detect_range = [(120, 250), (556, 306)]
        self.detect_out = 0
        self.detect_in = 0
        self.detect_tracks = 5

    def run_detect(self, id, img):
        classes, scores, boxes = self.algorithms[id].detect(img)
        tracking_result = self.algorithms[id].object_tracking(self.deep_sort, classes, scores, boxes, img)
        self.save_track(tracking_result)
        self.detect_count(tracking_result)
        img = self.algorithms[id].visualize(img, tracking_result, self.tracks, (self.detect_out, self.detect_in))
        cv2.rectangle(img, self.detect_range[0], self.detect_range[1], (0, 255, 0), 2)
        return img

    def save_track(self, tracking_result):
        for value in list(tracking_result):
            x1, y1, x2, y2, track_id = value
            if track_id not in self.tracks:
                self.tracks[track_id] = {'history': [], 'direction': None}
            if len(self.tracks[track_id]['history']) >= self.tracks_length:
                self.tracks[track_id]['history'].pop(0)
            self.tracks[track_id]['history'].append((int((x1 + x2) / 2), int((y1 + y2) / 2)))

    def detect_count(self, tracking_result):
        for value in list(tracking_result):
            x1, y1, x2, y2, track_id = value
            last = self.tracks[track_id]['history'][-1]
            if len(self.tracks[track_id]['history']) >= self.detect_tracks and self.is_in_detect_range(last[0], last[1]):
                direction_result = []
                for idx in range(1, self.detect_tracks):
                    penultimate = self.tracks[track_id]['history'][-idx]
                    direction_result.append(self.get_direction(last, penultimate))
                direction = max(set(direction_result), key=direction_result.count)
                if direction == 'out' and direction != self.tracks[track_id]['direction']:
                    if self.tracks[track_id]['direction'] == 'in':
                        self.detect_in -= 1
                    self.tracks[track_id]['direction'] = 'out'
                    self.detect_out += 1
                elif direction == 'in' and direction != self.tracks[track_id]['direction']:
                    if self.tracks[track_id]['direction'] == 'out':
                        self.detect_out -= 1
                    self.tracks[track_id]['direction'] = 'in'
                    self.detect_in += 1

    def get_direction(self, last, penultimate):
        angle = math.atan2(last[1] - penultimate[1], last[0] - penultimate[0]) * (180 / math.pi)
        if 0 < angle < 180:
            return 'out'
        elif -180 < angle < -1:
            return 'in'
        return None

    def is_in_detect_range(self, x, y):
        if (self.detect_range[0][0] < x < self.detect_range[1][0]) and (
                self.detect_range[0][1] < y < self.detect_range[1][1]):
            return True
        return False

    def list_algorithms(self):
        for i, v in enumerate(self.algorithms):
            print(str(i) + '. ' + v.name)
