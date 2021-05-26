import math
from typing import List
import torch
import cv2
from detection.algorithm_mrcnn import MRCNN
from detection.algorithm_yolact import YOLACT
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import requests

token = 'XXXXXXX'
my_id = "XXXXXXX"


def send_message(message, chat_id=my_id):
    url = "https://api.telegram.org/bot" + token + "/sendMessage?chat_id=" + chat_id + "&text=" + message
    response = requests.get(url)
    if not response.ok:
        return False
    else:
        return True


class Singleton(object):
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instances[cls]


class DetectionManager(Singleton):
    def __init__(self):
        # Singleton.__init__()
        # self.algorithms = [MRCNN()]
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
        self.counter = {'out': 0, 'in': 0}
        self.threshold = 4

    def run_detect(self, id, img):
        classes, scores, boxes = self.algorithms[id].detect(img)
        tracking_result = self.algorithms[id].object_tracking(self.deep_sort, classes, scores, boxes, img)
        if tracking_result is not None:
            self.save_track(tracking_result)
            self.detect_count(tracking_result)
            img = self.algorithms[id].visualize(img, tracking_result, self.tracks)
        cv2.putText(img, 'Out: {} | In:{}'.format(self.counter['out'], self.counter['in']), (10, 10), 0, 0.5,
                    (255, 255, 255))
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
            if track_id in self.tracks and len(self.tracks[track_id]['history']) >= self.tracks_length:
                direction = self.get_direction(self.tracks[track_id]['history'][-1],
                                               self.tracks[track_id]['history'][0])
                if direction is not None and direction != self.tracks[track_id]['direction']:
                    self.counter[direction] += 1
                    self.tracks[track_id]['direction'] = direction
                    '''if self.counter['in'] >= self.threshold:
                        send_message(
                            'the number of people has reached the limit [Out:{}, in:{}]'.format(self.counter['out'],
                                                                                                self.counter['in']))'''

    def get_direction(self, last, first):
        angle = math.atan2(last[1] - first[1], last[0] - first[0]) * (180 / math.pi)
        if 0 < angle < 180:
            return 'in'
        elif -180 < angle < -1:
            return 'out'
        return None

    def list_algorithms(self):
        for i, v in enumerate(self.algorithms):
            print(str(i) + '. ' + v.name)
