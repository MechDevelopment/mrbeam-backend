import abc
from typing import Tuple, List
import torch
import numpy as np

from utils.augmentations import letterbox
from utils.general import non_max_suppression


class Yolo5Detector:
    def __init__(self, weights_path: str, conf: float = 0.08):
        self.model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=weights_path,
        )
        self.model.conf = conf

    def __call__(self, image: np.ndarray):
        detections = self.model([image], size=640).pandas().xyxy[0]
        detections[["xmin", "xmax"]] = detections[["xmin", "xmax"]] / image.shape[1]
        detections[["ymin", "ymax"]] = detections[["ymin", "ymax"]] / image.shape[0]
        return detections.to_dict(orient="records")


class YOLOPreProcess:
    def __init__(self, image_size: Tuple[int, int], stride: int = 32, auto: bool = False):
        self._image_size = image_size
        self._stride = stride
        self._auto = auto

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        im = letterbox(image, self._image_size, stride=self._stride, auto=self._auto)[0]
        im = im.transpose((2, 0, 1))
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im)
        im = im.float()
        im /= 255

        im = im.unsqueeze(0)

        return im.numpy(), im.shape