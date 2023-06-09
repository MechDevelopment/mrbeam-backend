import abc
import torch
import numpy as np

import logging

logger = logging.getLogger(__name__)


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
