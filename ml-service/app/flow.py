from typing import List, Tuple, Optional, Dict
import numpy as np
import torch

from aqueduct import (
    BaseTask,
    BaseTaskHandler,
    Flow,
    FlowStep,
)

from pydantic import BaseSettings
from detector import YOLOV5Model, ImageLoader, YOLOPreProcess, YOLOPostProcess


class Task(BaseTask):
    def __init__(self, image: bytes):
        super().__init__()
        self.image = image
        self.orig_shape: Optional[Tuple[int, int]] = None
        self.padded_shape: Optional[Tuple[int, int]] = None
        self.preprocessed_shape: Optional[Tuple[int, int]] = None
        self.shifts: Optional[Tuple[int, int]] = None
        self.pred: Optional[Tuple[bytes, str]] = None


class ImageLoaderHandler(BaseTaskHandler):
    def __init__(self):
        self._model = ImageLoader()

    def handle(self, *tasks: Task):
        for task in tasks:
            task.image, task.orig_shape = self._model.process(task.image)


class YOLOPreProcessHandler(BaseTaskHandler):
    def __init__(self, image_size):
        self.image_size = image_size
        self._model = YOLOPreProcess(self.image_size)

    def handle(self, *tasks: Task):
        for task in tasks:
            task.image, task.preprocessed_shape = self._model.process(task.image)


class ModelHandler(BaseTaskHandler):
    def __init__(self, weights_path, device: str = "cpu"):
        self.weights_path = weights_path
        self.device = device
        self._model = None

    def on_start(self):
        self._model = YOLOV5Model(self.weights_path, self.device)

    def handle(self, *tasks: Task):
        preds = self._model.process_list(data=[task.image for task in tasks])
        for pred, task in zip(preds, tasks):
            task.pred = pred
            task.image = None


class YOLOPostProcessHandler(BaseTaskHandler):
    def __init__(self):
        self._model = YOLOPostProcess()

    def handle(self, *tasks: Task):
        for task in tasks:
            task.pred = self._model.process(
                task.pred, task.orig_shape, task.preprocessed_shape
            )


def get_flow(settings: BaseSettings) -> Flow:
    return Flow(
        FlowStep(ImageLoaderHandler(), nprocs=1),
        FlowStep(YOLOPreProcessHandler(settings.image_size), nprocs=1),
        FlowStep(ModelHandler(settings.model_weights), batch_size=1, nprocs=1),
        FlowStep(YOLOPostProcessHandler(), nprocs=1),
        metrics_enabled=False,
        # mp_start_method='spawn',
    )
