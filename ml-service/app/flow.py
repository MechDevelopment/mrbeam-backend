from typing import List, Tuple, Optional, Dict
import numpy as np
import torch

from aqueduct import (
    BaseTask,
    BaseTaskHandler,
    Flow,
    FlowStep,
)

from detector import default_producer, YOLOV5Model


class Task(BaseTask):
    def __init__(
            self,
            image: bytes
    ):
        super().__init__()
        self.image = image
        self.orig_shape: Optional[Tuple[int, int]] = None
        self.padded_shape: Optional[Tuple[int, int]] = None
        self.preprocessed_shape: Optional[Tuple[int, int]] = None
        self.shifts: Optional[Tuple[int, int]] = None
        self.pred: Optional[Tuple[bytes, str]] = None


class ImageLoaderHandler(BaseTaskHandler):
    def __init__(self):
        self._model = default_producer.get_data_loader()

    def handle(self, *tasks: Task):
        for task in tasks:
            task.image, task.padded_shape = self._model.process(task.image)


class YOLOPreProcessHandler(BaseTaskHandler):
    def __init__(self):
        self._model = default_producer.get_pre_proc()

    def handle(self, *tasks: Task):
        for task in tasks:
            task.image, task.preprocessed_shape = self._model.process(task.image)


class ModelHandler(BaseTaskHandler):
    def __init__(self):
        self._model: Optional[YOLOV5Model] = None

    def on_start(self):
        self._model = default_producer.get_oneformer_model()

    def handle(self, *tasks: Task):
        preds = self._model.process_list(
            data=[task.image for task in tasks]
        )
        for pred, task in zip(preds, tasks):
            task.pred = pred
            task.image = None


class YOLOPostProcessHandler(BaseTaskHandler):
    def __init__(self):
        self._model = default_producer.get_post_proc()

    def handle(self, *tasks: Task):
        for task in tasks:
            task.pred = self._model.process(task.pred, task.orig_shape, task.preprocessed_shape, task.padded_shape, task.shifts)


def get_flow() -> Flow:
    return Flow(
        FlowStep(ImageLoaderHandler(), nprocs=1),
        FlowStep(YOLOPreProcessHandler(), nprocs=1),
        FlowStep(ModelHandler(), batch_size=1, nprocs=1),
        FlowStep(YOLOPostProcessHandler(), nprocs=1),
        metrics_enabled=False,
        # mp_start_method='spawn',
    )
