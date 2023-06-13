from typing import List, Tuple, Optional, Dict
import numpy as np
import torch

from aqueduct import (
    BaseTask,
    BaseTaskHandler,
    Flow,
    FlowStep,
)

from detector import default_producer


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

