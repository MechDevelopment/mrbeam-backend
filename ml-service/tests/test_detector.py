import pytest

import numpy as np
import cv2
import torch

from app.settings import Settings
from app import detector


@pytest.fixture()
def settings():
    settings = Settings()
    settings.image_size = 640

    return settings


@pytest.fixture()
def image():
    return cv2.imread("tests/resources/beam.jpg")


def test_given_image_is_then_preprocessed_correctly_for_yolo(settings, image):
    preprocessor = detector.YOLOPreProcess(settings.image_size)
    prep = preprocessor.process(image)
    
    assert isinstance(prep, np.ndarray)
    assert prep.shape == (1, 3, 640, 640)


def test_given_valid_inputs_then_yolo_returns_valid_outputs(settings, image):
    ...


def test_given_valid_raw_outputs_are_then_correctly_postprocessed(image, settings):
    outputs = torch.load("tests/resources/raw_yolo_outputs.pt")
    postprocessor = detector.YOLOPostProcess()

    preds = postprocessor.process(
                outputs, image.shape[:2], (1, 3, settings.image_size, settings.image_size)
            )
    
    assert len(preds) > 0
    for key in ["xmin", "ymin", "xmax", "ymax", "conf", "class"]:
        assert key in preds[0]