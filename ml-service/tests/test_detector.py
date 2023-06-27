import pytest

import numpy as np
import cv2

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