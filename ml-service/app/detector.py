import io
from typing import Tuple, List
import dataclasses
import torch
import numpy as np
import cv2
from PIL import Image
import onnxruntime
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession

from app.utils.augmentations import letterbox
from app.utils.general import non_max_suppression, scale_coords


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
    def __init__(
        self, image_size: Tuple[int, int], stride: int = 32, auto: bool = False
    ):
        self._image_size = image_size
        self._stride = stride
        self._auto = auto

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        im = letterbox(image, self._image_size, stride=self._stride, auto=self._auto)[0]
        im = im.transpose((2, 0, 1))
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im)
        im = im.float()
        im /= 255

        im = im.unsqueeze(0)

        return im.numpy(), im.shape


class YOLOV5Model:
    def __init__(self, model_weights_path: str, device: str):
        self.model_weights_path = model_weights_path
        self.device = torch.device(device)
        (
            self.model,
            self.input_name,
            self.out_name_1,
            self.out_name_2,
        ) = self._get_model()

    def _get_model(self) -> Tuple[InferenceSession, str, str, str]:
        providers = ["CPUExecutionProvider"]
        model = onnxruntime.InferenceSession(
            self.model_weights_path, providers=providers
        )
        return (
            model,
            model.get_inputs()[0].name,
            model.get_outputs()[0].name,
            model.get_outputs()[-1].name,
        )

    def process_list(
        self, data: List[np.ndarray]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        preds, protos = self._process_batch(np.concatenate(data, axis=0))
        return [(preds[i], protos[i]) for i in range(preds.shape[0])]

    def _process_batch(self, data_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y = self.model.run(
            [self.out_name_1, self.out_name_2], {self.input_name: data_batch}
        )
        return y[0], y[1]


class YOLOPostProcess:
    def __init__(self):
        self.keys = ["xmin", "ymin", "xmax", "ymax", "conf", "class"]

    def process(
        self,
        data: Tuple[np.ndarray, np.ndarray],
        original_shape: Tuple[int, int],
        pred_shape: Tuple[int, int],
    ) -> List[dict]:
        pred, _ = data
        pred = torch.from_numpy(pred).unsqueeze(0)
        pred = non_max_suppression(pred, 0.4, 0.45, None, False, max_det=64)

        pred = pred[0]
        pred[:, :4] = scale_coords(pred_shape[2:], pred[:, :4], original_shape).round()
        pred[:, :4] = pred[:, :4] / torch.tensor(original_shape[::-1]).repeat(2)
        pred = [{k: v[i] for i, k in enumerate(self.keys)} for v in pred.tolist()]

        return pred


class ImageLoader:
    def __init__(self):
        self.convert_to_rgb: bool = True

    def process(
        self, im_bytes: bytes
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        img = Image.open(io.BytesIO(im_bytes))
        img = np.array(img)

        if self.convert_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img, img.shape[:2]
