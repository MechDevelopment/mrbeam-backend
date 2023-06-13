import abc
import dataclasses
from typing import Tuple, List
import torch
import numpy as np
import onnxruntime

from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords


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

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, List[int]]:
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

    def _get_model(self):
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
        pass

    def process(
        self,
        data: Tuple[np.ndarray, np.ndarray],
        original_shape: Tuple[int, int],
        pred_shape: Tuple[int, int],
    ) -> torch.Tensor:
        pred, _ = data
        pred = torch.from_numpy(pred).unsqueeze(0)
        pred = non_max_suppression(pred, 0.4, 0.45, None, False, max_det=64)

        pred = pred[0]
        pred[:, :4] = scale_coords(pred_shape, pred[:, :4], original_shape).round()

        return pred


@dataclasses.dataclass
class ModelInfo:
    name: str
    config_path: str
    weights_path: str
    device: str
    image_size: int


class ModelProducer:
    def __init__(self, model_info: ModelInfo):
        self._model_info = model_info

    def get_pre_proc(self) -> YOLOPreProcess:
        return YOLOPreProcess(self._model_info.image_size)

    def get_model(self) -> YOLOV5Model:
        return YOLOV5Model(
            self._model_info.weights_path,
            self._model_info.device
        )

    def get_post_proc(self) -> YOLOPostProcess:
        return YOLOPostProcess()