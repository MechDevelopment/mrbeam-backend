# mrbeam.app backend
Backend services for [mrbeam.app](https://mrbeam.app).

## Main API
Source: [`api/`](api/)

## Model inference API
Source: [`ml-service/`](ml-service/)

The [Aqueduct](https://github.com/avito-tech/aqueduct) was chosen for the model inference. At the moment, [YOLOv5](https://github.com/ultralytics/yolov5) is used as the baseline model. As it is supposed to be done in aqueduct, model inference is divided into several tasks: loading an image, preprocessing it, running model, and postprocessing the results. The code for this can be found [here](ml-service/app/flow.py). 

Environment variable `$ML_MODEL_WEIGHTS` allows you to specify which specific weights to use. By default, weights in the onnx format from the `weights` folder will be used. You can find the pretrained weights on google drive: [weights](https://drive.google.com/drive/folders/1R_gJcH66CDaGftfGHO9yuvFFMe7-ZDNU?usp=sharing).

#### TODO: 
- [ ] Add an option to download weights from `W&B` and/or `MLFLow` model registries.
- [ ] Support of other models than `YOLOv5.`
