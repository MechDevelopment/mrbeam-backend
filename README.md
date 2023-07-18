# mrbeam.app backend
Backend services for [mrbeam.app](https://mrbeam.app).

## Components of the architecture
### Main API
Source: [`api/`](api/)

The role of Main API is to:
- Provide access to the model inference service.
- Logging of predictions(optionally) and access to them for further analysis.

Stack:
- Rust
- [Actix-web](https://actix.rs)
- [SQLx](https://github.com/launchbadge/sqlx)
- PostgreSQL
- Minio
- [Sentry](https://sentry.io/)
### Model inference API
Source: [`ml-service/`](ml-service/)

The [Aqueduct](https://github.com/avito-tech/aqueduct) was chosen for the model inference. At the moment, [YOLOv5](https://github.com/ultralytics/yolov5) is used as the baseline model. As it is supposed to be done in aqueduct, model inference is divided into several tasks: loading an image, preprocessing it, running model, and postprocessing the results. The code for this can be found [here](ml-service/app/flow.py). 

Environment variable `$ML_MODEL_WEIGHTS` allows you to specify which specific weights to use. By default, weights in the onnx format from the `weights` folder will be used. You can find some of the pretrained weights on google drive: [weights](https://drive.google.com/drive/folders/1R_gJcH66CDaGftfGHO9yuvFFMe7-ZDNU?usp=sharing).

#### TODO: 
- [ ] Add an option to download weights from `W&B` and/or `MLFLow` model registries.
- [ ] Support of other models than `YOLOv5.`

### Model trainer
#### Dataset
You can find a dataset for recognizing different beam elements in the image on the roboflow page: [beams dataset](https://universe.roboflow.com/victor-penzurov-7b8xd/mrbeam). It contains the following beam elements:

- 0 - The whole beam
- 1 - Distribution load
- 2 - Fixed support
- 3 - Force
- 4 - Momentum
- 5 - Pin support
- 6 - Roller

Over time the dataset will be extended with new samples, of course, but even now it can be used to get some reasonable results. Download the dataset and place it in the `data` folder.

#### Training
*Work in progress*