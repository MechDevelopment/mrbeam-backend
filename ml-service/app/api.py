import time
from io import BytesIO
from typing import List

from fastapi import FastAPI, HTTPException, Request, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
from PIL import Image, ImageOps

from app.detector import Yolo5Detector
from app.settings import settings


app = FastAPI(title=settings.app_name, description=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
detector = Yolo5Detector(weights_path=settings.weights_path)


class Detections(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    name: str

    class Config:
        schema_extra = {
            "example": {
                "xmin": 0.0725200035568445,
                "ymin": 0.4295076423859501,
                "xmax": 0.20572580403892066,
                "ymax": 0.7179910437649034,
                "confidence": 0.9392452836036682,
                "name": "zadelka",
            }
        }


@app.get("/health")
def ping():
    return "OK"


@app.post("/predict", response_model=List[Detections])
def predict(file: bytes = File(...)):
    """Returns the list with recognized beam elements.

    The coordinates of each box have the following format:
    
    (`xmin`, `ymin`) - upper left point

    (`xmax`, `ymax`) - bottom right point

    These coordinates are normalized from 0 to 1.
    To convert the coordinates into absolute values relative to the shape of the image, 
    multiply `xmin`, `xmax` by the image length and `ymin`, `ymax` by the height.

    """
    start = time.time()

    try:
        image = Image.open(BytesIO(file)).convert("RGB")
        image = ImageOps.exif_transpose(image)
        image = np.array(image, dtype=np.uint8)
        detections = detector(image)
    except Exception as e:
        print(e)
        raise HTTPException(500, detail={"status": "Unable to process image"})

    print(f"inference finished in {time.time() - start}")

    return detections
