import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    app_name: str = "MrBeam inference service"

    app_dir: str = os.path.abspath(os.path.dirname(__file__))
    project_root: str = os.path.abspath(os.path.join(app_dir, os.pardir))
    weights_path: str = os.path.join(project_root, "weights/yolo5l_640.pt")


settings = Settings()
