import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    app_name: str = "MrBeam inference service"

    app_dir: str = os.path.abspath(os.path.dirname(__file__))
    project_root: str = os.path.abspath(os.path.join(app_dir, os.pardir))
    port: int = 8011
    environment: str = "development"

    model_weights: str = os.path.join(project_root, "weights/yolo5l_640.pt")
    model_conf: float = 0.4
    model_image_size: int = 640

    class Config:
        env_file = ".env"


settings = Settings()
