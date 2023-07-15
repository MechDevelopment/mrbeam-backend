import os
from pydantic import BaseSettings
from dotenv import load_dotenv
load_dotenv()


class Settings(BaseSettings):
    app_name: str = "MrBeam inference service"

    app_dir: str = os.path.abspath(os.path.dirname(__file__))
    project_root: str = os.path.abspath(os.path.join(app_dir, os.pardir))
    port: int = 8011
    environment: str = "dev"

    model_weights: str = os.path.join(project_root, "weights/best_s_june.onnx")
    model_conf: float = 0.4
    image_size: int = 640

    class Config:
        env_prefix = 'ML_'

settings = Settings()