from locust import HttpUser, task, between
from requests_toolbelt.multipart.encoder import MultipartEncoder


class QuickstartUser(HttpUser):
    wait_time = between(0.5, 1)

    @task
    def predict(self):
        with open('./test.jpg', 'rb') as f:
            image_data = f.read()
            # response = requests.post('http://localhost:8000/predict', files={'file': ('image.jpg', image_data)})
            self.client.post("/predict", files={'file': ('image.jpg', image_data)})