use crate::models::Beam;

pub struct MLService {
    http_client: reqwest::Client,
    base_url: String,
}

impl MLService {
    pub fn new(base_url: String) -> Self {
        let http_client = reqwest::Client::new();
        Self {
            http_client,
            base_url,
        }
    }

    pub async fn predict(&self, image: Vec<u8>) -> Result<Vec<Beam>, Box<dyn std::error::Error>> {
        let predict_url = format!("{}/predict", self.base_url);

        let beam_image = reqwest::multipart::Part::bytes(image)
            .file_name("beam.png")
            .mime_str("image/png")
            .unwrap();

        let form = reqwest::multipart::Form::new().part("file", beam_image);

        let res = self.http_client
            .post(&predict_url)
            .multipart(form)
            .send()
            .await
            .unwrap();

        let beams: Vec<Beam> = res.json().await.unwrap();
        Ok(beams)
    }
}
