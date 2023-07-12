use std::io::Error;

use awsregion::Region;
use s3::bucket::Bucket;
use s3::creds::Credentials;

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

        let res = self
            .http_client
            .post(&predict_url)
            .multipart(form)
            .send()
            .await?;

        let beams: Vec<Beam> = res.json().await.unwrap();
        Ok(beams)
    }
}

pub struct ImageStorage {
    s3_bucket: Bucket,
}

impl ImageStorage {
    pub fn new(
        bucket_name: String,
        endpoint: String,
        access_key: String,
        secret_key: String,
    ) -> Self {
        let s3_bucket = Bucket::new(
            &bucket_name,
            Region::Custom {
                region: "".to_owned(),
                endpoint: endpoint.to_owned(),
            },
            Credentials {
                access_key: Some(access_key.to_owned()),
                secret_key: Some(secret_key.to_owned()),
                security_token: None,
                session_token: None,
                expiration: None,
            },
        )
        .unwrap()
        .with_path_style();

        Self { s3_bucket }
    }

    pub async fn upload_image(
        &self,
        image: Vec<u8>,
        filename: String,
    ) -> Option<String> {
        let res = self.s3_bucket.head_object(filename.to_owned()).await;

        match res {
            Err(s3::error::S3Error::Http(404, _)) => {
                self.s3_bucket
                    .put_object(filename.to_owned(), &image)
                    .await
                    .unwrap();
                return Some(filename);
            }
            Err(_) => return None,
            Ok(_) => {
                return Some(filename);
            }
        };
        //     actix_web::rt::time::sleep(std::time::Duration::from_secs(2)).await;
    }
}
