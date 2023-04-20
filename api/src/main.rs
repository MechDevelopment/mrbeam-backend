use image::EncodableLayout;
use reqwest::{multipart, Client};

use actix_multipart::form::MultipartForm;
use actix_web::{web, App, Error, HttpResponse, HttpServer, Responder};

#[derive(serde::Deserialize, serde::Serialize)]
struct Beam {
    xmin: f32,
    ymin: f32,
    xmax: f32,
    ymax: f32,
    confidence: f32,
    name: String,
}

#[derive(serde::Serialize)]
struct PredictionInfo {
    data: Vec<Beam>,
    uuid: String
}

#[derive(Debug, MultipartForm)]
struct UploadForm {
    file: actix_multipart::form::bytes::Bytes,
}

async fn predict(
    MultipartForm(form): MultipartForm<UploadForm>,
    client: web::Data<Client>,
) -> Result<impl Responder, Error> {
    let bytes = form.file.data.as_bytes().to_vec();

    let hash = sha256::digest_bytes(&bytes);

    let beam_file = multipart::Part::bytes(bytes)
        .file_name("beam.png")
        .mime_str("image/png")
        .unwrap();
    
    let form = multipart::Form::new().part("file", beam_file);

    let res = client
        .post("https://vktrpnzrv.fvds.ru/predict")
        .multipart(form)
        .send()
        .await
        .unwrap();

    let beams: Vec<Beam> = res.json().await.unwrap();

    Ok(HttpResponse::Ok().json(
        PredictionInfo {
            data: beams,
            uuid: String::from("ue12-aks8-ca82- 892l"),
        }
    ))
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let client = web::Data::new(Client::new());

    HttpServer::new(move || {
        App::new()
            .app_data(client.clone())
            .route("/predict", web::post().to(predict))
    })
    .bind("127.0.0.1:8001")?
    .run()
    .await
}
