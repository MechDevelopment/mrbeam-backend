use actix_multipart::form::MultipartForm;
use actix_web::{web, App, Error, HttpResponse, HttpServer, Responder};
use image::EncodableLayout;
use reqwest::{multipart, Client};

use api::services::MLService;
use api::models::Beam;

#[derive(serde::Serialize)]
struct PredictionInfo {
    data: Vec<Beam>,
    uuid: String,
}

#[derive(Debug, MultipartForm)]
struct UploadForm {
    file: actix_multipart::form::bytes::Bytes,
}

async fn health() -> impl Responder {
    HttpResponse::Ok()
}

async fn predict(
    MultipartForm(form): MultipartForm<UploadForm>,
    ml_client: web::Data<MLService>,
) -> Result<impl Responder, Error> {
    let bytes = form.file.data.as_bytes().to_vec();

    let beams = ml_client.predict(bytes).await.unwrap();

    Ok(HttpResponse::Ok().json(PredictionInfo {
        data: beams,
        uuid: String::from("ue12-aks8-ca82- 892l"),
    }))
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let ml_client = web::Data::new(MLService::new(String::from(
        "https://vktrpnzrv.fvds.ru",
    )));

    HttpServer::new(move || {
        App::new()
            .app_data(ml_client.clone())
            .route("/health", web::get().to(health))
            .route("/predict", web::post().to(predict))
    })
    .bind("127.0.0.1:8001")?
    .run()
    .await
}
