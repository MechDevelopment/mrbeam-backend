use actix_multipart::form::MultipartForm;
use actix_web::{web, App, Error, HttpResponse, HttpServer, Responder};
use chrono::Utc;
use dotenvy::dotenv;
use image::EncodableLayout;
use sqlx::types::Uuid;
use sqlx::PgPool;
use sqlx::{postgres::PgPoolOptions, Pool, Postgres};
use std::env;
use std::str::FromStr;

use awsregion::Region;
use s3::bucket::{self, Bucket};
use s3::creds::Credentials;

use api::models::{PredictionId, PredictionUpload};
use api::services::{ImageStorage, MLService};

pub struct AppState {
    db: Pool<Postgres>,
}
#[derive(Debug, MultipartForm)]
struct UploadForm {
    file: actix_multipart::form::bytes::Bytes,
    save: Option<actix_multipart::form::text::Text<String>>,
}

async fn health() -> impl Responder {
    HttpResponse::Ok()
}

async fn predict(
    MultipartForm(form): MultipartForm<UploadForm>,
    image_storage: web::Data<ImageStorage>,
    ml_client: web::Data<MLService>,
    data: web::Data<AppState>,
) -> Result<impl Responder, Error> {
    let bytes = form.file.data.as_bytes().to_vec();

    let beams = ml_client.predict(bytes.clone()).await.unwrap();

    let row: (uuid::Uuid,) =
        sqlx::query_as("INSERT INTO predictions (prediction) VALUES ($1) RETURNING id")
            .bind(sqlx::types::Json(&beams))
            .fetch_one(&data.db)
            .await
            .expect("Unable to create.");

    let filename = "test.jpg".to_string();
    let _filename = filename.clone();

    // TODO (vpvpvpvp): Add gracefull shutdown!
    tokio::task::spawn(async move {
        let image_storage = image_storage.clone();
        let res = image_storage.upload_image(bytes, _filename).await.unwrap();
    });

    Ok(HttpResponse::Ok().json(PredictionUpload {
        data: beams,
        uuid: row.0.to_string(),
    }))
}

async fn correct(
    id: web::Path<String>,
    body: web::Json<PredictionUpload>,
    data: web::Data<AppState>,
) -> Result<impl Responder, Error> {
    let id = uuid::Uuid::from_str(&id.into_inner()).unwrap();
    let query_result =
        sqlx::query_as!(PredictionId, "SELECT id FROM predictions WHERE id = $1", id)
            .fetch_one(&data.db)
            .await;

    if query_result.is_err() {
        return Ok(HttpResponse::NotFound());
    }

    let query_result = sqlx::query_as!(
        PredictionId,
        "UPDATE predictions SET correction = $1 WHERE id = $2 RETURNING id",
        serde_json::json!(&body.data),
        id
    )
    .fetch_one(&data.db)
    .await
    .unwrap();

    Ok(HttpResponse::Ok())
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    dotenv().expect(".env file not found");

    let image_storage = web::Data::new(ImageStorage::new(
        "mrbeam".to_string(),
        "http://127.0.0.1:9000".to_string(),
        "EARfO36y2uaGLW3H".to_string(),
        "CKPmTIbMQHHRInAaq8F8L6YXY22x3Idm".to_string(),
    ));

    let ml_client = web::Data::new(MLService::new(String::from(
        env::var("ML_SERVICE").unwrap_or("http://127.0.0.1:8000".to_string()),
    )));
    let db_url = env::var("DATABASE_URL").expect("Database is unavailable.");
    let db_pool = PgPool::connect(&db_url)
        .await
        .expect("Failed to connect to the database.");

    HttpServer::new(move || {
        App::new()
            .app_data(ml_client.clone())
            .app_data(image_storage.clone())
            .app_data(web::Data::new(AppState {
                db: db_pool.clone(),
            }))
            .route("/health", web::get().to(health))
            .route("/predict", web::post().to(predict))
            .route("/correct/{id}", web::post().to(correct))
    })
    .bind("127.0.0.1:8001")?
    .run()
    .await
}
