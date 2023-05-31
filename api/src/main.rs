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

use api::models::Beam;
use api::services::MLService;

pub struct AppState {
    db: Pool<Postgres>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct PredictionInfo {
    data: Vec<Beam>,
    uuid: String,
}
#[derive(Debug, sqlx::FromRow, serde::Deserialize, serde::Serialize)]
#[allow(non_snake_case)]
pub struct PredictionModel {
    pub id: uuid::Uuid,
    pub prediction: Option<serde_json::Value>,
    // pub correction: serde_json::Value,
    // pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    // #[serde(rename = "updatedAt")]
    // pub updated_at: Option<chrono::DateTime<chrono::Utc>>,
}

pub struct CorrectionModel {
    pub id: uuid::Uuid,
    pub correction: sqlx::types::Json<Vec<Beam>>
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
    data: web::Data<AppState>,
) -> Result<impl Responder, Error> {
    let bytes = form.file.data.as_bytes().to_vec();

    let beams = ml_client.predict(bytes).await.unwrap();

    let row: (uuid::Uuid,) =
        sqlx::query_as("INSERT INTO predictions (prediction) VALUES ($1) RETURNING id")
            .bind(sqlx::types::Json(&beams))
            .fetch_one(&data.db)
            .await
            .expect("Unable to create.");

    Ok(HttpResponse::Ok().json(PredictionInfo {
        data: beams,
        uuid: row.0.to_string(),
    }))
}

async fn correct(
    id: web::Path<String>,
    body: web::Json<PredictionInfo>,
    data: web::Data<AppState>,
) -> Result<impl Responder, Error> {
    let id = uuid::Uuid::from_str(&id.into_inner()).unwrap();
    let query_result: Result<PredictionModel, sqlx::Error> = sqlx::query_as!(
        PredictionModel,
        "SELECT id, prediction FROM predictions WHERE id = $1",
        id
    )
    .fetch_one(&data.db)
    .await;

    if query_result.is_err() {
        return Ok(HttpResponse::NotFound());
    }

    dbg!(query_result.unwrap());

    let query_result: (uuid::Uuid,) =
        sqlx::query_as("UPDATE predictions SET correction = $1 WHERE id = $2 RETURNING id")
            .bind(sqlx::types::Json(&body.data))
            .bind(id)
            .fetch_one(&data.db)
            .await
            .unwrap();
        
    Ok(HttpResponse::Ok())
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    dotenv().expect(".env file not found");

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
