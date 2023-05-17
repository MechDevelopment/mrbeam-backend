use actix_multipart::form::MultipartForm;
use actix_web::{web, App, Error, HttpResponse, HttpServer, Responder};
use chrono::Utc;
use dotenvy::dotenv;
use image::EncodableLayout;
use sqlx::PgPool;
use sqlx::types::Uuid;
use sqlx::{postgres::PgPoolOptions, Pool, Postgres};
use std::env;

use api::models::Beam;
use api::services::MLService;

pub struct AppState {
    db: Pool<Postgres>,
}

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
    data: web::Data<AppState>,
) -> Result<impl Responder, Error> {
    let bytes = form.file.data.as_bytes().to_vec();

    let beams = ml_client.predict(bytes).await.unwrap();

    // let preds = sqlx::query!("SELECT * FROM predictions")
    //     .fetch_all(&data.db)
    //     .await
    //     .expect("Failed to fetch data");
    // println!("Rows: {:#?}", preds);

    // let insert_res = sqlx::query!(
    //     "INSERT INTO predictions (prediction) VALUES ($1)",
    //     sqlx::types::Json(&beams) as _,
    // )
    // .execute(&data.db)
    // .await
    // .expect("FAIL");

    let row: (sqlx::types::Uuid,) = sqlx::query_as(
        "INSERT INTO predictions (prediction) VALUES ($1) RETURNING id")
        .bind(sqlx::types::Json(&beams))
        .fetch_one(&data.db)
        .await
        .expect("Unable to create.");
    
    Ok(HttpResponse::Ok().json(PredictionInfo {
        data: beams,
        uuid: row.0.to_string()
    }))
}

async fn test() -> Result<impl Responder, Error> {
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
            .route("/test", web::get().to(test))
    })
    .bind("127.0.0.1:8001")?
    .run()
    .await
}
