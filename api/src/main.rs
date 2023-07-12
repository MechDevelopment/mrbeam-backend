use actix_multipart::form::MultipartForm;
use actix_web::{web, App, Error, HttpResponse, HttpServer, Responder};
use dotenvy::dotenv;
use image::EncodableLayout;
use sqlx::pool::PoolOptions;
use sqlx::postgres::PgConnectOptions;
use sqlx::{ConnectOptions, PgPool, Pool, Postgres};
use std::env;
use std::str::FromStr;
use std::{net::TcpListener, time::Duration};

use tracing::subscriber::set_global_default;
use tracing_actix_web::TracingLogger;
use tracing_bunyan_formatter::{BunyanFormattingLayer, JsonStorageLayer};
use tracing_subscriber::{layer::SubscriberExt, EnvFilter, Registry};

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

#[tracing::instrument(
    name = "Adding a new prediction",
    skip(form, image_storage, ml_client, data)
)]
async fn predict(
    MultipartForm(form): MultipartForm<UploadForm>,
    image_storage: web::Data<ImageStorage>,
    ml_client: web::Data<MLService>,
    data: web::Data<AppState>,
) -> Result<impl Responder, Error> {
    let bytes = form.file.data.as_bytes().to_vec();

    let hash = sha256::digest_bytes(&bytes);

    let beams = ml_client.predict(bytes.clone()).await?;
    let row: (uuid::Uuid,) =
        sqlx::query_as("INSERT INTO predictions (prediction) VALUES ($1) RETURNING id")
            .bind(sqlx::types::Json(&beams))
            .fetch_one(&data.db)
            .await
            .expect("Unable to add new prediction.");

    let filename = format!(
        "{}.{}",
        hash,
        std::path::Path::new(&form.file.file_name.unwrap())
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap()
            .to_owned()
    );
    let _filename = filename.clone();

    // TODO (vpvpvpvp): Add gracefull shutdown!
    tokio::task::spawn(async move {
        let image_storage = image_storage.clone();
        let res = image_storage.upload_image(bytes, _filename).await;
    });

    Ok(HttpResponse::Ok().json(PredictionUpload {
        data: beams,
        uuid: row.0.to_string(),
    }))
}

#[tracing::instrument(name = "Correct prediction", skip(body, data))]
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

    // let _guard = sentry::init((
    //     env::var("SENTRY_DSN").expect("$SENTRY_DSN must be set."),
    //     sentry::ClientOptions {
    //         release: sentry::release_name!(),
    //         traces_sample_rate: 1.0,
    //         enable_profiling: true,
    //         profiles_sample_rate: 1.0,
    //         ..Default::default()
    //     },
    // ));

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let skipped_fields = vec!["line", "file", "target"];
    let formatting_layer = BunyanFormattingLayer::new("mrbeam-api".into(), std::io::stdout)
        .skip_fields(skipped_fields.into_iter())
        .expect("One of the specified fields cannot be skipped");
    let subscriber = Registry::default()
        .with(env_filter)
        // .with(sentry_tracing::layer())
        .with(JsonStorageLayer)
        .with(formatting_layer);
    set_global_default(subscriber).expect("Failed to set subscriber.");

    let image_storage = web::Data::new(ImageStorage::new(
        env::var("MINIO_BUCKET").unwrap().to_string(),
        env::var("MINIO_URL").unwrap().to_string(),
        env::var("MINIO_ACCESS_KEY").unwrap().to_string(),
        env::var("MINIO_SECRET_KEY").unwrap().to_string(),
    ));

    let ml_client = web::Data::new(MLService::new(String::from(
        env::var("ML_SERVICE_URL").unwrap(),
    )));

    let options = PgConnectOptions::new().disable_statement_logging().clone();

    let db_pool = PoolOptions::default()
        .acquire_timeout(Duration::from_secs(5))
        .connect_with(options)
        .await
        .expect("Failed to connect to the database.");

    while let Err(e) = sqlx::migrate!("./migrations").run(&db_pool).await {
        tracing::error!("Failed to run migrations: {}", e);
        tokio::time::sleep(Duration::from_secs(30)).await;
    }

    HttpServer::new(move || {
        App::new()
            .app_data(ml_client.clone())
            .app_data(image_storage.clone())
            .app_data(web::Data::new(AppState {
                db: db_pool.clone(),
            }))
            .wrap(TracingLogger::default())
            .service(
                web::scope("/api/v1")
                    .route("/health", web::get().to(health))
                    .route("/predict", web::post().to(predict))
                    .route("/correct/{id}", web::post().to(correct)),
            )
    })
    .bind(format!(
        "{}:{}",
        env::var("API_HOST").unwrap().to_string(),
        env::var("API_PORT").unwrap().to_string(),
    ))?
    .run()
    .await
}
