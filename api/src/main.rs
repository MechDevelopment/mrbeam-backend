use actix_multipart::form::MultipartForm;
use actix_web::{web, App, Error, HttpResponse, HttpServer, Responder};
use dotenvy::dotenv;
use image::EncodableLayout;
use sqlx::pool::PoolOptions;
use sqlx::postgres::PgConnectOptions;
use sqlx::{ConnectOptions, Pool, Postgres};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;
use tracing_actix_web::TracingLogger;

use std::env;
use std::str::FromStr;
use std::time::Duration;

use api::models::{Beam, PredictionId, PredictionUpload, UploadForm};
use api::services::{ImageStorage, MLService};
use api::telemetry;

pub struct AppState {
    db: Pool<Postgres>,
}

async fn health() -> impl Responder {
    HttpResponse::Ok()
}

#[tracing::instrument(
    name = "New prediction",
    skip(form, image_storage, ml_client, data)
)]
#[utoipa::path(
    post,
    path="/api/v1/predict",
    tag="predict",
    responses(
        (status = 200, description = "List with recognized beam elements.", body = [PredictionUpload])
    ),
    request_body(
        content = UploadForm, 
        description = "Returns the list with recognized beam elements.
            \n\nThe coordinates of each box have the following format:
            \n\n(`xmin`, `ymin`) - upper left point
            \n\n(`xmax`, `ymax`) - bottom right point
            \n\nThese coordinates are normalized from 0 to 1.
            \n\nTo convert the coordinates into absolute values relative to the shape of the image, 
            multiply `xmin`, `xmax` by the image length and `ymin`, `ymax` by the height.", 
        content_type = "multipart/form-data"),
)]
async fn predict(
    MultipartForm(form): MultipartForm<UploadForm>,
    image_storage: web::Data<ImageStorage>,
    ml_client: web::Data<MLService>,
    data: web::Data<AppState>,
) -> Result<impl Responder, Error> {
    if !form
        .file
        .content_type
        .unwrap()
        .to_string()
        .starts_with("image/")
    {
        return Ok(HttpResponse::BadRequest().body("The file is not an image."));
    }

    let bytes = form.file.data.as_bytes().to_vec();

    let hash = sha256::digest_bytes(&bytes);
    let filename = format!(
        "{}.{}",
        hash,
        std::path::Path::new(&form.file.file_name.unwrap())
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap()
            .to_owned()
    );

    let beams = ml_client.predict(bytes.clone()).await;
    if beams.is_err() {
        tracing::error!("{:?}", beams.err().unwrap());
        return Ok(HttpResponse::InternalServerError()
            .body("Error during image processing. Try again later."));
    }
    let beams = beams.unwrap();
    let row: (uuid::Uuid,) =
        sqlx::query_as("INSERT INTO predictions (prediction, image) VALUES ($1, $2) RETURNING id")
            .bind(sqlx::types::Json(&beams))
            .bind(&filename)
            .fetch_one(&data.db)
            .await
            .expect("Unable to add new prediction.");

    // TODO (vpvpvpvp): Add gracefull shutdown!
    let _filename = filename.clone();
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

    telemetry::init_telemetry();

    let image_storage = web::Data::new(ImageStorage::new(
        env::var("MINIO_BUCKET").unwrap().to_string(),
        env::var("MINIO_URL").unwrap().to_string(),
        env::var("MINIO_ROOT_USER").unwrap().to_string(),
        env::var("MINIO_ROOT_PASSWORD").unwrap().to_string(),
    ));

    let ml_client = web::Data::new(MLService::new(String::from(
        env::var("ML_SERVICE_URL").unwrap(),
    )));

    let options = PgConnectOptions::new().disable_statement_logging().clone();

    let db_pool = PoolOptions::default()
        .acquire_timeout(Duration::from_secs(20))
        .connect_with(options)
        .await
        .expect("Failed to connect to the database.");

    while let Err(e) = sqlx::migrate!("./migrations").run(&db_pool).await {
        tracing::error!("Failed to run migrations: {}", e);
        tokio::time::sleep(Duration::from_secs(30)).await;
    }

    #[derive(OpenApi)]
    #[openapi(
        paths(
            predict
        ),
        components(
            schemas(Beam, UploadForm, PredictionUpload)
        ),
        tags(
            (name = "predict")
        )
    )]
    struct ApiDoc;
    let openapi = ApiDoc::openapi();

    HttpServer::new(move || {
        App::new()
            .app_data(ml_client.clone())
            .app_data(image_storage.clone())
            .app_data(web::Data::new(AppState {
                db: db_pool.clone(),
            }))
            .wrap(TracingLogger::default())
            .service(
                SwaggerUi::new("/swagger-ui/{_:.*}").url("/api-docs/openapi.json", openapi.clone()),
            )
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
