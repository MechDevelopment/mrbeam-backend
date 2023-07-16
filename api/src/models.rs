use actix_multipart::form::MultipartForm;

#[derive(serde::Deserialize, serde::Serialize, Debug, utoipa::ToSchema)]
pub struct Beam {
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
    pub conf: Option<f32>,
    pub class: f32,
}

pub struct PredictionId {
    pub id: uuid::Uuid,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, utoipa::ToSchema)]
pub struct PredictionUpload {
    pub data: Vec<Beam>,
    pub uuid: String,
}

#[derive(Debug, MultipartForm, utoipa::ToSchema)]
pub struct UploadForm {
    #[schema(format = Binary, value_type = String)]
    pub file: actix_multipart::form::bytes::Bytes,
    #[schema(value_type = Option<String>, example="false")]
    pub save: Option<actix_multipart::form::text::Text<String>>,
}
// #[derive(Debug, sqlx::FromRow, serde::Deserialize, serde::Serialize)]
// #[allow(non_snake_case)]
// pub struct PredictionModel {
//     pub id: uuid::Uuid,
//     pub prediction: Option<serde_json::Value>,
// pub correction: serde_json::Value,
// pub created_at: Option<chrono::DateTime<chrono::Utc>>,
// #[serde(rename = "updatedAt")]
// pub updated_at: Option<chrono::DateTime<chrono::Utc>>,
// }
