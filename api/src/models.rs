#[derive(serde::Deserialize, serde::Serialize, Debug)]
pub struct Beam {
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
    pub confidence: Option<f32>,
    pub name: String,
}

pub struct PredictionId {
    pub id: uuid::Uuid,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct PredictionUpload {
    pub data: Vec<Beam>,
    pub uuid: String,
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
