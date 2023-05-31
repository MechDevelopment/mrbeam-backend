#[derive(serde::Deserialize, serde::Serialize)]
pub struct Beam {
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
    pub confidence: Option<f32>,
    pub name: String,
}