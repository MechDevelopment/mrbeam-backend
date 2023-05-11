#[derive(serde::Deserialize, serde::Serialize)]
pub struct Beam {
    xmin: f32,
    ymin: f32,
    xmax: f32,
    ymax: f32,
    confidence: f32,
    name: String,
}