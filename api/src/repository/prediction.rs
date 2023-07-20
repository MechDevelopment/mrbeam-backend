use crate::models::Beam;

pub async fn add_prediction(
    conn: impl sqlx::PgExecutor<'_>,
    beams: &Vec<Beam>,
    filename: &String,
) -> Result<String, sqlx::postgres::PgDatabaseError> {
    let row: (uuid::Uuid,) =
        sqlx::query_as("INSERT INTO predictions (prediction, image) VALUES ($1, $2) RETURNING id")
            .bind(sqlx::types::Json(beams))
            .bind(filename)
            .fetch_one(conn)
            .await
            .expect("Unable to add new prediction.");

    Ok(row.0.to_string())
}
