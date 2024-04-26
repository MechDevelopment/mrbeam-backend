use tracing::subscriber::set_global_default;
use tracing_actix_web::TracingLogger;
use tracing_bunyan_formatter::{BunyanFormattingLayer, JsonStorageLayer};
use tracing_subscriber::{layer::SubscriberExt, EnvFilter, Registry};

pub fn init_telemetry() {

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let skipped_fields = vec![
        "line",
        "file",
        "target",
        "http.scheme",
        "http.flavor",
        "otel.name",
        "otel.kind",
        "http.target",
    ];
    let formatting_layer = BunyanFormattingLayer::new("mrbeam-api".into(), std::io::stdout)
        .skip_fields(skipped_fields.into_iter())
        .expect("One of the specified fields cannot be skipped");
    let subscriber = Registry::default()
        .with(env_filter)
        .with(sentry_tracing::layer())
        .with(JsonStorageLayer)
        .with(formatting_layer);
    set_global_default(subscriber).expect("Failed to set subscriber.");
}
