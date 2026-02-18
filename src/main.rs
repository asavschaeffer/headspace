use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod api;
mod cluster;
mod config;
mod cortex;
mod embeddings;
mod ingestion;
mod search;
mod storage;

use std::sync::Arc;

use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

#[tokio::main]
async fn main() -> eyre::Result<()> {
    // Initialize error reporting and tracing
    color_eyre::install()?;
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "headspace=info,tower_http=info".parse().unwrap()),
        )
        .init();

    // Load .env and config
    dotenvy::dotenv().ok();
    let config = config::Config::from_env()?;

    tracing::info!(port = config.port, "starting headspace");

    if config.has_embeddings() {
        tracing::info!("NVIDIA embedding API key found");
    } else {
        tracing::warn!("no embedding API key; search will be limited");
    }

    // Load existing store
    let store = storage::Store::load(&config.store_path())?;
    let documents = store.document_count().unwrap_or(0);
    tracing::info!(documents, "loaded store");

    let state = api::AppState {
        store: Arc::new(RwLock::new(store)),
        config: Arc::new(config.clone()),
        ingesting: Arc::new(RwLock::new(false)),
        last_ingest_stats: Arc::new(RwLock::new(None)),
    };

    // Build routes: API + static frontend
    let app = axum::Router::new()
        .merge(api::router(state))
        .fallback_service(ServeDir::new("frontend"))
        .layer(CorsLayer::permissive());

    let addr = format!("0.0.0.0:{}", config.port);
    tracing::info!(%addr, "server listening");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
