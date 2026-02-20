//! Headspace - Computer information organization and retrieval for humans and LLMs.
//!
//! This application provides semantic search and organization of local files through:
//!
//! - **Ingestion**: Discovers and indexes files from watched directories
//! - **Enrichment**: Extracts metadata, summaries, and topics using LLM providers
//! - **Embeddings**: Generates vector embeddings for semantic similarity
//! - **Clustering**: Groups related documents using HDBSCAN
//! - **Search**: Provides semantic search via cosine similarity
//!
//! # Architecture
//!
//! The crate is organized into the following modules:
//!
//! - [`api`]: HTTP API endpoints and WebSocket progress streaming
//! - [`config`]: Environment-based configuration
//! - [`storage`]: SQLite-backed document persistence
//! - [`cortex`]: Content extraction and metadata enrichment pipelines
//! - [`embeddings`]: Vector embedding generation via NVIDIA NIM
//! - [`ingestion`]: File discovery and content hashing
//! - [`cluster`]: Document clustering and 2D projection
//! - [`search`]: Semantic similarity search

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
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // Load .env and config
    if let Err(e) = dotenvy::dotenv() {
        tracing::warn!("failed to load .env: {}", e);
    }
    let config = config::Config::from_env()?;

    tracing::info!(port = config.port, "starting headspace");

    if config.has_embeddings() {
        tracing::info!(model = %config.embedding_model, "embedding API key found");
    } else {
        tracing::warn!("no embedding API key (EMBEDDING_API_KEY or NVIDIA_API_KEY); search will be limited");
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
        last_ingest_error: Arc::new(RwLock::new(None)),
        progress_tx: Arc::new(tokio::sync::broadcast::channel(64).0),
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
