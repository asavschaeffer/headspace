use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use axum::routing::{get, post};
use axum::Router;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::cluster::cluster_documents;
use crate::config::Config;
use crate::embeddings;
use crate::ingestion;
use crate::search;
use crate::storage::Store;

/// Shared application state.
#[derive(Debug, Clone)]
pub struct AppState {
    pub store: Arc<RwLock<Store>>,
    pub config: Arc<Config>,
    pub ingesting: Arc<RwLock<bool>>,
}

/// Creates the API router.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/api/status", get(status))
        .route("/api/ingest", post(ingest))
        .route("/api/documents", get(list_documents))
        .route("/api/document/{id}", get(get_document))
        .route("/api/search", get(search_documents))
        .route("/api/clusters", get(get_clusters))
        .with_state(state)
}

// ---------- Handlers ----------

#[derive(Serialize)]
struct StatusResponse {
    document_count: usize,
    has_embeddings: bool,
    is_ingesting: bool,
    root_path: String,
}

async fn status(State(state): State<AppState>) -> Json<StatusResponse> {
    let store = state.store.read().await;
    let ingesting = *state.ingesting.read().await;
    Json(StatusResponse {
        document_count: store.documents.len(),
        has_embeddings: state.config.has_embeddings(),
        is_ingesting: ingesting,
        root_path: store.root_path.clone(),
    })
}

#[derive(Deserialize)]
struct IngestRequest {
    path: String,
}

#[derive(Serialize)]
struct IngestResponse {
    message: String,
    document_count: usize,
}

async fn ingest(
    State(state): State<AppState>,
    Json(body): Json<IngestRequest>,
) -> Result<Json<IngestResponse>, (StatusCode, String)> {
    // Check if already ingesting
    {
        let ingesting = state.ingesting.read().await;
        if *ingesting {
            return Err((
                StatusCode::CONFLICT,
                "ingestion already in progress".to_string(),
            ));
        }
    }

    // Set ingesting flag
    {
        let mut ingesting = state.ingesting.write().await;
        *ingesting = true;
    }

    let config = state.config.clone();
    let store_lock = state.store.clone();
    let ingesting_flag = state.ingesting.clone();

    let path = PathBuf::from(&body.path);
    if !path.is_dir() {
        let mut ingesting = ingesting_flag.write().await;
        *ingesting = false;
        return Err((
            StatusCode::BAD_REQUEST,
            format!("not a directory: {}", body.path),
        ));
    }

    // Spawn the ingestion task
    tokio::spawn(async move {
        let result = run_ingest(&path, &config, &store_lock).await;

        let mut ingesting = ingesting_flag.write().await;
        *ingesting = false;

        if let Err(e) = result {
            tracing::error!("ingestion failed: {e:?}");
        }
    });

    Ok(Json(IngestResponse {
        message: "ingestion started".to_string(),
        document_count: 0,
    }))
}

/// Runs the full ingestion pipeline.
async fn run_ingest(
    path: &std::path::Path,
    config: &Config,
    store_lock: &Arc<RwLock<Store>>,
) -> eyre::Result<()> {
    tracing::info!(path = %path.display(), "starting ingestion");

    // Crawl directory
    let mut documents = ingestion::crawl(path)?;
    tracing::info!(count = documents.len(), "files discovered");

    // Generate embeddings
    let texts: Vec<String> = documents.iter().map(|d| d.content_preview.clone()).collect();
    let embeddings = embeddings::generate_embeddings(&texts, config).await?;

    for (doc, emb) in documents.iter_mut().zip(embeddings) {
        doc.embedding = emb;
    }
    tracing::info!("embeddings generated");

    // Cluster
    cluster_documents(&mut documents);
    tracing::info!("clustering complete");

    // Save
    let mut store = store_lock.write().await;
    store.root_path = path.to_string_lossy().to_string();
    store.documents = documents;
    store.save(&config.store_path())?;

    tracing::info!(count = store.documents.len(), "ingestion complete");
    Ok(())
}

#[derive(Serialize)]
struct DocumentSummary {
    id: String,
    name: String,
    rel_path: String,
    extension: String,
    content_length: usize,
    cluster_id: i32,
}

async fn list_documents(State(state): State<AppState>) -> Json<Vec<DocumentSummary>> {
    let store = state.store.read().await;
    let summaries: Vec<DocumentSummary> = store
        .documents
        .iter()
        .map(|d| DocumentSummary {
            id: d.id.clone(),
            name: d.name.clone(),
            rel_path: d.rel_path.clone(),
            extension: d.extension.clone(),
            content_length: d.content_length,
            cluster_id: d.cluster_id,
        })
        .collect();
    Json(summaries)
}

async fn get_document(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let store = state.store.read().await;
    let doc = store
        .find_by_id(&id)
        .ok_or(StatusCode::NOT_FOUND)?;

    // Return document without the full embedding vector (too large for JSON response)
    let mut value = serde_json::to_value(doc).unwrap_or_default();
    if let Some(obj) = value.as_object_mut() {
        obj.remove("embedding");
    }

    Ok(Json(value))
}

#[derive(Deserialize)]
struct SearchQuery {
    q: String,
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    20
}

#[derive(Serialize)]
struct SearchResultResponse {
    id: String,
    name: String,
    rel_path: String,
    extension: String,
    content_preview: String,
    score: f64,
    cluster_id: i32,
}

async fn search_documents(
    State(state): State<AppState>,
    Query(query): Query<SearchQuery>,
) -> Json<Vec<SearchResultResponse>> {
    let config = state.config.clone();
    let store = state.store.read().await;

    // Generate query embedding
    let query_embedding = embeddings::generate_query_embedding(&query.q, &config)
        .await
        .unwrap_or_default();

    let results = search::search(&query_embedding, &store.documents, query.limit);

    let response: Vec<SearchResultResponse> = results
        .into_iter()
        .map(|r| SearchResultResponse {
            id: r.document.id,
            name: r.document.name,
            rel_path: r.document.rel_path,
            extension: r.document.extension,
            content_preview: r.document.content_preview.chars().take(500).collect(),
            score: r.score,
            cluster_id: r.document.cluster_id,
        })
        .collect();

    Json(response)
}

#[derive(Serialize)]
struct ClusterPoint {
    id: String,
    name: String,
    rel_path: String,
    extension: String,
    cluster_id: i32,
    x: f64,
    y: f64,
}

async fn get_clusters(State(state): State<AppState>) -> Json<Vec<ClusterPoint>> {
    let store = state.store.read().await;
    let points: Vec<ClusterPoint> = store
        .documents
        .iter()
        .map(|d| ClusterPoint {
            id: d.id.clone(),
            name: d.name.clone(),
            rel_path: d.rel_path.clone(),
            extension: d.extension.clone(),
            cluster_id: d.cluster_id,
            x: d.x,
            y: d.y,
        })
        .collect();
    Json(points)
}
