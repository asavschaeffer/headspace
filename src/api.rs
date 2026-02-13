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
    /// Stats from the most recent ingestion (if any).
    pub last_ingest_stats: Arc<RwLock<Option<IngestStats>>>,
}

/// Statistics from a diff-based ingestion run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestStats {
    pub new_files: usize,
    pub changed_files: usize,
    pub unchanged_files: usize,
    pub deleted_files: usize,
    pub total_documents: usize,
    pub embeddings_generated: usize,
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
    last_ingest: Option<IngestStats>,
}

async fn status(State(state): State<AppState>) -> Json<StatusResponse> {
    let store = state.store.read().await;
    let ingesting = *state.ingesting.read().await;
    let last_ingest = state.last_ingest_stats.read().await.clone();
    Json(StatusResponse {
        document_count: store.documents.len(),
        has_embeddings: state.config.has_embeddings(),
        is_ingesting: ingesting,
        root_path: store.root_path.clone(),
        last_ingest,
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
    let stats_lock = state.last_ingest_stats.clone();

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
        let result = run_ingest(&path, &config, &store_lock, &stats_lock).await;

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

/// Runs the diff-based ingestion pipeline.
///
/// 1. Crawl directory → discover files with content hashes
/// 2. Diff against existing store (new / changed / unchanged / deleted)
/// 3. Embed ONLY new + changed files
/// 4. Merge results into store
/// 5. Re-cluster entire store
/// 6. Save
async fn run_ingest(
    path: &std::path::Path,
    config: &Config,
    store_lock: &Arc<RwLock<Store>>,
    stats_lock: &Arc<RwLock<Option<IngestStats>>>,
) -> eyre::Result<()> {
    tracing::info!(path = %path.display(), "starting ingestion");

    // Step 1: Crawl directory
    let crawl_result = ingestion::crawl(path)?;
    tracing::info!(count = crawl_result.discovered.len(), "files discovered");

    // Step 2: Diff against existing store
    let mut new_docs = Vec::new();
    let mut changed_docs = Vec::new();
    let mut unchanged_count: usize = 0;

    {
        let store = store_lock.read().await;
        for doc in crawl_result.discovered {
            match store.find_by_path(&doc.abs_path) {
                Some(existing) if existing.content_hash == doc.content_hash => {
                    // File unchanged — skip embedding
                    unchanged_count += 1;
                }
                Some(_) => {
                    // File changed — needs re-embedding
                    changed_docs.push(doc);
                }
                None => {
                    // New file
                    new_docs.push(doc);
                }
            }
        }
    }

    let new_count = new_docs.len();
    let changed_count = changed_docs.len();
    let needs_embedding = new_count + changed_count;
    tracing::info!(
        new = new_count,
        changed = changed_count,
        unchanged = unchanged_count,
        "diff complete"
    );

    // Step 3: Embed only new + changed files
    let mut docs_to_embed: Vec<crate::storage::Document> = Vec::with_capacity(needs_embedding);
    docs_to_embed.extend(new_docs);
    docs_to_embed.extend(changed_docs);

    if docs_to_embed.is_empty() {
        tracing::info!("no files changed — skipping embedding API calls");
    } else {
        let texts: Vec<String> = docs_to_embed.iter().map(|d| d.content_preview.clone()).collect();
        let embeddings = embeddings::generate_embeddings(&texts, config).await?;

        for (doc, emb) in docs_to_embed.iter_mut().zip(embeddings) {
            doc.embedding = emb;
        }
        tracing::info!(count = docs_to_embed.len(), "embeddings generated");
    }

    // Step 4: Merge into store
    let mut store = store_lock.write().await;
    store.root_path = path.to_string_lossy().to_string();

    // Remove deleted files
    let deleted_count = store.retain_existing(&crawl_result.discovered_paths);
    if deleted_count > 0 {
        tracing::info!(deleted = deleted_count, "removed deleted files");
    }

    // Upsert new + changed files
    for doc in docs_to_embed {
        store.upsert(doc);
    }

    // Step 5: Re-cluster entire store
    let mut all_docs: Vec<crate::storage::Document> = store.documents.values().cloned().collect();
    cluster_documents(&mut all_docs);

    // Write clustered docs back
    store.documents.clear();
    for doc in all_docs {
        store.documents.insert(doc.abs_path.clone(), doc);
    }
    tracing::info!("clustering complete");

    // Step 6: Save
    store.save(&config.store_path())?;

    let stats = IngestStats {
        new_files: new_count,
        changed_files: changed_count,
        unchanged_files: unchanged_count,
        deleted_files: deleted_count,
        total_documents: store.documents.len(),
        embeddings_generated: needs_embedding,
    };

    drop(store); // Release the write lock before acquiring stats lock

    {
        let mut last_stats = stats_lock.write().await;
        *last_stats = Some(stats.clone());
    }

    tracing::info!(
        new = stats.new_files,
        changed = stats.changed_files,
        unchanged = stats.unchanged_files,
        deleted = stats.deleted_files,
        total = stats.total_documents,
        "ingestion complete"
    );

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
        .documents_sorted()
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

    let docs: Vec<&crate::storage::Document> = store.documents.values().collect();
    let results = search::search(&query_embedding, &docs, query.limit);

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
        .values()
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
