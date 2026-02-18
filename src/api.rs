use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::Router;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use axum::routing::{get, post};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::cluster::cluster_documents;
use crate::config::Config;
use crate::cortex::{self, IngestInput};
use crate::embeddings;
use crate::ingestion;
use crate::search;
use crate::storage::{Document, Store};

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
    let document_count = store.document_count().unwrap_or_else(|e| {
        tracing::error!("failed to count documents: {e:?}");
        0
    });

    Json(StatusResponse {
        document_count,
        has_embeddings: state.config.has_embeddings(),
        is_ingesting: ingesting,
        root_path: store.root_path().to_string(),
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
    {
        let ingesting = state.ingesting.read().await;
        if *ingesting {
            return Err((
                StatusCode::CONFLICT,
                "ingestion already in progress".to_string(),
            ));
        }
    }

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

#[derive(Debug)]
struct IngestWork {
    doc: Document,
    input: IngestInput,
}

/// Runs the diff-based ingestion pipeline.
async fn run_ingest(
    path: &std::path::Path,
    config: &Config,
    store_lock: &Arc<RwLock<Store>>,
    stats_lock: &Arc<RwLock<Option<IngestStats>>>,
) -> eyre::Result<()> {
    tracing::info!(path = %path.display(), "starting ingestion");

    let crawl_result = ingestion::crawl(path, config.cortex_max_bytes)?;
    tracing::info!(count = crawl_result.discovered.len(), "files discovered");

    let mut embed_jobs: Vec<IngestWork> = Vec::new();
    let mut metadata_jobs: Vec<IngestWork> = Vec::new();
    let mut unchanged_count: usize = 0;
    let mut new_count: usize = 0;
    let mut changed_count: usize = 0;

    {
        let store = store_lock.read().await;
        for entry in crawl_result.discovered {
            let existing = match store.find_by_file_id(&entry.file_id)? {
                Some(existing) => Some(existing),
                None => store.find_by_path(&entry.abs_path)?,
            };

            let input = IngestInput::new(
                entry.abs_path.clone(),
                entry.rel_path.clone(),
                entry.file_id.clone(),
                entry.bytes,
                entry.extension.clone(),
                entry.modified_at,
                entry.content_hash.clone(),
            );

            let mut doc = Document::new(
                entry.file_id.clone(),
                entry.content_hash.clone(),
                entry.rel_path.clone(),
                entry.abs_path.clone(),
                entry.name.clone(),
                entry.extension.clone(),
                String::new(),
                input.bytes.len(),
                entry.modified_at,
            );

            match existing {
                Some(existing) if existing.content_hash == entry.content_hash => {
                    unchanged_count += 1;
                    let path_changed = existing.abs_path != entry.abs_path
                        || existing.rel_path != entry.rel_path
                        || existing.modified_at != entry.modified_at
                        || existing.name != entry.name
                        || existing.extension != entry.extension;
                    let metadata_missing = existing.summary.trim().is_empty()
                        || existing.status.trim().is_empty()
                        || existing.mime_type.trim().is_empty()
                        || existing.pipeline_kind.trim().is_empty()
                        || existing.pipeline_kind.eq_ignore_ascii_case("unknown")
                        || existing.metadata_version < 1;

                    if path_changed || metadata_missing {
                        let mut updated = existing;
                        updated.abs_path = entry.abs_path;
                        updated.rel_path = entry.rel_path;
                        updated.name = entry.name;
                        updated.extension = entry.extension;
                        updated.content_hash = entry.content_hash;
                        updated.modified_at = entry.modified_at;
                        metadata_jobs.push(IngestWork {
                            doc: updated,
                            input,
                        });
                    }
                }
                Some(existing) => {
                    changed_count += 1;
                    doc.id = existing.id;
                    doc.cluster_id = existing.cluster_id;
                    doc.x = existing.x;
                    doc.y = existing.y;
                    doc.embedding = existing.embedding;
                    embed_jobs.push(IngestWork { doc, input });
                }
                None => {
                    new_count += 1;
                    embed_jobs.push(IngestWork { doc, input });
                }
            }
        }
    }

    let needs_embedding = embed_jobs.len();
    tracing::info!(
        new = new_count,
        changed = changed_count,
        unchanged = unchanged_count,
        metadata_refresh = metadata_jobs.len(),
        "diff complete"
    );

    let mut pipeline_counts: HashMap<String, usize> = HashMap::new();
    let mut metadata_source_counts: HashMap<String, usize> = HashMap::new();
    let mut total_enrich_ms: u128 = 0;
    let mut enrich_count: usize = 0;

    let mut unchanged_updates = Vec::with_capacity(metadata_jobs.len());
    for mut work in metadata_jobs {
        let started = Instant::now();
        let _extraction = cortex::enrich_document(&mut work.doc, &work.input, config).await?;
        let elapsed = started.elapsed().as_millis();
        total_enrich_ms += elapsed;
        enrich_count += 1;
        increment(&mut pipeline_counts, &work.doc.pipeline_kind);
        increment(&mut metadata_source_counts, &work.doc.metadata_source);
        tracing::debug!(
            file = %work.doc.rel_path,
            pipeline = %work.doc.pipeline_kind,
            metadata_source = %work.doc.metadata_source,
            elapsed_ms = elapsed,
            "metadata refresh complete"
        );
        unchanged_updates.push(work.doc);
    }

    let mut docs_to_embed: Vec<Document> = Vec::with_capacity(needs_embedding);
    let mut embed_texts: Vec<String> = Vec::with_capacity(needs_embedding);
    for mut work in embed_jobs {
        let started = Instant::now();
        let extraction = cortex::enrich_document(&mut work.doc, &work.input, config).await?;
        let elapsed = started.elapsed().as_millis();
        total_enrich_ms += elapsed;
        enrich_count += 1;
        increment(&mut pipeline_counts, &work.doc.pipeline_kind);
        increment(&mut metadata_source_counts, &work.doc.metadata_source);
        tracing::debug!(
            file = %work.doc.rel_path,
            pipeline = %work.doc.pipeline_kind,
            metadata_source = %work.doc.metadata_source,
            elapsed_ms = elapsed,
            "document enrichment complete"
        );
        embed_texts.push(extraction.canonical_text);
        docs_to_embed.push(work.doc);
    }

    if docs_to_embed.is_empty() {
        tracing::info!("no files changed, skipping embedding API calls");
    } else {
        let embeddings = embeddings::generate_embeddings(&embed_texts, config).await?;
        for (doc, emb) in docs_to_embed.iter_mut().zip(embeddings) {
            doc.embedding = emb;
        }
        tracing::info!(count = docs_to_embed.len(), "embeddings generated");
    }

    let mut store = store_lock.write().await;
    store.set_root_path(path)?;

    let deleted_count = store.retain_existing(&crawl_result.discovered_file_ids)?;
    if deleted_count > 0 {
        tracing::info!(deleted = deleted_count, "removed deleted files");
    }

    for doc in unchanged_updates {
        store.upsert(doc)?;
    }
    for doc in docs_to_embed {
        store.upsert(doc)?;
    }

    let mut all_docs = store.all_documents()?;
    cluster_documents(&mut all_docs);
    let total_documents = all_docs.len();
    store.replace_documents(&all_docs)?;
    store.save()?;

    let avg_enrich_ms = if enrich_count == 0 {
        0
    } else {
        (total_enrich_ms / u128::from(enrich_count as u64)) as u64
    };
    tracing::info!(
        ?pipeline_counts,
        ?metadata_source_counts,
        average_enrich_ms = avg_enrich_ms,
        "cortex metrics"
    );

    let stats = IngestStats {
        new_files: new_count,
        changed_files: changed_count,
        unchanged_files: unchanged_count,
        deleted_files: deleted_count,
        total_documents,
        embeddings_generated: needs_embedding,
    };

    drop(store);

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

fn increment(map: &mut HashMap<String, usize>, key: &str) {
    *map.entry(key.to_string()).or_default() += 1;
}

#[derive(Serialize)]
struct DocumentSummary {
    id: String,
    name: String,
    rel_path: String,
    extension: String,
    content_length: usize,
    cluster_id: i32,
    status: String,
    summary: String,
    topics: Vec<String>,
}

async fn list_documents(State(state): State<AppState>) -> Json<Vec<DocumentSummary>> {
    let store = state.store.read().await;
    let docs = store.documents_sorted().unwrap_or_else(|e| {
        tracing::error!("failed to list documents: {e:?}");
        Vec::new()
    });

    let summaries: Vec<DocumentSummary> = docs
        .iter()
        .map(|d| DocumentSummary {
            id: d.id.clone(),
            name: d.name.clone(),
            rel_path: d.rel_path.clone(),
            extension: d.extension.clone(),
            content_length: d.content_length,
            cluster_id: d.cluster_id,
            status: d.status.clone(),
            summary: d.summary.clone(),
            topics: d.topics.clone(),
        })
        .collect();
    Json(summaries)
}

async fn get_document(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = state.store.read().await;
    let doc = store
        .find_by_id(&id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or_else(|| (StatusCode::NOT_FOUND, "document not found".to_string()))?;

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
    summary: String,
    score: f64,
    cluster_id: i32,
}

async fn search_documents(
    State(state): State<AppState>,
    Query(query): Query<SearchQuery>,
) -> Json<Vec<SearchResultResponse>> {
    let config = state.config.clone();
    let store = state.store.read().await;

    let query_embedding = embeddings::generate_query_embedding(&query.q, &config)
        .await
        .unwrap_or_default();

    let docs = store.all_documents().unwrap_or_else(|e| {
        tracing::error!("failed to load documents for search: {e:?}");
        Vec::new()
    });
    let doc_refs: Vec<&Document> = docs.iter().collect();
    let results = search::search(&query_embedding, &doc_refs, query.limit);

    let response: Vec<SearchResultResponse> = results
        .into_iter()
        .map(|r| SearchResultResponse {
            id: r.document.id,
            name: r.document.name,
            rel_path: r.document.rel_path,
            extension: r.document.extension,
            content_preview: r.document.content_preview.chars().take(500).collect(),
            summary: r.document.summary,
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
    let docs = store.all_documents().unwrap_or_else(|e| {
        tracing::error!("failed to load clusters: {e:?}");
        Vec::new()
    });

    let points: Vec<ClusterPoint> = docs
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
