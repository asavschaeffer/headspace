use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;

use axum::Router;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::{delete, get, post};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, broadcast};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;

use crate::cluster::cluster_documents;
use crate::config::Config;
use crate::cortex::{self, IngestInput};
use crate::embeddings;
use crate::ingestion;
use crate::search;
use crate::storage::{Document, OverseenDir, Store};

/// Shared application state.
#[derive(Debug, Clone)]
pub struct AppState {
    pub store: Arc<RwLock<Store>>,
    pub config: Arc<Config>,
    pub ingesting: Arc<RwLock<bool>>,
    /// Stats from the most recent ingestion (if any).
    pub last_ingest_stats: Arc<RwLock<Option<IngestStats>>>,
    /// Broadcast stream for ingestion progress updates.
    pub progress_tx: Arc<broadcast::Sender<ProgressEvent>>,
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

/// Progress event emitted while ingestion is running.
#[derive(Debug, Clone, Serialize)]
pub struct ProgressEvent {
    pub phase: String,
    pub message: String,
    pub done: usize,
    pub total: usize,
    pub current_file: Option<String>,
}

/// Creates the API router.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/api/status", get(status))
        .route("/api/ingest", post(ingest))
        .route("/api/ingest/stream", get(ingest_stream))
        .route("/api/documents", get(list_documents))
        .route("/api/document/{id}", get(get_document))
        .route("/api/search", get(search_documents))
        .route("/api/clusters", get(get_clusters))
        .route("/api/review/queue", get(review_queue))
        .route("/api/review/{id}/approve", post(approve_review))
        .route("/api/review/{id}/reject", post(reject_review))
        .route("/api/review/bulk", post(bulk_review))
        .route("/api/review/finish", post(finish_review))
        .route("/api/overseen", get(list_overseen).post(add_overseen))
        .route("/api/overseen/{id}", delete(remove_overseen))
        .with_state(state)
}

#[derive(Serialize)]
struct StatusResponse {
    document_count: usize,
    approved_count: usize,
    pending_review_count: usize,
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
    let approved_count = store.approved_documents().map(|d| d.len()).unwrap_or(0);
    let pending_review_count = store
        .pending_review_documents()
        .map(|d| d.len())
        .unwrap_or(0);

    Json(StatusResponse {
        document_count,
        approved_count,
        pending_review_count,
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
    start_ingest_job(state.clone(), PathBuf::from(&body.path), None).await?;
    Ok(Json(IngestResponse {
        message: "ingestion started".to_string(),
        document_count: 0,
    }))
}

async fn ingest_stream(
    State(state): State<AppState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = state.progress_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|result| {
        let serialized = result
            .ok()
            .and_then(|event| serde_json::to_string(&event).ok());
        serialized.map(|data| Ok(Event::default().data(data)))
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

#[derive(Debug)]
struct EnrichWork {
    discovered: ingestion::DiscoveredFile,
    doc: Document,
}

async fn start_ingest_job(
    state: AppState,
    path: PathBuf,
    overseen_id: Option<String>,
) -> Result<(), (StatusCode, String)> {
    if !path.is_dir() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("not a directory: {}", path.display()),
        ));
    }

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

    emit_progress(
        &state.progress_tx,
        ProgressEvent {
            phase: "discover".to_string(),
            message: format!("Starting discovery in {}", path.display()),
            done: 0,
            total: 0,
            current_file: None,
        },
    );

    let config = state.config.clone();
    let store_lock = state.store.clone();
    let ingesting_flag = state.ingesting.clone();
    let stats_lock = state.last_ingest_stats.clone();
    let progress_tx = state.progress_tx.clone();

    tokio::spawn(async move {
        let result = run_ingest(&path, &config, &store_lock, &stats_lock, &progress_tx).await;

        if result.is_ok() {
            if let Some(id) = overseen_id {
                let mut store = store_lock.write().await;
                if let Err(err) = store.touch_overseen_dir(&id) {
                    tracing::warn!(id = %id, error = %err, "failed to touch overseen dir");
                }
            }
        } else if let Err(err) = &result {
            emit_progress(
                &progress_tx,
                ProgressEvent {
                    phase: "error".to_string(),
                    message: format!("Ingestion failed: {err}"),
                    done: 0,
                    total: 0,
                    current_file: None,
                },
            );
        }

        let mut ingesting = ingesting_flag.write().await;
        *ingesting = false;

        if let Err(err) = result {
            tracing::error!("ingestion failed: {err:?}");
        }
    });

    Ok(())
}

/// Runs the diff-based ingestion pipeline and leaves files pending review.
async fn run_ingest(
    path: &FsPath,
    config: &Config,
    store_lock: &Arc<RwLock<Store>>,
    stats_lock: &Arc<RwLock<Option<IngestStats>>>,
    progress_tx: &broadcast::Sender<ProgressEvent>,
) -> eyre::Result<()> {
    tracing::info!(path = %path.display(), "starting ingestion");

    let discovery = ingestion::discover(path, config.cortex_max_bytes)?;
    let discovered_count = discovery.discovered.len();
    emit_progress(
        progress_tx,
        ProgressEvent {
            phase: "discover".to_string(),
            message: format!("Discovered {discovered_count} files"),
            done: discovered_count,
            total: discovered_count,
            current_file: None,
        },
    );

    let discovered_file_ids = discovery.discovered_file_ids;
    let mut enrich_queue: Vec<EnrichWork> = Vec::new();
    let mut unchanged_updates: Vec<Document> = Vec::new();
    let mut unchanged_count = 0usize;
    let mut new_count = 0usize;
    let mut changed_count = 0usize;

    {
        let store = store_lock.read().await;
        for discovered in discovery.discovered {
            let existing = match store.find_by_file_id(&discovered.file_id)? {
                Some(existing) => Some(existing),
                None => store.find_by_path(&discovered.abs_path)?,
            };

            match existing {
                Some(existing) if existing.content_hash == discovered.content_hash => {
                    unchanged_count += 1;
                    let mut updated = existing.clone();
                    let new_len = usize::try_from(discovered.size_bytes).unwrap_or(0);
                    let path_changed = updated.abs_path != discovered.abs_path
                        || updated.rel_path != discovered.rel_path
                        || updated.name != discovered.name
                        || updated.extension != discovered.extension
                        || updated.modified_at != discovered.modified_at
                        || updated.content_length != new_len;

                    if path_changed {
                        updated.abs_path = discovered.abs_path.clone();
                        updated.rel_path = discovered.rel_path.clone();
                        updated.name = discovered.name.clone();
                        updated.extension = discovered.extension.clone();
                        updated.modified_at = discovered.modified_at;
                        updated.content_length = new_len;
                        unchanged_updates.push(updated);
                    }
                }
                Some(existing) => {
                    changed_count += 1;
                    let mut doc = Document::new(
                        discovered.file_id.clone(),
                        discovered.content_hash.clone(),
                        discovered.rel_path.clone(),
                        discovered.abs_path.clone(),
                        discovered.name.clone(),
                        discovered.extension.clone(),
                        String::new(),
                        usize::try_from(discovered.size_bytes).unwrap_or(0),
                        discovered.modified_at,
                    );
                    doc.id = existing.id;
                    doc.review_status = "pending_review".to_string();
                    enrich_queue.push(EnrichWork { discovered, doc });
                }
                None => {
                    new_count += 1;
                    let mut doc = Document::new(
                        discovered.file_id.clone(),
                        discovered.content_hash.clone(),
                        discovered.rel_path.clone(),
                        discovered.abs_path.clone(),
                        discovered.name.clone(),
                        discovered.extension.clone(),
                        String::new(),
                        usize::try_from(discovered.size_bytes).unwrap_or(0),
                        discovered.modified_at,
                    );
                    doc.review_status = "pending_review".to_string();
                    enrich_queue.push(EnrichWork { discovered, doc });
                }
            }
        }
    }

    emit_progress(
        progress_tx,
        ProgressEvent {
            phase: "diff".to_string(),
            message: format!(
                "{new_count} new | {changed_count} changed | {unchanged_count} unchanged"
            ),
            done: new_count + changed_count,
            total: discovered_count,
            current_file: None,
        },
    );

    let mut enriched_docs = Vec::with_capacity(enrich_queue.len());
    let enrich_total = enrich_queue.len();
    for (idx, mut work) in enrich_queue.into_iter().enumerate() {
        tracing::debug!(
            file = %work.discovered.rel_path,
            "reading bytes for enrichment"
        );
        let bytes = match std::fs::read(&work.discovered.abs_path) {
            Ok(bytes) => bytes,
            Err(err) => {
                tracing::warn!(
                    file = %work.discovered.abs_path,
                    error = %err,
                    "failed to read file during enrichment; skipping"
                );
                emit_progress(
                    progress_tx,
                    ProgressEvent {
                        phase: "enrich".to_string(),
                        message: format!(
                            "Skipped unreadable file {} ({}/{})",
                            work.discovered.rel_path,
                            idx + 1,
                            enrich_total
                        ),
                        done: idx + 1,
                        total: enrich_total,
                        current_file: Some(work.discovered.rel_path.clone()),
                    },
                );
                continue;
            }
        };

        let crawl_entry = ingestion::CrawlEntry {
            discovered: work.discovered,
            bytes,
        };

        let input = IngestInput::new(
            crawl_entry.discovered.abs_path.clone(),
            crawl_entry.discovered.rel_path.clone(),
            crawl_entry.discovered.file_id.clone(),
            crawl_entry.bytes,
            crawl_entry.discovered.extension.clone(),
            crawl_entry.discovered.modified_at,
            crawl_entry.discovered.content_hash.clone(),
        );

        emit_progress(
            progress_tx,
            ProgressEvent {
                phase: "enrich".to_string(),
                message: format!(
                    "Enriching {} ({}/{})",
                    input.rel_path,
                    idx + 1,
                    enrich_total
                ),
                done: idx + 1,
                total: enrich_total,
                current_file: Some(input.rel_path.clone()),
            },
        );

        let extraction = cortex::enrich_document(&mut work.doc, &input, config).await?;
        work.doc.content_canonical = extraction.canonical_text;
        work.doc.embedding.clear();
        work.doc.cluster_id = -1;
        work.doc.x = 0.0;
        work.doc.y = 0.0;
        work.doc.review_status = "pending_review".to_string();
        enriched_docs.push(work.doc);
    }

    let stats;
    let pending_count;
    {
        let mut store = store_lock.write().await;
        store.set_root_path(path)?;
        let deleted_count = store.retain_existing(&discovered_file_ids)?;

        for doc in unchanged_updates {
            store.upsert(doc)?;
        }
        for doc in enriched_docs {
            store.upsert(doc)?;
        }
        store.save()?;

        pending_count = store.pending_review_documents()?.len();
        let total_documents = store.document_count()?;
        stats = IngestStats {
            new_files: new_count,
            changed_files: changed_count,
            unchanged_files: unchanged_count,
            deleted_files: deleted_count,
            total_documents,
            embeddings_generated: 0,
        };
    }

    {
        let mut last_stats = stats_lock.write().await;
        *last_stats = Some(stats.clone());
    }

    emit_progress(
        progress_tx,
        ProgressEvent {
            phase: "complete".to_string(),
            message: format!("Complete - {pending_count} files pending review"),
            done: pending_count,
            total: pending_count,
            current_file: None,
        },
    );

    tracing::info!(
        new = stats.new_files,
        changed = stats.changed_files,
        unchanged = stats.unchanged_files,
        deleted = stats.deleted_files,
        total = stats.total_documents,
        pending = pending_count,
        "ingestion complete"
    );

    Ok(())
}

fn emit_progress(tx: &broadcast::Sender<ProgressEvent>, event: ProgressEvent) {
    let _ = tx.send(event);
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
    review_status: String,
}

async fn list_documents(State(state): State<AppState>) -> Json<Vec<DocumentSummary>> {
    let store = state.store.read().await;
    let docs = store.documents_sorted_approved().unwrap_or_else(|e| {
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
            review_status: d.review_status.clone(),
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
        .map_err(internal_error)?
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

    let docs = store.approved_documents().unwrap_or_else(|e| {
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
    let docs = store.approved_documents().unwrap_or_else(|e| {
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

#[derive(Serialize)]
struct ReviewQueueItem {
    id: String,
    name: String,
    rel_path: String,
    abs_path: String,
    extension: String,
    content_length: usize,
    cluster_id: i32,
    status: String,
    summary: String,
    topics: Vec<String>,
    review_status: String,
    content_hash: String,
    dupe_count: usize,
    dir_path: String,
    modified_at: u64,
}

async fn review_queue(State(state): State<AppState>) -> Json<Vec<ReviewQueueItem>> {
    let store = state.store.read().await;
    let docs = store.pending_review_documents().unwrap_or_else(|e| {
        tracing::error!("failed to load review queue: {e:?}");
        Vec::new()
    });

    let mut by_hash: HashMap<String, usize> = HashMap::new();
    for doc in &docs {
        *by_hash.entry(doc.content_hash.clone()).or_default() += 1;
    }

    let items = docs
        .into_iter()
        .map(|doc| {
            let dupe_total = by_hash.get(&doc.content_hash).copied().unwrap_or(1);
            let dir_path = FsPath::new(&doc.abs_path)
                .parent()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();

            ReviewQueueItem {
                id: doc.id,
                name: doc.name,
                rel_path: doc.rel_path,
                abs_path: doc.abs_path,
                extension: doc.extension,
                content_length: doc.content_length,
                cluster_id: doc.cluster_id,
                status: doc.status,
                summary: doc.summary,
                topics: doc.topics,
                review_status: doc.review_status,
                content_hash: doc.content_hash,
                dupe_count: dupe_total.saturating_sub(1),
                dir_path,
                modified_at: doc.modified_at,
            }
        })
        .collect();

    Json(items)
}

#[derive(Serialize)]
struct ReviewActionResponse {
    message: String,
}

async fn approve_review(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<ReviewActionResponse>, (StatusCode, String)> {
    let mut doc = {
        let store = state.store.read().await;
        store
            .find_by_id(&id)
            .map_err(internal_error)?
            .ok_or_else(|| (StatusCode::NOT_FOUND, "document not found".to_string()))?
    };

    let texts = vec![embedding_text_for_doc(&doc)];
    let mut generated = embeddings::generate_embeddings(&texts, &state.config)
        .await
        .map_err(internal_error)?;
    doc.embedding = generated.pop().unwrap_or_default();
    doc.review_status = "approved".to_string();

    let mut store = state.store.write().await;
    store.upsert(doc).map_err(internal_error)?;
    store.save().map_err(internal_error)?;

    Ok(Json(ReviewActionResponse {
        message: "approved".to_string(),
    }))
}

async fn reject_review(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<ReviewActionResponse>, (StatusCode, String)> {
    let file_id = {
        let store = state.store.read().await;
        store
            .find_by_id(&id)
            .map_err(internal_error)?
            .ok_or_else(|| (StatusCode::NOT_FOUND, "document not found".to_string()))?
            .file_id
    };

    let mut store = state.store.write().await;
    store
        .set_review_status(&file_id, "rejected")
        .map_err(internal_error)?;
    store.save().map_err(internal_error)?;

    Ok(Json(ReviewActionResponse {
        message: "rejected".to_string(),
    }))
}

#[derive(Deserialize)]
struct BulkReviewRequest {
    ids: Vec<String>,
    action: String,
}

#[derive(Serialize)]
struct BulkReviewResponse {
    approved: usize,
    rejected: usize,
    skipped: usize,
}

async fn bulk_review(
    State(state): State<AppState>,
    Json(body): Json<BulkReviewRequest>,
) -> Result<Json<BulkReviewResponse>, (StatusCode, String)> {
    if body.ids.is_empty() {
        return Ok(Json(BulkReviewResponse {
            approved: 0,
            rejected: 0,
            skipped: 0,
        }));
    }

    match body.action.as_str() {
        "approve" => bulk_approve(state, body.ids).await,
        "reject" => bulk_reject(state, body.ids).await,
        _ => Err((
            StatusCode::BAD_REQUEST,
            "action must be 'approve' or 'reject'".to_string(),
        )),
    }
}

async fn bulk_approve(
    state: AppState,
    ids: Vec<String>,
) -> Result<Json<BulkReviewResponse>, (StatusCode, String)> {
    let id_set: HashSet<String> = ids.into_iter().collect();

    let docs = {
        let store = state.store.read().await;
        let mut docs = Vec::new();
        for id in id_set {
            if let Some(doc) = store.find_by_id(&id).map_err(internal_error)? {
                docs.push(doc);
            }
        }
        docs
    };

    let mut approved_docs = Vec::new();
    let mut texts = Vec::new();
    let skipped = 0usize;

    for mut doc in docs {
        doc.review_status = "approved".to_string();
        texts.push(embedding_text_for_doc(&doc));
        approved_docs.push(doc);
    }

    if !approved_docs.is_empty() {
        let embeddings = embeddings::generate_embeddings(&texts, &state.config)
            .await
            .map_err(internal_error)?;
        for (doc, embedding) in approved_docs.iter_mut().zip(embeddings) {
            doc.embedding = embedding;
        }

        let mut store = state.store.write().await;
        for doc in &approved_docs {
            store.upsert(doc.clone()).map_err(internal_error)?;
        }
        store.save().map_err(internal_error)?;
    }

    Ok(Json(BulkReviewResponse {
        approved: approved_docs.len(),
        rejected: 0,
        skipped,
    }))
}

async fn bulk_reject(
    state: AppState,
    ids: Vec<String>,
) -> Result<Json<BulkReviewResponse>, (StatusCode, String)> {
    let id_set: HashSet<String> = ids.into_iter().collect();
    let file_ids = {
        let store = state.store.read().await;
        let mut file_ids = Vec::new();
        for id in id_set {
            if let Some(doc) = store.find_by_id(&id).map_err(internal_error)? {
                file_ids.push(doc.file_id);
            }
        }
        file_ids
    };

    let mut store = state.store.write().await;
    for file_id in &file_ids {
        store
            .set_review_status(file_id, "rejected")
            .map_err(internal_error)?;
    }
    store.save().map_err(internal_error)?;

    Ok(Json(BulkReviewResponse {
        approved: 0,
        rejected: file_ids.len(),
        skipped: 0,
    }))
}

#[derive(Serialize)]
struct FinishReviewResponse {
    approved_documents: usize,
    cluster_count: usize,
    noise_documents: usize,
}

async fn finish_review(
    State(state): State<AppState>,
) -> Result<Json<FinishReviewResponse>, (StatusCode, String)> {
    let mut approved_docs = {
        let store = state.store.read().await;
        store.approved_documents().map_err(internal_error)?
    };

    if approved_docs.is_empty() {
        return Ok(Json(FinishReviewResponse {
            approved_documents: 0,
            cluster_count: 0,
            noise_documents: 0,
        }));
    }

    cluster_documents(&mut approved_docs);

    let cluster_count = approved_docs
        .iter()
        .filter(|d| d.cluster_id >= 0)
        .map(|d| d.cluster_id)
        .collect::<HashSet<_>>()
        .len();
    let noise_documents = approved_docs.iter().filter(|d| d.cluster_id < 0).count();

    let mut store = state.store.write().await;
    store
        .replace_documents(&approved_docs)
        .map_err(internal_error)?;
    store.save().map_err(internal_error)?;

    Ok(Json(FinishReviewResponse {
        approved_documents: approved_docs.len(),
        cluster_count,
        noise_documents,
    }))
}

#[derive(Deserialize)]
struct OverseenRequest {
    path: String,
}

#[derive(Serialize)]
struct OverseenAddResponse {
    overseen: OverseenDir,
    message: String,
}

async fn list_overseen(State(state): State<AppState>) -> Json<Vec<OverseenDir>> {
    let store = state.store.read().await;
    let dirs = store.list_overseen_dirs().unwrap_or_else(|e| {
        tracing::error!("failed to list overseen dirs: {e:?}");
        Vec::new()
    });
    Json(dirs)
}

async fn add_overseen(
    State(state): State<AppState>,
    Json(body): Json<OverseenRequest>,
) -> Result<Json<OverseenAddResponse>, (StatusCode, String)> {
    let canonical = std::fs::canonicalize(&body.path).map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            format!("not a directory: {}", body.path),
        )
    })?;
    if !canonical.is_dir() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("not a directory: {}", body.path),
        ));
    }
    let canonical_string = canonical.to_string_lossy().to_string();

    let overseen = {
        let mut store = state.store.write().await;
        store
            .add_overseen_dir(&canonical_string)
            .map_err(internal_error)?
    };

    start_ingest_job(state.clone(), canonical, Some(overseen.id.clone())).await?;

    Ok(Json(OverseenAddResponse {
        overseen,
        message: "overseen directory added and ingestion started".to_string(),
    }))
}

async fn remove_overseen(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let mut store = state.store.write().await;
    store.remove_overseen_dir(&id).map_err(internal_error)?;
    Ok(StatusCode::NO_CONTENT)
}

fn embedding_text_for_doc(doc: &Document) -> String {
    let canonical = doc.content_canonical.trim();
    if canonical.is_empty() {
        doc.content_preview.clone()
    } else {
        canonical.to_string()
    }
}

fn internal_error<E: std::fmt::Display>(err: E) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, err.to_string())
}
