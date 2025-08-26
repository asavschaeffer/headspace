# Globule API Reference

This document provides reference documentation for the `GlobuleAPI`, the primary interface for interacting with the Globule application.

## Overview

The `GlobuleAPI` class, defined in `src/globule/core/api.py`, provides a stable, high-level interface for all core application features. All frontends (CLI, TUI, Glass Engine) use this API to ensure consistent behavior.

## Core Methods

### `async def add_thought(self, text: str, source: str = "api") -> ProcessedGlobuleV1`

Adds and processes a new thought through the complete pipeline.

-   **Parameters:**
    -   `text` (str): The raw text of the thought.
    -   `source` (str): Optional. The source of the input (e.g., 'tui', 'cli', 'glass_engine'). Defaults to 'api'.
-   **Returns:**
    -   A `ProcessedGlobuleV1` object containing the enriched and saved thought with embeddings, parsed data, and file decisions.

### `async def search_semantic(self, query: str, limit: int = 10) -> List[ProcessedGlobuleV1]`

Performs a semantic vector search for globules using natural language.

-   **Parameters:**
    -   `query` (str): The natural language search query.
    -   `limit` (int): Optional. The maximum number of results to return. Defaults to 10.
-   **Returns:**
    -   A list of `ProcessedGlobuleV1` objects that are semantically similar to the query, ordered by relevance.

### `async def search_structured(self, query: StructuredQuery) -> List[ProcessedGlobuleV1]`

Performs a structured search based on metadata filters.

-   **Parameters:**
    -   `query` (StructuredQuery): A `StructuredQuery` object with filters for domain, category, etc.
-   **Returns:**
    -   A list of `ProcessedGlobuleV1` objects matching the structured query criteria.

### `async def get_globule_by_id(self, globule_id: UUID) -> Optional[ProcessedGlobuleV1]`

Retrieves a single globule by its unique ID.

-   **Parameters:**
    -   `globule_id` (UUID): The UUID of the globule to retrieve.
-   **Returns:**
    -   The `ProcessedGlobuleV1` object, or `None` if not found.

### `async def get_all_globules(self, limit: int = 100) -> List[ProcessedGlobuleV1]`

Retrieves all globules from storage, up to a specified limit.

-   **Parameters:**
    -   `limit` (int): Optional. The maximum number of globules to retrieve. Defaults to 100.
-   **Returns:**
    -   A list of all `ProcessedGlobuleV1` objects, ordered by creation date.

### `async def add_from_input_message(self, input_message: InputMessage) -> List[ProcessedGlobuleV1]`

Process an InputMessage from external sources (WhatsApp, Telegram, email) and store all resulting globules.

This method handles bundled content - text and attachments from a single message are processed together and linked with shared metadata.

-   **Parameters:**
    -   `input_message` (InputMessage): InputMessage object from the inputs module containing message content, attachments, and platform metadata.
-   **Returns:**
    -   A list of `ProcessedGlobuleV1` objects that were created and stored, including separate globules for text content and each attachment.

## Advanced Analysis Methods

### `async def get_clusters(self) -> ClusteringAnalysis`

Analyzes all globules and groups them into semantic clusters using machine learning.

-   **Returns:**
    -   A `ClusteringAnalysis` object containing discovered clusters with labels, descriptions, keywords, and confidence scores.

### `async def natural_language_query(self, question: str) -> List[Dict[str, Any]]`

Converts a natural language question to SQL, executes it, and returns structured results.

-   **Parameters:**
    -   `question` (str): The user's natural language question about their data (e.g., "How many globules contain the word 'python'?").
-   **Returns:**
    -   The SQL query result as a list of dictionaries with column names as keys.

### `async def get_summary_for_text(self, text: str) -> str`

Generates an intelligent summary for a given piece of text.

-   **Parameters:**
    -   `text` (str): The text to summarize.
-   **Returns:**
    -   A concise string summary, falling back to truncation if parsing fails.

## Utility Methods

### `async def reconcile_files(self) -> Dict[str, Any]`

Reconciles the file system with the database, ensuring consistency.

-   **Returns:**
    -   Statistics dictionary with counts of files processed, added, updated, and any errors encountered.

### `async def export_draft(self, draft_content: str, file_path: str) -> bool`

Exports draft content to a specified file path.

-   **Parameters:**
    -   `draft_content` (str): The content of the draft to export.
    -   `file_path` (str): The absolute or relative path where to save the file.
-   **Returns:**
    -   `True` if the export was successful, `False` if an I/O error occurred.

## Skeleton Management Methods

The skeleton system provides templated canvas layouts for different use cases.

### `def list_skeletons(self) -> List[Dict[str, Any]]`

Lists all available canvas skeleton templates.

-   **Returns:**
    -   A list of dictionaries containing skeleton metadata (name, description, modules).

### `def apply_skeleton(self, name: str, query_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]`

Applies a skeleton template to create a new canvas layout.

-   **Parameters:**
    -   `name` (str): The name of the skeleton template to apply.
    -   `query_data` (Dict, optional): Additional context data for template customization.
-   **Returns:**
    -   A list of canvas module dictionaries representing the generated layout.

### `def get_skeleton_stats(self) -> Dict[str, Any]`

Gets statistics about skeleton templates and their usage.

-   **Returns:**
    -   A dictionary containing template counts, usage statistics, and available modules.

### `def create_default_skeletons(self) -> List[str]`

Creates default skeleton templates for common use cases.

-   **Returns:**
    -   A list of strings representing the names of the created default templates.

## Usage Examples

```python
# Initialize the API
api = GlobuleAPI(storage_manager, orchestrator)

# Add a thought
result = await api.add_thought("Machine learning for semantic search", source="cli")
print(f"Thought saved with ID: {result.globule_id}")

# Search for related thoughts
results = await api.search_semantic("artificial intelligence", limit=5)
for globule in results:
    print(f"- {globule.original_globule.raw_text[:50]}...")

# Discover patterns
clusters = await api.get_clusters()
for cluster in clusters.clusters:
    print(f"Cluster: {cluster.label} ({cluster.size} thoughts)")

# Ask natural language questions
sql_results = await api.natural_language_query("How many thoughts mention Python?")
print(f"Found {len(sql_results)} matching thoughts")

# Process messages from external platforms
from globule.inputs.models import InputMessage, Attachment, AttachmentType
from datetime import datetime

# Create an InputMessage (typically from webhook)
whatsapp_message = InputMessage(
    content="Progressive overload principles could apply to creative stamina",
    source="whatsapp",
    user_identifier="+1234567890",
    timestamp=datetime.now(),
    message_id="whatsapp_msg_123",
    platform_metadata={
        "whatsapp_message_type": "text",
        "phone_number_id": "123456789"
    }
)

# Process the message (handles both text and any attachments)
processed_globules = await api.add_from_input_message(whatsapp_message)
print(f"Created {len(processed_globules)} globules from WhatsApp message")

# Messages from all sources appear in searches
all_results = await api.search_semantic("creative stamina", limit=10)
whatsapp_results = [g for g in all_results if g.provider_metadata.get('input_source') == 'whatsapp']
print(f"Found {len(whatsapp_results)} WhatsApp thoughts about creative stamina")
```

## Error Handling

All API methods handle errors gracefully:
- Methods return `None` or empty lists when no data is found
- Exceptions are caught and logged, with fallback behaviors where appropriate
- The `export_draft` method returns `False` on I/O errors instead of raising exceptions
- Network errors with external services (Ollama) are handled with mock fallbacks

## Thread Safety

The `GlobuleAPI` is designed to be used in async contexts and handles concurrent access to shared resources (database connections, embedding services) safely.