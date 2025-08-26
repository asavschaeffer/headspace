# Globule Architecture

This document describes the high-level architecture of the Globule application. The system is designed to be modular, maintainable, and testable, with a clear separation of concerns between its layers.

## Core Principles

- **API-Driven:** All application features are exposed through a clean, UI-agnostic `GlobuleAPI`. This is the single point of entry for any frontend or client.
- **Separation of Concerns:** The codebase is divided into distinct layers, each with a single responsibility.
- **Centralized State:** The core application logic is stateless where possible, and any necessary state (like database connections or API clients) is managed within a well-defined context.

## Architectural Layers

The application is structured as three primary layers:

1.  **Interface Layer:** The user-facing components.
2.  **API Layer:** The central, unified interface to the application's core logic.
3.  **Core Logic Layer:** The backend services responsible for data processing, storage, and external integrations.

```
+----------------------------------------------------+
|                 Interface Layer                    |
|  (CLI, TUI, Web UI, Glass Engine)                  |
+-----------------------+----------------------------+
                        |
                        v
+-----------------------+----------------------------+
|                     API Layer                      |
|                  (GlobuleAPI)                      |
+-----------------------+----------------------------+
                        |
                        v
+-----------------------+----------------------------+
|                 Core Logic Layer                   |
| (Orchestrator, Services, Storage, Models)          |
+----------------------------------------------------+
```

### 1. Interface Layer

This layer is responsible for presenting data to the user and accepting user input. It does not contain any business logic.

-   **`interfaces/cli/main.py`:** The command-line interface, built with `click`. It parses commands and arguments and calls the `GlobuleAPI`.
-   **`tui/app.py`:** The Textual-based Terminal User Interface. It is a pure presentation layer that receives all its data from the `GlobuleAPI`.
-   **`tutorial/`:** The Glass Engine tutorial system. It acts as a client of the `GlobuleAPI` to demonstrate the application's features in a sandboxed environment.
-   **`inputs/`:** Messaging platform integrations that capture thoughts from external sources:
    -   **`adapters/`:** Platform-specific adapters (WhatsApp, Telegram, Email) that parse webhook messages into standardized `InputMessage` objects.
    -   **`relay_service.py`:** Minimal cloud service that forwards webhooks from messaging platforms to local instances.
    -   **`manager.py`:** Coordinates multiple input sources and routes messages to the `GlobuleAPI`.

### 2. API Layer

This is the most critical layer for maintaining the application's structure.

-   **`core/api.py`:** Defines the `GlobuleAPI` class. This class is the sole entry point for the interface layer. It exposes high-level, business-oriented methods like `add_thought()`, `add_from_input_message()`, `search_semantic()`, and `get_clusters()`. It encapsulates the complexity of the core logic layer.

### 3. Core Logic Layer

This layer contains the implementation details of the application's features.

-   **`orchestration/engine.py`:** The `OrchestrationEngine` is the heart of the core logic. It coordinates the various services to perform complex tasks, such as enriching raw text with metadata and vector embeddings.
-   **`services/`:** Contains modules for interacting with external services or performing complex data processing. This includes:
    -   `embedding/`: Generating vector embeddings via Ollama.
    -   `parsing/`: Extracting structured data from text via Ollama.
    -   `clustering/`: Grouping related thoughts using machine learning.
-   **`storage/`:** The data persistence layer. The `SQLiteStorageManager` is responsible for all database interactions.
-   **`core/models.py`:** Defines the Pydantic data models (e.g., `GlobuleV1`, `ProcessedGlobuleV1`) that serve as the data contracts between the layers.

## Data Flow Example: `globule add`

1.  The user runs `globule add "some text"`.
2.  The `add` command in `interfaces/cli/main.py` is executed.
3.  It calls `api.add_thought("some text")` on the `GlobuleAPI` instance.
4.  The `GlobuleAPI` calls the `OrchestrationEngine`.
5.  The `OrchestrationEngine` concurrently calls the `EmbeddingService` and `ParsingService`.
6.  The services make requests to the Ollama API.
7.  The `OrchestrationEngine` combines the results into a `ProcessedGlobuleV1` object.
8.  The `OrchestrationEngine` passes the processed globule to the `StorageManager`, which saves it to the database.
9.  The result is returned up the call stack to the CLI, which prints a confirmation message.
