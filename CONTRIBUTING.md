# Contributing to Globule

Thank you for your interest in contributing to Globule! We welcome contributions of all kinds, from bug fixes to new features and documentation improvements.

## Project Philosophy

Our goal is to build a powerful, flexible, and transparent tool for thought. We adhere to a few core principles:

- **API-Driven Architecture:** The application is built around a clean, stable `GlobuleAPI`. All features are implemented at this layer first, and all UIs are simple clients of the API.
- **Clarity and Simplicity:** We prefer clear, simple code over complex abstractions.
- **Robustness:** We value comprehensive testing and robust error handling.

Before starting any major work, please review the [Architecture Guide](docs/architecture.md).

## Setting Up the Development Environment

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/asavschaeffer/globule.git
    cd globule
    ```

2.  **Create a Virtual Environment:**
    We strongly recommend using a virtual environment to manage dependencies.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    Install the base package in editable mode, along with all development and optional dependencies.
    ```bash
    pip install -e .[dev,clustering]
    ```

4.  **Set Up Ollama:**
    Globule depends on a running Ollama instance for its AI capabilities. Please [download and install Ollama](https://ollama.com/) for your platform.

    Once installed, you must pull the models required by the application:
    ```bash
    ollama pull <embedding_model_name>  # See pyproject.toml for model names
    ollama pull <parsing_model_name>   # See pyproject.toml for model names
    ```
    Ensure the Ollama application is running before you start Globule.

## Running Tests

We use `pytest` for testing. To run the full test suite:

```bash
pytest
```

## Code Style and Quality

We use a few tools to maintain code quality. Please run them before submitting a pull request.

-   **Formatting (`black` and `isort`):**
    ```bash
    black .
    isort .
    ```
-   **Linting (`ruff`):**
    ```bash
    ruff check .
    ```
-   **Type Checking (`mypy`):**
    ```bash
    mypy src
    ```

## Commit Message Convention

We adhere to the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. This helps in automating changelog generation and keeps the project history clean and understandable.

The format is `type(scope): message`.

-   **Types**: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `style`, `ci`, `perf`
-   **Scope**: The part of the codebase affected (e.g., `core`, `api`, `tui`, `ci`).

**Examples:**
-   `feat(api): add method for natural language search`
-   `fix(tui): correct off-by-one error in palette display`
-   `docs(schemas): create new schema authoring guide`

## Submitting Changes

1.  Create a new feature branch from the `main` branch.
2.  Make your changes on the feature branch.
3.  Ensure all tests and quality checks pass.
4.  Push your branch and open a pull request against the `main` branch.
5.  Provide a clear description of your changes in the pull request.