# Project Refactoring and Analysis Pipeline Plan

## 1. Project Goal

The primary goal is to transform the project into a streamlined, end-to-end analysis tool. The tool will take a directory of Markdown documents as input and produce a set of insightful outputs, including cluster visualizations and keyword-based topic summaries.

## 2. The New `analyze_documents.py` Script

The core of the refactoring will be to consolidate the existing Python logic into a single, powerful script named `analyze_documents.py`. This script will be the new entry point for the entire analysis pipeline and will be responsible for all processing steps.

### Key Features:

*   **Command-Line Interface:** The script will accept the input directory path as a command-line argument for ease of use.
*   **End-to-End Pipeline:** It will handle every step of the process internally:
    1.  **Parsing:** Recursively find and parse all Markdown files (`.md`) in the input directory.
    2.  **Embedding:** Use the `sentence-transformers` library to generate a vector embedding for the content of each document.
    3.  **Clustering:** Employ the `hdbscan` library to cluster the documents based on their embeddings. This will group semantically similar documents together.
    4.  **Keyword Extraction:** For each cluster, use a TF-IDF (Term Frequency-Inverse Document Frequency) approach to identify the most representative keywords. This will provide an "abstract understanding" of the topics within each cluster.
    5.  **Visualization:** Generate an interactive 2D scatter plot of the document clusters using `plotly`. This allows for visual exploration of the relationships between documents.

## 3. Output Organization

To ensure that results are never overwritten and can be easily compared, the script will create a new, uniquely named directory for each run inside the `output` folder. The directory name will be timestamped (e.g., `run_YYYY-MM-DD_HH-MM-SS`).

### Each run directory will contain:

*   `clustered_data.json`: A comprehensive JSON file containing the original document content, its vector embedding, and its assigned cluster label.
*   `cluster_keywords.json`: A JSON file mapping each cluster ID to a list of its top extracted keywords.
*   `cluster_visualization.html`: An interactive HTML file containing the Plotly scatter plot of the document clusters.

## 4. Project Cleanup

To simplify the project structure and remove obsolete components, the following files will be deleted:

*   `parse.py`
*   `embed.py`
*   `main.go`
*   `claudescaffold.txt`
*   `repomix-output.xml`

## 5. Documentation

The `readme.md` file will be updated with clear, concise instructions on how to:
1.  Install the required dependencies from `requirements.txt`.
2.  Run the analysis using the `analyze_documents.py` script.
3.  Interpret the generated output files.

## 6. Final Project Structure

After the refactoring, the project directory will look like this:

```
C:/Users/18312/Projects/JARVIS/
├───.gitignore
├───analyze_documents.py  <-- New
├───plan.md
├───readme.md
├───requirements.txt
├───todo.md
├───.git/
├───.venv/
├───input/
│   └── (Your Markdown files)
└───output/
    └── run_YYYY-MM-DD_HH-MM-SS/  <-- Example output
        ├── clustered_data.json
        ├── cluster_keywords.json
        └── cluster_visualization.html
```
