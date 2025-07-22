package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "os"
    "os/exec"
    "path/filepath"
    "strings"
)

// NoteMetadata stores parsed metadata
type NoteMetadata struct {
    Path      string   `json:"path"`
    Vault     string   `json:"vault"`
    Frontmatter map[string]interface{} `json:"frontmatter"`
    Hashtags  []string `json:"hashtags"`
}

// EmbeddingResult stores embedding
type EmbeddingResult struct {
    Path      string    `json:"path"`
    Embedding []float32 `json:"embedding"`
}

// ClusterResult stores clustering
type ClusterResult struct {
    ClusterID int      `json:"cluster_id"`
    Files     []string `json:"files"`
}

// AnalysisAgent handles analysis
type AnalysisAgent struct {
    inputDir     string
    outputDir    string
    ollamaURL    string
    model        string
    parseScript  string
    embedScript  string
    notes        []NoteMetadata
    embeddings   []EmbeddingResult
    clusters     []ClusterResult
}

// NewAnalysisAgent initializes
func NewAnalysisAgent(inputDir, outputDir, ollamaURL, model, parseScript, embedScript string) *AnalysisAgent {
    return &AnalysisAgent{
        inputDir:    inputDir,
        outputDir:   outputDir,
        ollamaURL:   ollamaURL,
        model:       model,
        parseScript: parseScript,
        embedScript: embedScript,
    }
}

// listFiles lists .md files
func (a *AnalysisAgent) listFiles() ([]string, error) {
    var files []string
    err := filepath.Walk(a.inputDir, func(path string, info os.FileInfo, err error) error {
        if err != nil {
            return err
        }
        if !info.IsDir() && strings.HasSuffix(path, ".md") {
            files = append(files, path)
        }
        return nil
    })
    return files, err
}

// readFile reads content
func readFile(path string) (string, error) {
    content, err := os.ReadFile(path)
    if err != nil {
        return "", err
    }
    return string(content), nil
}

// parseNote extracts metadata
func (a *AnalysisAgent) parseNote(path string) (NoteMetadata, error) {
    content, err := readFile(path)
    if err != nil {
        return NoteMetadata{}, err
    }

    tmpFile, err := os.CreateTemp("", "note_*.md")
    if err != nil {
        return NoteMetadata{}, err
    }
    defer os.Remove(tmpFile.Name())
    if _, err := tmpFile.WriteString(content); err != nil {
        return NoteMetadata{}, err
    }
    tmpFile.Close()

    cmd := exec.Command("python", a.parseScript, tmpFile.Name())
    output, err := cmd.Output()
    if err != nil {
        return NoteMetadata{}, err
    }

    var metadata NoteMetadata
    if err := json.Unmarshal(output, &metadata); err != nil {
        return NoteMetadata{}, err
    }
    metadata.Path = path
    metadata.Vault = filepath.Base(filepath.Dir(path))
    return metadata, nil
}

// getEmbeddingsAndClusters generates embeddings
func (a *AnalysisAgent) getEmbeddingsAndClusters() error {
    files, err := a.listFiles()
    if err != nil {
        return err
    }

    for i, path := range files {
        fmt.Printf("Processing %s (%d/%d)\n", path, i+1, len(files))
        metadata, err := a.parseNote(path)
        if err != nil {
            fmt.Printf("Error parsing %s: %v\n", path, err)
            continue
        }
        a.notes = append(a.notes, metadata)

        content, err := readFile(path)
        if err != nil {
            continue
        }

        tmpFile, err := os.CreateTemp("", "doc_*.txt")
        if err != nil {
            continue
        }
        defer os.Remove(tmpFile.Name())
        if _, err := tmpFile.WriteString(content); err != nil {
            continue
        }
        tmpFile.Close()

        cmd := exec.Command("python", a.embedScript, tmpFile.Name())
        output, err := cmd.Output()
        if err != nil {
            fmt.Printf("Error embedding %s: %v\n", path, err)
            continue
        }

        var embedding []float32
        if err := json.Unmarshal(output, &embedding); err != nil {
            continue
        }
        a.embeddings = append(a.embeddings, EmbeddingResult{Path: path, Embedding: embedding})
    }

    // Save vault summaries
    for _, vault := range uniqueVaults(a.notes) {
        vaultNotes := filterNotesByVault(a.notes, vault)
        vaultSummary := map[string]interface{}{
            "vault":  vault,
            "notes":  vaultNotes,
            "count": len(vaultNotes),
        }
        data, err := json.MarshalIndent(vaultSummary, "", "  ")
        if err != nil {
            continue
        }
        os.MkdirAll(filepath.Join(a.outputDir, vault), 0755)
        os.WriteFile(filepath.Join(a.outputDir, vault, "summary.json"), data, 0644)
    }

    // Cluster
    tmpFile, err := os.CreateTemp("", "embeddings_*.json")
    if err != nil {
        return err
    }
    defer os.Remove(tmpFile.Name())
    if err := json.NewEncoder(tmpFile).Encode(a.embeddings); err != nil {
        return err
    }
    tmpFile.Close()

    cmd := exec.Command("python", a.embedScript, tmpFile.Name(), filepath.Join(a.outputDir, "clusters.json"), "--cluster")
    if _, err := cmd.Output(); err != nil {
        return err
    }

    data, err := os.ReadFile(filepath.Join(a.outputDir, "clusters.json"))
    if err != nil {
        return err
    }
    var result struct {
        Clusters []ClusterResult `json:"clusters"`
    }
    if err := json.Unmarshal(data, &result); err != nil {
        return err
    }
    a.clusters = result.Clusters
    return nil
}

// queryOllama queries LLM
func (a *AnalysisAgent) queryOllama(prompt string) (string, error) {
    payload := map[string]interface{}{
        "model":  a.model,
        "prompt": prompt,
        "stream": false,
    }
    body, err := json.Marshal(payload)
    if err != nil {
        return "", err
    }

    resp, err := http.Post(a.ollamaURL, "application/json", bytes.NewBuffer(body))
    if err != nil {
        return "", err
    }
    defer resp.Body.Close()

    respBody, err := io.ReadAll(resp.Body)
    if err != nil {
        return "", err
    }

    var result struct {
        Response string `json:"response"`
    }
    if err := json.Unmarshal(respBody, &result); err != nil {
        return "", err
    }
    return result.Response, nil
}

// AnalyzeClusters queries LLM
func (a *AnalysisAgent) AnalyzeClusters() (string, error) {
    var clusterStr strings.Builder
    for _, cluster := range a.clusters {
        clusterStr.WriteString(fmt.Sprintf("Cluster %d:\n", cluster.ClusterID))
        for i, path := range cluster.Files {
            content, err := readFile(path)
            if err != nil {
                continue
            }
            if len(content) > 500 {
                content = content[:500] + "..."
            }
            metadata := findNoteByPath(a.notes, path)
            clusterStr.WriteString(fmt.Sprintf(
                "File %d: %s\nVault: %s\nTags: %v\nContent Snippet: %s\n\n",
                i+1, path, metadata.Vault, metadata.Hashtags, content))
        }
    }

    prompt := fmt.Sprintf(`
Analyze these clusters of markdown notes:
%s

For each cluster:
- List main topics in bullets.
- Suggest additional tags.
- Propose a subdirectory (e.g., Documents/Programming/Go).
- Suggest YAML frontmatter (e.g., tags, vault).
Return concise plain text, no JSON.
`, clusterStr.String())

    return a.queryOllama(prompt)
}

// Helpers
func uniqueVaults(notes []NoteMetadata) []string {
    vaults := make(map[string]bool)
    for _, note := range notes {
        vaults[note.Vault] = true
    }
    var result []string
    for vault := range vaults {
        result = append(result, vault)
    }
    return result
}

func filterNotesByVault(notes []NoteMetadata, vault string) []NoteMetadata {
    var result []NoteMetadata
    for _, note := range notes {
        if note.Vault == vault {
            result = append(result, note)
        }
    }
    return result
}

func findNoteByPath(notes []NoteMetadata, path string) NoteMetadata {
    for _, note := range notes {
        if note.Path == path {
            return note
        }
    }
    return NoteMetadata{}
}

func main() {
    if len(os.Args) < 2 {
        fmt.Println("Usage: go run main.go <directory_path>")
        return
    }
    inputDir := os.Args[1]

    agent := NewAnalysisAgent(
        inputDir,
        "output",
        "http://localhost:11434/api/generate",
        "llama3.2:3b",
        "parse.py",
        "embed.py",
    )

    if err := agent.getEmbeddingsAndClusters(); err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }

    response, err := agent.AnalyzeClusters()
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    fmt.Printf("LLM Response:\n%s\n", response)
}