## JARVIS Mini
### System Requirements
- Go: https://go.dev/dl/ (1.22+)
- Python: 3.10.11
- Ollama: https://ollama.com, with `llama3.2:3b`
### Setup
```bash
python3 -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
ollama serve
go run main.go
```
### Usage
1) put .md files in input folder
2) run `go run main.go`
3) ...?