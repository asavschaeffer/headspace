import json
from typing import List, Dict, Any
from core.reasoning import ReasoningStrategy, Decision

class LLMReasoningStrategy(ReasoningStrategy):
    """
    Uses an LLM to propose organization decisions based on file metadata.
    """
    
    def __init__(self, llm_client):
        self.client = llm_client
        
    def _clean_json_response(self, response: str) -> str:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        return response.strip()

    def reason(self, file_metadata: List[Dict[str, Any]]) -> List[Decision]:
        # Prepare context for LLM
        # We need to be careful about context window limits.
        # For now, we'll send a simplified list of files.
        
        files_context = []
        for meta in file_metadata:
            files_context.append({
                "path": meta['path'],
                "type": meta.get('type', 'unknown'),
                "summary": meta.get('summary', ''), # Assuming summary might be in metadata or we fetch it
                "imports": meta.get('imports', [])
            })
            
        prompt = f"""
You are an expert software architect. Your task is to analyze the following list of files and propose a better directory structure or organization.

Files:
{json.dumps(files_context, indent=2)}

Rules:
1. Group related files together.
2. Separate concerns (e.g., core logic vs UI vs utils).
3. Identify potential duplicates or files that should be merged.
4. Propose specific actions: "move", "merge", "delete", "rename".

Output Format:
Return a JSON object with a key "decisions" containing a list of decisions.
Each decision should have:
- "action": "move", "merge", "delete", "rename"
- "target_path": The current path of the file.
- "destination_path": The new path (for move/rename) or target file (for merge).
- "reasoning": A brief explanation.
- "confidence": A score between 0.0 and 1.0.

Example Output:
{{
  "decisions": [
    {{
      "action": "move",
      "target_path": "/src/utils.py",
      "destination_path": "/src/common/utils.py",
      "reasoning": "Utils should be in a common directory.",
      "confidence": 0.9
    }}
  ]
}}

Respond ONLY with the JSON.
"""
        try:
            response = self.client.ask(prompt)
            cleaned_response = self._clean_json_response(response)
            
            try:
                data = json.loads(cleaned_response)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON from LLM response. Response snippet: {cleaned_response[:100]}...")
                return []
            
            decisions = []
            for d in data.get("decisions", []):
                decisions.append(Decision(
                    action=d.get("action"),
                    target_path=d.get("target_path"),
                    destination_path=d.get("destination_path"),
                    reasoning=d.get("reasoning"),
                    confidence=d.get("confidence", 0.5),
                    metadata={"source": "llm"}
                ))
            return decisions
            
        except Exception as e:
            print(f"Error in LLM reasoning: {e}")
            return []
