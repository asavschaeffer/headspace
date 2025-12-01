import json
from typing import List, Dict
from pathlib import Path
from dataclasses import asdict
from core.reasoning import Decision

class ProposalGenerator:
    """
    Generates human-readable reports and machine-readable plans from decisions.
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        
    def generate_report(self, decisions: List[Decision]) -> str:
        """
        Generates a Markdown report of the proposed changes.
        """
        filtered_decisions = [d for d in decisions if d.confidence >= self.confidence_threshold]
        
        if not filtered_decisions:
            return "No changes proposed (no decisions met the confidence threshold)."
            
        # Group by action
        grouped = {}
        for d in filtered_decisions:
            if d.action not in grouped:
                grouped[d.action] = []
            grouped[d.action].append(d)
            
        report = ["# AI-OS Reorganization Proposal\n"]
        report.append(f"**Total Proposed Changes:** {len(filtered_decisions)}\n")
        
        for action, items in grouped.items():
            report.append(f"## Action: {action.upper()} ({len(items)})")
            for d in items:
                report.append(f"- **{d.target_path}**")
                if d.destination_path:
                    report.append(f"  - -> {d.destination_path}")
                report.append(f"  - *Reasoning:* {d.reasoning}")
                report.append(f"  - *Confidence:* {d.confidence:.2f}")
            report.append("")
            
        return "\n".join(report)
        
    def save_proposals(self, decisions: List[Decision], output_path: str = "proposed_changes.json"):
        """
        Saves the filtered decisions to a JSON file.
        """
        filtered_decisions = [d for d in decisions if d.confidence >= self.confidence_threshold]
        
        data = {
            "metadata": {
                "total_decisions": len(decisions),
                "filtered_decisions": len(filtered_decisions),
                "threshold": self.confidence_threshold
            },
            "decisions": [asdict(d) for d in filtered_decisions]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        return Path(output_path).resolve()
