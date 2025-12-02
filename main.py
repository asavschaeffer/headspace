import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from indexer import index_directory
from database import Database
from llm_client import get_llm_client
from core.reasoning import DecisionMaker, Decision
from core.graph_analysis import DependencyGraphReasoner, CouplingReasoner
from core.clustering import SemanticClusteringReasoner, DuplicateDetector
from core.llm_reasoning import LLMReasoningStrategy
from core.proposals import ProposalGenerator
from core.execution import FileOperationExecutor

def cmd_index(args):
    """Run the file indexer."""
    print(f"Indexing directory: {args.directory}")
    index_directory(args.directory, use_llm=not args.no_llm)

def cmd_reason(args):
    """Run reasoning strategies and generate proposals."""
    print("Loading file metadata...")
    db = Database()
    
    # Fetch all files
    # We need a method to get all files as dictionaries
    # For now, we'll do a raw query
    files = []
    with db._get_conn() as conn:
        cursor = conn.execute("SELECT path, summary, type, topics, extra_metadata FROM files")
        for row in cursor:
            path, summary, type_, topics, extra_json = row
            meta = {
                "path": path,
                "summary": summary,
                "type": type_,
                "topics": topics
            }
            if extra_json:
                try:
                    meta.update(json.loads(extra_json))
                except:
                    pass
            files.append(meta)
            
    print(f"Loaded {len(files)} files.")
    
    # Initialize Decision Maker
    maker = DecisionMaker()
    
    # Add Strategies
    # TODO: Make this configurable via args
    print("Initializing strategies...")
    maker.add_strategy(DependencyGraphReasoner())
    maker.add_strategy(CouplingReasoner())
    maker.add_strategy(SemanticClusteringReasoner())
    maker.add_strategy(DuplicateDetector())
    
    if args.use_llm:
        print("Initializing LLM strategy (this may take time/cost)...")
        client = get_llm_client()
        maker.add_strategy(LLMReasoningStrategy(client))
        
    # Run Reasoning
    print("Running reasoning strategies...")
    decisions = maker.make_decisions(files)
    print(f"Generated {len(decisions)} raw decisions.")
    
    # Generate Proposal
    generator = ProposalGenerator(confidence_threshold=args.threshold)
    
    # Report
    report = generator.generate_report(decisions)
    report_path = Path("proposal.md")
    report_path.write_text(report, encoding="utf-8")
    print(f"Proposal report saved to {report_path}")
    
    # JSON Plan
    plan_path = generator.save_proposals(decisions, "proposed_changes.json")
    print(f"Execution plan saved to {plan_path}")
    print("\nTo execute these changes, run:")
    print(f"python main.py execute {plan_path}")

def cmd_execute(args):
    """Execute a proposal plan."""
    plan_path = Path(args.plan_file)
    if not plan_path.exists():
        print(f"Plan file {plan_path} not found.")
        return
        
    print(f"Loading plan from {plan_path}...")
    with open(plan_path, 'r') as f:
        data = json.load(f)
        
    decisions_data = data.get("decisions", [])
    if not decisions_data:
        print("No decisions found in plan.")
        return
        
    print(f"Found {len(decisions_data)} decisions to execute.")
    if not args.yes:
        confirm = input("Are you sure you want to proceed? (y/N): ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return
            
    executor = FileOperationExecutor(workspace_root=".") # Assume current dir is root for now
    
    success_count = 0
    fail_count = 0
    
    for d_data in decisions_data:
        # Reconstruct Decision object
        # Note: Decision dataclass might have more fields than in JSON if we added some, 
        # but asdict/unpacking should work for basic fields.
        # We need to be careful about fields that are not in the JSON but required by Decision constructor.
        # Our Decision class has defaults for most things except action and target_path.
        
        # Filter out unknown fields just in case
        valid_fields = Decision.__annotations__.keys()
        filtered_data = {k: v for k, v in d_data.items() if k in valid_fields}
        
        decision = Decision(**filtered_data)
        
        print(f"Executing: {decision.action} {decision.target_path}...")
        if executor.execute(decision):
            print("  Success")
            success_count += 1
        else:
            print("  Failed")
            fail_count += 1
            
    print(f"\nExecution complete. Success: {success_count}, Failed: {fail_count}")

def cmd_rollback(args):
    """Rollback a transaction."""
    executor = FileOperationExecutor(workspace_root=".")
    
    if args.transaction_id:
        print(f"Rolling back transaction {args.transaction_id}...")
        if executor.rollback(args.transaction_id):
            print("Rollback successful.")
        else:
            print("Rollback failed.")
    else:
        # Show recent transactions
        print("Recent transactions:")
        for t in reversed(executor.log.transactions[-10:]):
            print(f"[{t.id}] {t.timestamp}: {t.action} {t.target_path} ({t.status})")

def main():
    parser = argparse.ArgumentParser(description="AI-OS Reorganization Engine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Index Command
    parser_index = subparsers.add_parser("index", help="Index a directory")
    parser_index.add_argument("directory", nargs="?", default=".", help="Directory to index")
    parser_index.add_argument("--no-llm", action="store_true", help="Disable LLM summarization")
    parser_index.set_defaults(func=cmd_index)
    
    # Reason Command
    parser_reason = subparsers.add_parser("reason", help="Run reasoning strategies")
    parser_reason.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser_reason.add_argument("--use-llm", action="store_true", help="Enable LLM reasoning (slower/costly)")
    parser_reason.set_defaults(func=cmd_reason)
    
    # Execute Command
    parser_exec = subparsers.add_parser("execute", help="Execute a proposal plan")
    parser_exec.add_argument("plan_file", help="Path to JSON plan file")
    parser_exec.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    parser_exec.set_defaults(func=cmd_execute)
    
    # Rollback Command
    parser_rollback = subparsers.add_parser("rollback", help="Rollback a transaction")
    parser_rollback.add_argument("transaction_id", nargs="?", help="Transaction ID to rollback")
    parser_rollback.set_defaults(func=cmd_rollback)
    
    args = parser.parse_args()
    
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
