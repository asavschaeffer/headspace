from core.graph_analysis import DependencyGraphReasoner
from pathlib import Path

def test_graph_reasoner():
    print("Testing DependencyGraphReasoner...")
    
    metadata_list = [
        {
            "path": "/src/app/main.py",
            "name": "main.py",
            "imports": ["utils", "config"],
            "type": "code"
        },
        {
            "path": "/src/lib/utils.py",
            "name": "utils.py",
            "imports": [],
            "type": "code"
        },
        {
            "path": "/src/app/config.py",
            "name": "config.py",
            "imports": [],
            "type": "code"
        }
    ]
    
    reasoner = DependencyGraphReasoner()
    decisions = reasoner.reason(metadata_list)
    
    print(f"Decisions: {decisions}")
    
    # Expect a decision to group utils.py with main.py
    group_decisions = [d for d in decisions if d.action == "group_files"]
    assert len(group_decisions) >= 1
    
    # Check specific decision
    # We look for a decision where target is utils.py
    utils_decision = next((d for d in group_decisions if "utils.py" in d.target_path), None)
    assert utils_decision is not None
    
    # Normalize paths for comparison
    expected_dest = str(Path("/src/app"))
    # On windows /src/app might become \src\app or similar depending on drive
    # Let's just check if the name of the parent matches
    assert Path(utils_decision.destination_path).name == "app"
    
    print("DependencyGraphReasoner Verified!")

    print("Testing CouplingReasoner...")
    from core.graph_analysis import CouplingReasoner
    
    # Add a "God Object"
    god_object = {
        "path": "/src/god.py",
        "name": "god.py",
        "imports": ["a", "b", "c", "d", "e", "f"],
        "type": "code"
    }
    # Add dependencies for god object
    deps = [
        {"path": f"/src/{x}.py", "name": f"{x}.py", "imports": ["god"], "type": "code"}
        for x in ["a", "b", "c", "d", "e", "f"]
    ]
    
    coupling_metadata = [god_object] + deps
    
    c_reasoner = CouplingReasoner()
    c_decisions = c_reasoner.reason(coupling_metadata)
    
    print(f"Coupling Decisions: {c_decisions}")
    
    refactor_decision = next((d for d in c_decisions if d.action == "refactor" and "god.py" in d.target_path), None)
    assert refactor_decision is not None
    assert refactor_decision.metadata['fan_out'] == 6
    assert refactor_decision.metadata['fan_in'] == 6
    
    print("CouplingReasoner Verified!")

if __name__ == "__main__":
    test_graph_reasoner()
