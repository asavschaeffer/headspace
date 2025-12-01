from pathlib import Path
from core.analyzers import BaseFileAnalyzer
from core.reasoning import DecisionMaker, ReasoningStrategy, Decision
from typing import List, Dict, Any

# Mock Strategy
class MockStrategy(ReasoningStrategy):
    def reason(self, file_metadata: List[Dict[str, Any]]) -> List[Decision]:
        decisions = []
        for meta in file_metadata:
            if meta['size'] > 1000:
                decisions.append(Decision(
                    action="review",
                    target_path=meta['path'],
                    reasoning="File is large",
                    confidence=0.8
                ))
        return decisions

def test_core():
    print("Testing Core Abstractions...")
    
    # 1. Test Analyzer
    analyzer = BaseFileAnalyzer()
    # Analyze self
    meta = analyzer.analyze(Path(__file__))
    print(f"Analyzed self: {meta}")
    assert meta['name'] == 'test_core.py'
    
    # 2. Test Decision Maker
    maker = DecisionMaker()
    maker.add_strategy(MockStrategy())
    
    decisions = maker.make_decisions([meta])
    print(f"Decisions: {decisions}")
    
    # Since this file is small, it might not trigger the mock strategy if size < 1000
    # Let's force a large size in metadata
    meta['size'] = 2000
    decisions = maker.make_decisions([meta])
    print(f"Decisions (forced large): {decisions}")
    assert len(decisions) == 1
    assert decisions[0].action == "review"
    
    print("Core Abstractions Verified!")

if __name__ == "__main__":
    test_core()
