from pathlib import Path
from core.analyzers import CodeFileAnalyzer
import json

def test_analyzers():
    print("Testing CodeFileAnalyzer...")
    
    # Create a dummy python file
    dummy_code = """
import os
from datetime import datetime

class TestClass:
    def method_one(self):
        pass

def global_function():
    pass
"""
    dummy_path = Path("dummy_test.py")
    dummy_path.write_text(dummy_code)
    
    try:
        analyzer = CodeFileAnalyzer()
        metadata = analyzer.analyze(dummy_path)
        
        print(f"Metadata: {json.dumps(metadata, indent=2)}")
        
        assert "os" in metadata['imports']
        assert "datetime.datetime" in metadata['imports']
        assert "TestClass" in metadata['classes']
        assert "global_function" in metadata['functions']
        assert metadata['type'] == 'code'
        
        print("CodeFileAnalyzer Verified!")
        
    finally:
        if dummy_path.exists():
            dummy_path.unlink()

if __name__ == "__main__":
    test_analyzers()
