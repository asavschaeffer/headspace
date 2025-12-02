from core.relationships import RelationshipDetector

def test_relationships():
    print("Testing RelationshipDetector...")
    
    metadata_list = [
        {
            "path": "/src/utils.py",
            "name": "utils.py",
            "imports": ["os", "json"],
            "type": "code"
        },
        {
            "path": "/src/main.py",
            "name": "main.py",
            "imports": ["utils", "sys"],
            "type": "code"
        },
        {
            "path": "/src/other.py",
            "name": "other.py",
            "imports": ["main"],
            "type": "code"
        }
    ]
    
    detector = RelationshipDetector()
    relationships = detector.detect_relationships(metadata_list)
    
    print(f"Relationships: {relationships}")
    
    assert "/src/utils.py" in relationships["/src/main.py"]
    assert "/src/main.py" in relationships["/src/other.py"]
    
    print("RelationshipDetector Verified!")

if __name__ == "__main__":
    test_relationships()
