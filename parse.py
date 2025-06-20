import sys
import json
import frontmatter

def parse_note(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        note = frontmatter.load(f)
    
    hashtags = []
    for line in note.content.split('\n'):
        for word in line.split():
            if word.startswith('#') and len(word) > 1:
                hashtags.append(word[1:].lower())
    
    return {
        "path": file_path,
        "vault": "",
        "frontmatter": note.metadata,
        "hashtags": list(set(hashtags))
    }

if __name__ == "__main__":
    file_path = sys.argv[1]
    metadata = parse_note(file_path)
    print(json.dumps(metadata))