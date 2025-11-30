"""
Headspace Generator V2 - Enhanced data preparation for cosmic visualization
With clustering, semantic connections, and orbital systems
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import hashlib
import colorsys
from pathlib import Path


@dataclass
class ChunkData:
    """Data structure for a chunk in the visualization"""
    id: str
    text: str
    position: List[float]
    color: str
    size: float
    document_id: str
    index: int
    importance: float = 1.0
    attachments: List[Dict] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.attachments is None:
            self.attachments = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ClusterData:
    """Data structure for a semantic cluster (nebula)"""
    id: str
    center: List[float]
    radius: float
    color: str
    chunk_ids: List[str]
    keywords: List[str] = None

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


@dataclass
class ConnectionData:
    """Data structure for connections between chunks"""
    from_id: str
    to_id: str
    type: str  # 'sequential', 'semantic', 'reference'
    similarity: float = 0.0
    strength: float = 1.0


class HeadspaceGenerator:
    """Generate enhanced Headspace visualization data"""

    def __init__(self):
        self.chunks: List[ChunkData] = []
        self.clusters: List[ClusterData] = []
        self.connections: List[ConnectionData] = []
        self.embeddings = None
        self.positions_3d = None

    def generate_from_rust_code(self, code: str, filename: str = "main.rs") -> Dict:
        """
        Generate Headspace visualization from Rust code

        Args:
            code: Rust source code
            filename: Name of the file

        Returns:
            Dictionary with visualization data
        """
        # Split code into logical chunks
        chunks_raw = self._split_rust_code(code)

        # Generate mock embeddings (in production, use real embeddings)
        embeddings = self._generate_mock_embeddings(chunks_raw)

        # Calculate 3D positions
        positions = self._calculate_3d_positions_simple(embeddings)

        # Create chunk objects
        for i, chunk_text in enumerate(chunks_raw):
            chunk_type = self._identify_rust_chunk_type(chunk_text)
            color = self._get_color_for_code_type(chunk_type)

            chunk_data = ChunkData(
                id=f"{filename}:chunk_{i}",
                text=chunk_text,
                position=positions[i].tolist(),
                color=color,
                size=5 + len(chunk_text) / 100,  # Size based on text length
                document_id=filename,
                index=i,
                importance=self._calculate_importance(chunk_text),
                attachments=self._generate_mock_attachments(i, chunk_text),
                metadata={'type': chunk_type, 'line_count': len(chunk_text.split('\n'))}
            )
            self.chunks.append(chunk_data)

        # Generate connections
        self._generate_code_connections()

        # Generate clusters
        self._generate_code_clusters(embeddings)

        return self._export_data()

    def _split_rust_code(self, code: str) -> List[str]:
        """Split Rust code into logical chunks"""
        chunks = []
        lines = code.split('\n')
        current_chunk = []

        for line in lines:
            current_chunk.append(line)

            # Split on major code boundaries
            if any(keyword in line for keyword in ['fn ', 'impl ', 'struct ', 'enum ', 'mod ']):
                if len(current_chunk) > 1:
                    # Save previous chunk
                    chunks.append('\n'.join(current_chunk[:-1]))
                    current_chunk = [line]
            elif line.strip() == '}' and len(current_chunk) > 10:
                # End of a block
                chunks.append('\n'.join(current_chunk))
                current_chunk = []

        # Add remaining lines
        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        # Filter out empty chunks
        return [c for c in chunks if c.strip()]

    def _identify_rust_chunk_type(self, chunk: str) -> str:
        """Identify the type of Rust code chunk"""
        if 'pub enum' in chunk or 'enum ' in chunk:
            return 'enum'
        elif 'impl ' in chunk:
            return 'impl'
        elif 'pub fn' in chunk or 'fn ' in chunk:
            return 'function'
        elif 'pub struct' in chunk or 'struct ' in chunk:
            return 'struct'
        elif 'use ' in chunk or 'extern crate' in chunk:
            return 'imports'
        elif '//' in chunk or '/*' in chunk:
            return 'comment'
        else:
            return 'code'

    def _get_color_for_code_type(self, code_type: str) -> str:
        """Get color based on code type"""
        colors = {
            'enum': '#9B59B6',      # Purple for types
            'struct': '#8E44AD',    # Darker purple for structs
            'impl': '#E67E22',      # Orange for implementations
            'function': '#3498DB',  # Blue for functions
            'imports': '#95A5A6',   # Gray for imports
            'comment': '#27AE60',   # Green for comments
            'code': '#667EEA'       # Default blue
        }
        return colors.get(code_type, '#667EEA')

    def _calculate_importance(self, chunk: str) -> float:
        """Calculate importance score for a chunk"""
        score = 0.5

        # Public items are more important
        if 'pub ' in chunk:
            score += 0.3

        # Main function is important
        if 'fn main' in chunk:
            score += 0.4

        # Enums and structs are important
        if 'enum ' in chunk or 'struct ' in chunk:
            score += 0.2

        return min(score, 1.0)

    def _generate_mock_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate mock embeddings for demo (replace with real embeddings)"""
        n_chunks = len(chunks)
        embeddings = []

        for i, chunk in enumerate(chunks):
            # Create a simple embedding based on content
            chunk_type = self._identify_rust_chunk_type(chunk)

            # Base embedding from chunk type
            type_embeddings = {
                'enum': [1, 0, 0, 0, 0],
                'struct': [0.9, 0.1, 0, 0, 0],
                'impl': [0, 1, 0, 0, 0],
                'function': [0, 0, 1, 0, 0],
                'imports': [0, 0, 0, 1, 0],
                'comment': [0, 0, 0, 0, 1],
                'code': [0.2, 0.2, 0.2, 0.2, 0.2]
            }

            base = type_embeddings.get(chunk_type, [0.2, 0.2, 0.2, 0.2, 0.2])

            # Add some noise and position influence
            embedding = base + np.random.randn(5) * 0.1
            embedding = np.append(embedding, [i / n_chunks, len(chunk) / 1000])

            embeddings.append(embedding)

        return np.array(embeddings)

    def _calculate_3d_positions_simple(self, embeddings: np.ndarray) -> np.ndarray:
        """Simple 3D position calculation"""
        n = len(embeddings)
        positions = np.zeros((n, 3))

        # Create a spiral layout for code
        for i in range(n):
            angle = i * 0.5
            radius = 20 + i * 0.5
            height = i * 2 - n

            positions[i] = [
                radius * np.cos(angle),
                height,
                radius * np.sin(angle)
            ]

        # Add some variation based on embeddings
        for i in range(n):
            positions[i] += embeddings[i][:3] * 5

        return positions

    def _generate_mock_attachments(self, index: int, chunk: str) -> List[Dict]:
        """Generate mock attachments for demo"""
        attachments = []

        # Add explanations for important chunks
        if 'pub enum' in chunk:
            attachments.append({
                'id': f'explanation_{index}',
                'title': 'Type Explanation',
                'type': 'explanation',
                'color': '#FF6B6B',
                'content': 'This enum defines the token types used in parsing.'
            })
        elif 'impl ' in chunk:
            attachments.append({
                'id': f'impl_note_{index}',
                'title': 'Implementation Details',
                'type': 'note',
                'color': '#4ECDC4',
                'content': 'Implementation block containing methods for this type.'
            })

        return attachments

    def _generate_code_connections(self):
        """Generate connections for code visualization"""
        # Sequential connections
        for i in range(len(self.chunks) - 1):
            self.connections.append(ConnectionData(
                from_id=self.chunks[i].id,
                to_id=self.chunks[i+1].id,
                type='sequential',
                similarity=0.0,
                strength=0.5
            ))

        # Add semantic connections based on code relationships
        for i, chunk1 in enumerate(self.chunks):
            for j, chunk2 in enumerate(self.chunks[i+1:], i+1):
                # Connect enum to its impl
                if 'enum TokenType' in chunk1.text and 'impl TokenType' in chunk2.text:
                    self.connections.append(ConnectionData(
                        from_id=chunk1.id,
                        to_id=chunk2.id,
                        type='semantic',
                        similarity=0.9,
                        strength=0.8
                    ))

                # Connect functions that reference each other
                if chunk1.metadata['type'] == 'function' and chunk2.metadata['type'] == 'function':
                    # Simple check for function calls (in production, use AST)
                    func_name1 = self._extract_function_name(chunk1.text)
                    func_name2 = self._extract_function_name(chunk2.text)

                    if func_name1 and func_name2:
                        if func_name1 in chunk2.text or func_name2 in chunk1.text:
                            self.connections.append(ConnectionData(
                                from_id=chunk1.id,
                                to_id=chunk2.id,
                                type='function_call',
                                similarity=0.7,
                                strength=0.6
                            ))

    def _extract_function_name(self, chunk: str) -> str:
        """Extract function name from chunk"""
        lines = chunk.split('\n')
        for line in lines:
            if 'fn ' in line:
                # Extract function name
                parts = line.split('fn ')
                if len(parts) > 1:
                    name = parts[1].split('(')[0].strip()
                    return name
        return ""

    def _generate_code_clusters(self, embeddings: np.ndarray):
        """Generate clusters for code visualization"""
        # Group by code type
        type_groups = {}
        for i, chunk in enumerate(self.chunks):
            chunk_type = chunk.metadata.get('type', 'code')
            if chunk_type not in type_groups:
                type_groups[chunk_type] = []
            type_groups[chunk_type].append(i)

        # Create a cluster for each type with multiple chunks
        for chunk_type, indices in type_groups.items():
            if len(indices) > 1:
                # Calculate cluster center
                positions = np.array([self.chunks[i].position for i in indices])
                center = positions.mean(axis=0)

                # Calculate radius
                distances = np.linalg.norm(positions - center, axis=1)
                radius = distances.max() * 1.2

                # Extract keywords
                keywords = self._extract_keywords_from_chunks([self.chunks[i] for i in indices])

                cluster = ClusterData(
                    id=f"cluster_{chunk_type}",
                    center=center.tolist(),
                    radius=float(radius),
                    color=self._get_color_for_code_type(chunk_type),
                    chunk_ids=[self.chunks[i].id for i in indices],
                    keywords=keywords
                )
                self.clusters.append(cluster)

    def _extract_keywords_from_chunks(self, chunks: List[ChunkData]) -> List[str]:
        """Extract keywords from a group of chunks"""
        keywords = set()
        for chunk in chunks:
            # Extract Rust keywords and identifiers
            text = chunk.text
            # Simple extraction of pub names
            if 'pub fn' in text:
                keywords.add('public function')
            if 'pub struct' in text:
                keywords.add('public struct')
            if 'impl' in text:
                keywords.add('implementation')
            if 'Vec<' in text:
                keywords.add('Vector')

        return list(keywords)[:5]

    def _export_data(self) -> Dict:
        """Export all data as dictionary"""
        return {
            'chunks': [asdict(chunk) for chunk in self.chunks],
            'clusters': [asdict(cluster) for cluster in self.clusters],
            'connections': [asdict(conn) for conn in self.connections]
        }

    def generate_headspace_html(self, data: Dict = None,
                               template_path: str = "headspace_v3_template.html",
                               output_path: str = "headspace_output.html") -> str:
        """Generate HTML file with visualization"""
        if data is None:
            data = self._export_data()

        # Read template
        template_file = Path(template_path)
        if not template_file.exists():
            print(f"Template not found: {template_path}")
            print("Using minimal template")
            template = self._get_minimal_template()
        else:
            with open(template_file, 'r', encoding='utf-8') as f:
                template = f.read()

        # Create JavaScript data
        js_data = f"""
        const chunksData = {json.dumps(data, indent=2)};
        const CHUNKS_DATA = chunksData;
        """

        # Inject data - handle multiline placeholder
        if "const CHUNKS_DATA" in template:
            # Find and replace the multiline placeholder using string operations
            # to avoid regex issues with special characters in JSON
            start_marker = "const CHUNKS_DATA = typeof chunksData"
            end_marker = "};"

            start_idx = template.find(start_marker)
            if start_idx != -1:
                # Find the end of the CHUNKS_DATA definition
                # Look for the closing }; after the start marker
                search_from = start_idx + len(start_marker)
                end_idx = template.find(end_marker, search_from)

                if end_idx != -1:
                    # Include the }; in the replacement
                    end_idx += len(end_marker)
                    # Replace the entire block
                    template = template[:start_idx] + js_data + template[end_idx:]
                else:
                    # Fallback: just replace the const CHUNKS_DATA line
                    # Find the end of the line
                    line_end = template.find('\n', start_idx)
                    if line_end != -1:
                        # Look for the closing }; on subsequent lines
                        block_end = template.find('};', line_end)
                        if block_end != -1:
                            template = template[:start_idx] + js_data + template[block_end + 2:]
                        else:
                            template = template[:start_idx] + js_data + template[line_end:]
            else:
                # Simple fallback
                template = template.replace("const CHUNKS_DATA", js_data, 1)
        else:
            # Insert before closing script tag
            template = template.replace("</script>", f"{js_data}\n</script>", 1)

        # Write output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template)

        print(f"✨ Headspace visualization generated: {output_path}")
        return output_path

    def _get_minimal_template(self) -> str:
        """Minimal template if file is missing"""
        return """<!DOCTYPE html>
<html>
<head>
    <title>Headspace - Cosmic Visualization</title>
    <style>
        body { margin: 0; overflow: hidden; background: #000; color: white; font-family: Arial; }
        #info { position: absolute; top: 10px; left: 10px; }
    </style>
</head>
<body>
    <div id="info">Headspace Visualization</div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // Data will be injected here
        console.log('Chunks:', CHUNKS_DATA);
    </script>
</body>
</html>"""


# Demo Rust code for testing
DEMO_RUST_CODE = """
use std::collections::HashMap;

/// Token types for the lexer
#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
    // Literals
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),

    // Identifiers and Keywords
    Identifier(String),
    Keyword(String),

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Equal,

    // Delimiters
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    Semicolon,

    // Special
    EOF,
}

/// Token structure containing type and position
#[derive(Debug, Clone)]
pub struct Token {
    pub token_type: TokenType,
    pub line: usize,
    pub column: usize,
    pub lexeme: String,
}

impl Token {
    /// Create a new token
    pub fn new(token_type: TokenType, line: usize, column: usize, lexeme: String) -> Self {
        Token {
            token_type,
            line,
            column,
            lexeme,
        }
    }

    /// Check if token is a keyword
    pub fn is_keyword(&self) -> bool {
        matches!(self.token_type, TokenType::Keyword(_))
    }

    /// Check if token is an operator
    pub fn is_operator(&self) -> bool {
        matches!(
            self.token_type,
            TokenType::Plus | TokenType::Minus | TokenType::Star | TokenType::Slash | TokenType::Equal
        )
    }
}

/// Lexer for tokenizing source code
pub struct Lexer {
    input: Vec<char>,
    position: usize,
    current_line: usize,
    current_column: usize,
}

impl Lexer {
    /// Create a new lexer
    pub fn new(input: &str) -> Self {
        Lexer {
            input: input.chars().collect(),
            position: 0,
            current_line: 1,
            current_column: 1,
        }
    }

    /// Get the next token
    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();

        if self.is_at_end() {
            return self.make_token(TokenType::EOF, "");
        }

        let start_column = self.current_column;
        let ch = self.advance();

        match ch {
            '+' => self.make_token(TokenType::Plus, "+"),
            '-' => self.make_token(TokenType::Minus, "-"),
            '*' => self.make_token(TokenType::Star, "*"),
            '/' => self.make_token(TokenType::Slash, "/"),
            '=' => self.make_token(TokenType::Equal, "="),
            '(' => self.make_token(TokenType::LeftParen, "("),
            ')' => self.make_token(TokenType::RightParen, ")"),
            '{' => self.make_token(TokenType::LeftBrace, "{"),
            '}' => self.make_token(TokenType::RightBrace, "}"),
            ';' => self.make_token(TokenType::Semicolon, ";"),
            _ => {
                if ch.is_alphabetic() {
                    self.identifier(start_column)
                } else if ch.is_numeric() {
                    self.number(start_column)
                } else {
                    panic!("Unexpected character: {}", ch);
                }
            }
        }
    }

    fn skip_whitespace(&mut self) {
        while !self.is_at_end() {
            match self.peek() {
                ' ' | '\r' | '\t' => {
                    self.advance();
                }
                '\n' => {
                    self.current_line += 1;
                    self.current_column = 0;
                    self.advance();
                }
                _ => break,
            }
        }
    }

    fn is_at_end(&self) -> bool {
        self.position >= self.input.len()
    }

    fn advance(&mut self) -> char {
        let ch = self.input[self.position];
        self.position += 1;
        self.current_column += 1;
        ch
    }

    fn peek(&self) -> char {
        if self.is_at_end() {
            '\0'
        } else {
            self.input[self.position]
        }
    }

    fn make_token(&self, token_type: TokenType, lexeme: &str) -> Token {
        Token::new(
            token_type,
            self.current_line,
            self.current_column - lexeme.len(),
            lexeme.to_string(),
        )
    }

    fn identifier(&mut self, start_column: usize) -> Token {
        let start = self.position - 1;
        while self.peek().is_alphanumeric() || self.peek() == '_' {
            self.advance();
        }

        let lexeme: String = self.input[start..self.position].iter().collect();
        let token_type = if is_keyword(&lexeme) {
            TokenType::Keyword(lexeme.clone())
        } else {
            TokenType::Identifier(lexeme.clone())
        };

        Token::new(token_type, self.current_line, start_column, lexeme)
    }

    fn number(&mut self, start_column: usize) -> Token {
        let start = self.position - 1;
        while self.peek().is_numeric() {
            self.advance();
        }

        if self.peek() == '.' && self.peek_next().is_numeric() {
            self.advance(); // consume '.'
            while self.peek().is_numeric() {
                self.advance();
            }
            let lexeme: String = self.input[start..self.position].iter().collect();
            let value = lexeme.parse::<f64>().unwrap();
            Token::new(
                TokenType::Float(value),
                self.current_line,
                start_column,
                lexeme,
            )
        } else {
            let lexeme: String = self.input[start..self.position].iter().collect();
            let value = lexeme.parse::<i64>().unwrap();
            Token::new(
                TokenType::Integer(value),
                self.current_line,
                start_column,
                lexeme,
            )
        }
    }

    fn peek_next(&self) -> char {
        if self.position + 1 >= self.input.len() {
            '\0'
        } else {
            self.input[self.position + 1]
        }
    }
}

fn is_keyword(word: &str) -> bool {
    matches!(
        word,
        "if" | "else" | "while" | "for" | "return" | "let" | "fn" | "struct" | "enum"
    )
}

fn main() {
    let input = "let x = 42 + 3.14;";
    let mut lexer = Lexer::new(input);

    println!("Tokenizing: {}", input);
    loop {
        let token = lexer.next_token();
        println!("{:?}", token);
        if matches!(token.token_type, TokenType::EOF) {
            break;
        }
    }
}
"""


if __name__ == "__main__":
    # Generate demo visualization
    generator = HeadspaceGenerator()
    data = generator.generate_from_rust_code(DEMO_RUST_CODE, "tokenizer.rs")
    output_path = generator.generate_headspace_html(data, output_path="rust_tokenizer_headspace.html")

    print(f"\n📊 Statistics:")
    print(f"  - Chunks: {len(generator.chunks)}")
    print(f"  - Clusters: {len(generator.clusters)}")
    print(f"  - Connections: {len(generator.connections)}")
    print(f"\n🚀 Open {output_path} in your browser to explore!")