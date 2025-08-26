"""
Enhanced Ollama Parser with Pydantic validation and dynamic prompting.

This module implements robust content analysis using Ollama LLMs with:
- Dynamic prompt generation from schema
- Pydantic validation for structured outputs  
- Few-shot learning for better extraction
- Graceful error handling and retries

Author: Globule Team
Version: 3.0.0 (Pydantic-enhanced)
"""

import json
import logging
import asyncio
import re
from typing import Dict, Any, Optional, List, Type
from datetime import datetime

import aiohttp
from pydantic import BaseModel, ValidationError as PydanticError, create_model, field_validator
from jsonschema import validate, ValidationError

from globule.core.interfaces import IParserProvider
from globule.config.settings import get_config
from globule.schemas.manager import get_schema_manager, format_schema_for_llm, get_schema_for_domain

logger = logging.getLogger(__name__)


class OllamaParser(IParserProvider):
    """
    Enhanced Ollama parser with dynamic prompting and Pydantic validation.
    
    Features:
    - Dynamic few-shot prompt generation from schema
    - Pydantic models for robust validation and defaults
    - Intelligent retry logic with error context
    - Clean JSON extraction from LLM responses
    """

    def __init__(self):
        """Initialize the enhanced Ollama parser."""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        self.schema_manager = get_schema_manager()
        self.max_retries = 2
        
        # Enhanced base prompt template with schema context
        self.base_parsing_prompt = """
You are a precise text analyzer. Parse the following text and return ONLY a valid JSON object with the extracted data according to the provided schema.

Text: {text}

Schema: {schema_info}

Return ONLY the JSON object:"""

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _ensure_session(self) -> None:
        """Ensure HTTP session is initialized."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.ollama_timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        """Clean up HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def health_check(self) -> bool:
        """Check if Ollama service is available."""
        try:
            await self._ensure_session()
            url = f"{self.config.ollama_base_url}/api/tags"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model["name"] for model in data.get("models", [])]
                    return self.config.default_parsing_model in models
                    
        except Exception as e:
            self.logger.warning(f"Ollama health check failed: {e}")
            
        return False

    async def parse(self, text: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse text using hybrid approach: regex fast-path + LLM with Pydantic validation.
        
        Args:
            text: Input text to analyze
            schema: Optional schema specification
            
        Returns:
            Dict containing structured parsing results
        """
        # Determine which schema to use - try auto-detection for better accuracy
        schema_name = self._determine_schema_name(schema)
        
        # Auto-detect schema if no specific schema requested or if using default
        if text.strip() and (not schema or schema_name == 'default'):
            detected_schema = self.schema_manager.detect_schema_for_text(text)
            if detected_schema and detected_schema != 'default':
                schema_name = detected_schema
        
        schema_dict = self.schema_manager.get_schema(schema_name) if schema_name else None
        
        # Apply schema actions for enrichment
        enriched_data = {}
        if schema_dict and text.strip():
            try:
                enrichment = self.schema_manager.apply_schema_actions(text, schema_name)
                enriched_data = enrichment.get('enriched', {})
            except Exception as e:
                self.logger.warning(f"Schema actions failed: {e}")
        
        # Try fast-path regex parsing first
        fast_result = self._fast_path_parse(text, schema_name, enriched_data)
        if fast_result:
            # Validate with Pydantic if schema available
            if schema_dict:
                try:
                    dynamic_model = self._create_dynamic_model(schema_dict)
                    validated = dynamic_model(**fast_result)
                    result = validated.model_dump()
                    result["metadata"] = {
                        "parser_type": "hybrid_regex",
                        "parser_version": "3.0.0", 
                        "schema_used": schema_name,
                        "confidence_score": fast_result.get("confidence_score", 0.9)
                    }
                    # Ensure confidence_score is in the main result too
                    if "confidence_score" not in result:
                        result["confidence_score"] = fast_result.get("confidence_score", 0.9)
                    return result
                except PydanticError:
                    # Fall through to LLM if validation fails
                    self.logger.debug("Fast-path validation failed, falling back to LLM")
            else:
                return fast_result
        
        # Health check before LLM parsing
        if not await self.health_check():
            return self._create_fallback_result(text, "Ollama unavailable")
        
        # Use fallback if no schema available  
        if not schema_dict:
            return await self._parse_without_schema(text)
        
        # Generate dynamic prompt with few-shots
        few_shots = self._generate_few_shots(schema_dict)
        prompt = self._build_dynamic_prompt(text, schema_dict, few_shots)
        
        # Attempt parsing with retries
        for attempt in range(1, self.max_retries + 1):
            try:
                await self._ensure_session()
                response = await self._call_ollama_api(prompt)
                cleaned = self._clean_llm_response(response)
                parsed = json.loads(cleaned)
                
                # Create dynamic Pydantic model and validate
                dynamic_model = self._create_dynamic_model(schema_dict)
                
                try:
                    validated = dynamic_model(**parsed)
                    
                    # Return with parser metadata
                    result = validated.model_dump()
                    result["metadata"] = {
                        "parser_type": "ollama_llm",
                        "parser_version": "3.0.0",
                        "schema_used": schema_name,
                        "attempts": attempt
                    }
                    
                    return result
                    
                except PydanticError as pydantic_err:
                    # Try to salvage the data by filtering out problematic fields
                    cleaned_data = self._clean_data_for_schema(parsed, schema_dict)
                    try:
                        validated = dynamic_model(**cleaned_data)
                        result = validated.model_dump()
                        result["metadata"] = {
                            "parser_type": "ollama_llm",
                            "parser_version": "3.0.0",
                            "schema_used": schema_name,
                            "attempts": attempt,
                            "pydantic_cleaned": True
                        }
                        return result
                    except PydanticError:
                        # Re-raise original error for retry logic
                        raise pydantic_err
                
            except (json.JSONDecodeError, PydanticError, ValidationError) as e:
                self.logger.warning(f"Parsing attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    # Enhanced retry prompt with error context
                    prompt = self._build_retry_prompt(text, schema_dict, str(e))
                else:
                    self.logger.error(f"Parsing failed after {self.max_retries} attempts: {e}")
                    return self._create_fallback_result(text, str(e), schema_name)
            except Exception as e:
                self.logger.error(f"Unexpected error during parsing: {e}")
                return self._create_fallback_result(text, str(e), schema_name)
        
        return self._create_fallback_result(text, "Max retries exceeded", schema_name)

    async def text_to_sql(self, question: str, db_schema: str) -> str:
        """
        Translates a natural language question into a SQL query.

        Args:
            question: The user's natural language question.
            db_schema: The schema of the database to query.

        Returns:
            A SQL query string.
        """
        prompt = self._build_text_to_sql_prompt(question, db_schema)
        
        try:
            await self._ensure_session()
            response = await self._call_ollama_api(prompt)
            sql_query = self._clean_sql_response(response)
            return sql_query
        except Exception as e:
            self.logger.error(f"Text-to-SQL conversion failed: {e}")
            raise

    def _build_text_to_sql_prompt(self, question: str, db_schema: str) -> str:
        """Builds the prompt for the text-to-SQL LLM call."""
        return f"""
As an expert SQL developer, your task is to translate the following natural language question into a valid SQLite query. You must only respond with the raw SQL query, with no explanations, comments, or markdown.

Database Schema:
```sql
{db_schema}
```

The `parsed_data` column is a JSON object. You can query it using `json_extract`.
For example, to find globules with the category 'incident', you would use:
`WHERE json_extract(parsed_data, '$.category') = 'incident'`

Natural Language Question:
`{question}`

SQL Query:"""

    def _clean_sql_response(self, response: str) -> str:
        """Cleans the LLM response to extract a valid SQL query."""
        response = response.strip()
        # Remove markdown code blocks
        if response.startswith("```sql"):
            response = response[6:].rstrip("```").strip()
        elif response.startswith("```"):
            response = response[3:].rstrip("```").strip()

        # Find the start of the SELECT statement
        select_pos = response.upper().find("SELECT")
        if select_pos != -1:
            response = response[select_pos:]

        # Remove trailing semicolons or other junk
        response = response.split(';')[0].strip()

        return response

    async def _parse_without_schema(self, text: str) -> Dict[str, Any]:
        """Parse text without schema using basic prompt."""
        prompt = self.base_parsing_prompt.format(text=text)
        
        try:
            await self._ensure_session()
            response = await self._call_ollama_api(prompt)
            cleaned = self._clean_llm_response(response)
            parsed = json.loads(cleaned)
            
            # Add basic metadata
            parsed["metadata"] = {
                "parser_type": "ollama_llm",
                "parser_version": "3.0.0",
                "schema_used": "none"
            }
            
            return parsed
            
        except Exception as e:
            self.logger.error(f"Schema-less parsing failed: {e}")
            return self._create_fallback_result(text, str(e))

    def _determine_schema_name(self, schema: Optional[Dict[str, Any]]) -> Optional[str]:
        """Determine which schema to use based on the schema parameter."""
        if schema is None:
            return self.config.default_schema
        
        # If schema has a 'name' key, use that
        if isinstance(schema, dict) and 'name' in schema:
            return schema['name']
        
        # If schema has a 'domain' key, auto-select based on domain
        if isinstance(schema, dict) and 'domain' in schema:
            return get_schema_for_domain(schema['domain'])
        
        return self.config.default_schema

    def _build_dynamic_prompt(self, text: str, schema: Dict[str, Any], few_shots: List[str]) -> str:
        """Build dynamic prompt with schema rules and few-shot examples."""
        # Generate field-specific rules
        rules = []
        for field, props in schema.get('properties', {}).items():
            rule = f"- {field}: {props.get('description', 'No description')} (Type: {props.get('type', 'any')})"
            
            if field in schema.get('required', []):
                rule += " [REQUIRED - infer if not explicit]"
            else:
                rule += " [OPTIONAL - use null/empty if not present]"
                
            # Add enum constraints
            if 'enum' in props:
                rule += f" [Options: {', '.join(props['enum'])}]"
                
            rules.append(rule)
        
        rules_str = "\n".join(rules)
        shots_str = "\n\n".join(few_shots)
        
        return f"""
You are a precise JSON extractor. Return ONLY a valid JSON object conforming exactly to this schema.

SCHEMA: {schema.get('title', 'Unknown Schema')}
{schema.get('description', '')}

FIELD RULES:
{rules_str}

CRITICAL INSTRUCTIONS:
- Return ONLY valid JSON, no explanations
- For required fields: infer reasonable values if not explicit
- For optional fields: use null or empty arrays if not present  
- For confidence_score: use 0.85-0.95 based on text clarity
- For sentiment: use "neutral" if not clearly positive/negative
- For arrays: use empty [] if no relevant data found

EXAMPLES:
{shots_str}

TEXT TO PARSE:
{text}

JSON OUTPUT:"""

    def _build_retry_prompt(self, text: str, schema: Dict[str, Any], error: str) -> str:
        """Build retry prompt with specific error context."""
        return f"""
RETRY: Your previous JSON was invalid. Fix the error and return ONLY valid JSON.

ERROR: {error}

SCHEMA REQUIREMENTS:
- All required fields must be present: {', '.join(schema.get('required', []))}
- Use exact field names from schema
- Follow type constraints (string, array, etc.)
- No additional fields beyond schema

TEXT TO PARSE:
{text}

CORRECTED JSON:"""

    def _generate_few_shots(self, schema: Dict[str, Any]) -> List[str]:
        """Generate synthetic few-shot examples based on schema."""
        examples = []
        
        # Create base example with reasonable defaults
        base_example = {}
        for field, props in schema.get('properties', {}).items():
            field_type = props.get('type', 'string')
            
            if 'enum' in props:
                base_example[field] = props['enum'][0]
            elif field_type == 'string':
                if 'name' in field.lower():
                    base_example[field] = "John Smith"
                elif 'vehicle' in field.lower():
                    base_example[field] = "Toyota" if 'make' in field else "Camry"
                elif 'license' in field.lower():
                    base_example[field] = "ABC-123"
                elif 'spot' in field.lower():
                    base_example[field] = "A1"
                else:
                    base_example[field] = f"Sample {field}"
            elif field_type == 'array':
                if 'damage' in field.lower():
                    base_example[field] = ["small scratch on door"]
                elif 'keyword' in field.lower():
                    base_example[field] = ["parking", "vehicle"]
                else:
                    base_example[field] = ["sample item"]
            elif field_type == 'number':
                base_example[field] = 0.9 if 'confidence' in field else 1
            elif field_type == 'integer':
                base_example[field] = int(datetime.now().timestamp()) if 'timestamp' in field else 1
            elif field_type == ['integer', 'null']:
                base_example[field] = None
        
        # Example 1: Complete entry with damage
        examples.append(f"Input: John parked a Toyota Camry with plate ABC-123 in spot A1. Small dent noted.\nOutput: {json.dumps(base_example)}")
        
        # Example 2: Clean entry without damage  
        clean_example = base_example.copy()
        clean_example['damage_notes'] = []
        if 'timestamp' in clean_example:
            clean_example['timestamp'] = None
        examples.append(f"Input: Alice parked Honda Civic, plate XYZ-789 in B2. No issues.\nOutput: {json.dumps(clean_example)}")
        
        return examples

    def _fast_path_parse(self, text: str, schema_name: Optional[str], enriched_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Fast regex-based parsing for common patterns.
        
        Args:
            text: Input text to parse
            schema_name: Schema name if available
            enriched_data: Pre-enriched data from schema actions
            
        Returns:
            Parsed result dict if successful, None if no patterns match
        """
        if not text.strip():
            return None
        
        result = {}
        confidence = 0.0
        patterns_matched = 0
        
        # Use enriched data if available (from schema actions)
        if enriched_data:
            result.update(enriched_data)
            confidence = 0.8  # High confidence from schema actions
            patterns_matched += len([k for k, v in enriched_data.items() if v and k != 'text'])
        
        # Schema-specific fast paths
        if schema_name == 'valet' or schema_name == 'valet_enhanced':
            # License plate patterns
            plate_patterns = [
                r'(?:plate|license)\s+([A-Z0-9\-]{3,10})',
                r'([A-Z]{1,3}[-\s]?\d{3,4})',
                r'(\d{3}[-\s]?[A-Z]{3})',
            ]
            
            for pattern in plate_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match and 'license_plate' not in result:
                    result['license_plate'] = match.group(1).upper()
                    confidence += 0.3
                    patterns_matched += 1
                    break
            
            # Parking spot patterns  
            spot_patterns = [
                r'(?:spot|space|bay)\s+([A-Z0-9]+)',
                r'(?:level|floor)\s+(\d+)[A-Z]?[-\s]?([A-Z0-9]+)',
            ]
            
            for pattern in spot_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match and 'parking_spot' not in result:
                    result['parking_spot'] = match.group(1).upper()
                    confidence += 0.2
                    patterns_matched += 1
                    break
            
            # Damage keywords
            damage_patterns = r'\b(scratch|dent|damage|broken|cracked|chipped)\b'
            damage_matches = re.findall(damage_patterns, text, re.IGNORECASE)
            if damage_matches and 'damage_notes' not in result:
                result['damage_notes'] = list(set(damage_matches))
                confidence += 0.1
                patterns_matched += 1
        
        # General patterns for any schema
        
        # URL pattern
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, text)
        if urls:
            result['urls'] = urls
            confidence += 0.1
            patterns_matched += 1
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            result['emails'] = emails
            confidence += 0.1
            patterns_matched += 1
        
        # Question detection
        if text.strip().endswith('?'):
            result['category'] = 'question'
            confidence += 0.1
            patterns_matched += 1
        
        # Code detection (triple backticks or function patterns)
        if '```' in text or re.search(r'\b(def|function|class|import)\s+\w+', text):
            result['content_type'] = 'code'
            result['category'] = 'reference'
            confidence += 0.2
            patterns_matched += 1
        
        # Only return if we found meaningful patterns
        if patterns_matched >= 1 and confidence > 0.3:
            # Add metadata
            result.update({
                'confidence_score': min(confidence, 1.0),
                'title': text[:50] + ('...' if len(text) > 50 else ''),
                'category': result.get('category', 'note'),
                'domain': result.get('domain', 'general'),
                'keywords': result.get('keywords', []),
                'entities': result.get('entities', []),
                'sentiment': 'neutral',
                'content_type': result.get('content_type', 'prose')
            })
            return result
        
        return None

    def _create_dynamic_model(self, schema: Dict[str, Any]) -> Type[BaseModel]:
        """Create dynamic Pydantic model from JSON schema."""
        fields = {}
        
        for field, props in schema.get('properties', {}).items():
            field_type = props.get('type', 'string')
            is_required = field in schema.get('required', [])
            
            # Map JSON schema types to Python types
            if field_type == 'string':
                python_type = str
                default = ... if is_required else None
            elif field_type == 'array':
                python_type = List[str]
                default = ... if is_required else []
            elif field_type == 'number':
                python_type = float
                default = ... if is_required else 0.85
            elif field_type == 'integer':
                python_type = int
                default = ... if is_required else None
            elif field_type == ['integer', 'null']:
                python_type = Optional[int]
                default = None
            else:
                python_type = Any
                default = ... if is_required else None
            
            fields[field] = (python_type, default)
        
        return create_model('DynamicSchemaModel', **fields)

    def _clean_data_for_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Clean data to match schema requirements, removing invalid fields and fixing types."""
        cleaned = {}
        properties = schema.get('properties', {})
        
        for field, props in properties.items():
            if field not in data:
                continue
                
            value = data[field]
            field_type = props.get('type', 'string')
            
            try:
                # Type conversion attempts
                if field_type == 'string' and not isinstance(value, str):
                    cleaned[field] = str(value)
                elif field_type == 'number' and not isinstance(value, (int, float)):
                    # Try to convert strings to float
                    if isinstance(value, str) and value.isdigit():
                        cleaned[field] = float(value)
                    else:
                        # Use default for invalid numeric values
                        cleaned[field] = 0.85 if 'confidence' in field else 0.0
                elif field_type == 'integer' and not isinstance(value, int):
                    if isinstance(value, str) and value.isdigit():
                        cleaned[field] = int(value)
                    else:
                        cleaned[field] = None
                elif field_type == 'array' and not isinstance(value, list):
                    cleaned[field] = []
                else:
                    cleaned[field] = value
                    
            except (ValueError, TypeError):
                # Use schema defaults or safe fallbacks
                if field_type == 'string':
                    cleaned[field] = ""
                elif field_type == 'array':
                    cleaned[field] = []
                elif field_type == 'number':
                    cleaned[field] = 0.85 if 'confidence' in field else 0.0
                elif field_type == 'integer':
                    cleaned[field] = None
                else:
                    cleaned[field] = value
        
        return cleaned

    async def _call_ollama_api(self, prompt: str) -> str:
        """Make Ollama API call with enhanced configuration."""
        payload = {
            "model": self.config.default_parsing_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Slightly higher for creativity while maintaining structure
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 1000,  # Ensure sufficient space for complete JSON
            }
        }
        
        url = f"{self.config.ollama_base_url}/api/generate"
        
        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                raise Exception(f"Ollama request failed with status {response.status}")
                
            data = await response.json()
            raw_response = data.get("response", "").strip()
            
            return raw_response

    def _clean_llm_response(self, response: str) -> str:
        """Clean LLM response to extract valid JSON with minimal processing."""
        response = response.strip()
        
        # Remove markdown code blocks
        if response.startswith("```json"):
            response = response[7:].rstrip("```").strip()
        elif response.startswith("```"):
            response = response[3:].rstrip("```").strip()
        
        # Remove common prefixes
        prefixes = ["Here is the JSON:", "JSON:", "Output:", "Result:"]
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
                break
        
        # Extract JSON object boundaries
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            response = response[start_idx:end_idx + 1]
        
        return response.strip()

    def _create_fallback_result(self, text: str, error_message: Optional[str] = None, schema_name: Optional[str] = None) -> Dict[str, Any]:
        """Create safe fallback result when parsing fails."""
        return {
            "title": text[:50] + "..." if len(text) > 50 else text,
            "category": "note",
            "domain": "general",
            "keywords": [],
            "entities": [],
            "error": error_message or "Parsing failed",
            "metadata": {
                "parser_type": "fatal_fallback",
                "parser_version": "3.0.0",
                "schema_used": schema_name,
                "error_details": error_message
            }
        }