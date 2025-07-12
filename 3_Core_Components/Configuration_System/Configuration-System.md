\# Configuration System - Low Level Design

\*Version: 1.0\*  

\*Date: 2025-07-11\*  

\*Status: Draft for Review\*



\## 1. Introduction



This document provides the detailed low-level design for Globule's Configuration System, translating the high-level requirements into concrete technical specifications. It builds upon the extensive research conducted and documented in "Low-Level Design of Globule's Configuration System.txt".



\### 1.1 Scope



This LLD covers:

\- Storage format and file locations

\- Loading and validation strategies

\- Configuration cascade implementation

\- User-facing API design

\- Performance optimizations

\- Error handling and recovery



\### 1.2 Dependencies from HLD



From the High Level Design document:

\- Three-tier cascade model (System → User → Context)

\- Support for hot-reloading in development

\- Schema validation and type safety

\- User empowerment through progressive configuration



\## 2. Storage Format Decision



\### 2.1 Selected Format: YAML with ruamel.yaml



\*\*Decision\*\*: YAML is selected as the primary configuration format.



\*\*Rationale\*\*:

\- Superior human readability for complex nested structures

\- Native comment support with preservation via ruamel.yaml

\- Matches user expectations from similar tools (Kubernetes, CI/CD)

\- Handles deeply nested structures required for schemas and LLM prompts



\*\*Security Mitigation\*\*:

\- Mandatory use of `yaml.safe\_load()` for all parsing operations

\- No direct execution of YAML content

\- Strict Pydantic validation post-parsing



\### 2.2 File Locations (XDG Compliance)



Configuration files follow the XDG Base Directory Specification:



```

System defaults:  /etc/globule/config.yaml (read-only)

User preferences: $XDG\_CONFIG\_HOME/globule/config.yaml 

&nbsp;                (defaults to ~/.config/globule/config.yaml)

Context configs:  $XDG\_CONFIG\_HOME/globule/contexts/\*.yaml

Schema definitions: $XDG\_CONFIG\_HOME/globule/schemas/\*.yaml

```



\### 2.3 File Structure Example



```yaml

\# System defaults (/etc/globule/config.yaml)

version: "1.0"

system:

&nbsp; processing\_transparency: "concise"

&nbsp; file\_organization: "semantic"

&nbsp; ai\_models:

&nbsp;   embedding: "mxbai-embed-large"

&nbsp;   parsing: "llama3.2:3b"

&nbsp; hot\_reload:

&nbsp;   enabled: false  # Production default

&nbsp;   

\# User preferences (~/.config/globule/config.yaml)  

version: "1.0"

user:

&nbsp; processing\_transparency: "verbose"

&nbsp; theme: "dark"

&nbsp; synthesis:

&nbsp;   default\_cluster\_view: true

&nbsp;   ai\_suggestions\_aggression: "moderate"

&nbsp;   

\# Context override (~/.config/globule/contexts/creative\_writing.yaml)

version: "1.0"

context:

&nbsp; name: "creative\_writing"

&nbsp; parent: null  # No inheritance for MVP

&nbsp; overrides:

&nbsp;   processing\_transparency: "verbose"

&nbsp;   synthesis:

&nbsp;     ai\_suggestions\_aggression: "proactive"

&nbsp;     show\_semantic\_connections: true

```



\## 3. Configuration Loading Strategy



\### 3.1 Load Sequence



1\. \*\*System defaults\*\* loaded from package-bundled YAML

2\. \*\*User preferences\*\* loaded from XDG config directory

3\. \*\*Active context\*\* (if any) loaded from contexts directory

4\. \*\*Environment variables\*\* processed via Pydantic BaseSettings

5\. \*\*Merge cascade\*\* applied in precedence order

6\. \*\*Validation\*\* against GlobalConfig Pydantic model



\### 3.2 Hot-Reload Mechanism



\*\*Development Mode\*\* (`hot\_reload.enabled: true`):

\- Uses watchdog to monitor configuration files

\- Detection latency: 500-700ms

\- Triggers full application restart for consistency

\- All reload events logged with timestamp



\*\*Production Mode\*\* (`hot\_reload.enabled: false`):

\- No file watching

\- Configuration changes require explicit restart

\- Ensures stability and predictable behavior



\### 3.3 Error Handling



```python

class ConfigurationError(Exception):

&nbsp;   """Base exception for configuration-related errors"""

&nbsp;   pass



class ConfigurationLoadError(ConfigurationError):

&nbsp;   """Raised when configuration files cannot be loaded"""

&nbsp;   pass

&nbsp;   

class ConfigurationValidationError(ConfigurationError):

&nbsp;   """Raised when configuration fails Pydantic validation"""

&nbsp;   pass

```



\*\*Recovery Strategy\*\*:

1\. Log detailed error with file location and line number

2\. Fall back to Last Known Good (LKG) configuration

3\. If no LKG exists, use system defaults only

4\. Notify user via CLI of degraded configuration



\## 4. Configuration Cascade Implementation



\### 4.1 Cascade Resolution Order



```

1\. Environment variables (highest precedence)

2\. Active context configuration  

3\. User preferences

4\. System defaults (lowest precedence)

```



\### 4.2 Nested Key Access



Keys use dot notation for hierarchical access:

\- `synthesis.ai\_suggestions.aggression`

\- `ai\_models.embedding`



Implementation uses recursive dictionary traversal with graceful handling of missing intermediate keys.



\### 4.3 Context Inheritance (Post-MVP)



For MVP, contexts are independent. Future enhancement will support one-level inheritance:

```yaml

context:

&nbsp; name: "creative\_writing.novel"

&nbsp; parent: "creative\_writing"

```



\## 5. Schema Definition



\### 5.1 Core Pydantic Models



```python

from pydantic import BaseModel, Field, validator

from pydantic\_settings import BaseSettings

from typing import Literal, Optional, Dict, Any

from pathlib import Path



class AIModelsConfig(BaseModel):

&nbsp;   """AI model configuration"""

&nbsp;   embedding: str = Field(

&nbsp;       default="mxbai-embed-large",

&nbsp;       description="Model for semantic embeddings"

&nbsp;   )

&nbsp;   parsing: str = Field(

&nbsp;       default="llama3.2:3b", 

&nbsp;       description="Model for structural parsing"

&nbsp;   )

&nbsp;   

class SynthesisConfig(BaseModel):

&nbsp;   """Interactive synthesis engine configuration"""

&nbsp;   default\_cluster\_view: bool = True

&nbsp;   ai\_suggestions\_aggression: Literal\["passive", "moderate", "proactive"] = "moderate"

&nbsp;   show\_semantic\_connections: bool = True

&nbsp;   progressive\_discovery\_depth: int = Field(default=2, ge=1, le=5)

&nbsp;   

class HotReloadConfig(BaseModel):

&nbsp;   """Hot reload configuration"""

&nbsp;   enabled: bool = False

&nbsp;   watch\_delay: float = Field(default=0.5, ge=0.1, le=5.0)

&nbsp;   

class GlobalConfig(BaseSettings):

&nbsp;   """Root configuration model with environment variable support"""

&nbsp;   version: str = Field(default="1.0", const=True)

&nbsp;   processing\_transparency: Literal\["silent", "concise", "verbose"] = "concise"

&nbsp;   file\_organization: Literal\["semantic", "chronological", "hybrid"] = "semantic"

&nbsp;   theme: Literal\["light", "dark", "auto"] = "auto"

&nbsp;   

&nbsp;   ai\_models: AIModelsConfig = Field(default\_factory=AIModelsConfig)

&nbsp;   synthesis: SynthesisConfig = Field(default\_factory=SynthesisConfig)

&nbsp;   hot\_reload: HotReloadConfig = Field(default\_factory=HotReloadConfig)

&nbsp;   

&nbsp;   class Config:

&nbsp;       env\_prefix = "GLOBULE\_"

&nbsp;       env\_nested\_delimiter = "\_\_"

&nbsp;       env\_file = ".env"

&nbsp;       env\_file\_encoding = "utf-8"

```



\### 5.2 Validation Rules



\- Version field ensures configuration compatibility

\- Literal types enforce valid enum values

\- Nested models provide structured validation

\- Field constraints (e.g., `ge=1, le=5`) ensure reasonable values



\### 5.3 Environment Variable Mapping



```bash

\# Maps to processing\_transparency

export GLOBULE\_PROCESSING\_TRANSPARENCY=verbose



\# Maps to ai\_models.embedding  

export GLOBULE\_AI\_MODELS\_\_EMBEDDING=all-MiniLM-L6-v2



\# Maps to synthesis.ai\_suggestions\_aggression

export GLOBULE\_SYNTHESIS\_\_AI\_SUGGESTIONS\_AGGRESSION=proactive

```



\## 6. User-Facing API



\### 6.1 CLI Commands



```bash

\# Get configuration value

globule config get <key>

globule config get synthesis.ai\_suggestions\_aggression



\# Set configuration value (saves to user config)

globule config set <key> <value>

globule config set theme dark



\# Set context-specific value

globule config set --context creative\_writing synthesis.ai\_suggestions\_aggression proactive



\# List all configuration

globule config list \[--show-source]



\# Open configuration in editor

globule config edit \[--context <name>]



\# Validate configuration

globule config validate \[--file <path>]



\# Show active configuration cascade

globule config cascade

```



\### 6.2 Programmatic API



```python

class ConfigManager:

&nbsp;   """Main configuration management interface"""

&nbsp;   

&nbsp;   def \_\_init\_\_(self, 

&nbsp;                system\_config\_path: Optional\[Path] = None,

&nbsp;                user\_config\_path: Optional\[Path] = None,

&nbsp;                context\_configs\_dir: Optional\[Path] = None):

&nbsp;       """Initialize with optional custom paths"""

&nbsp;       

&nbsp;   def get(self, key: str, context: Optional\[str] = None) -> Any:

&nbsp;       """Get configuration value with cascade resolution"""

&nbsp;       

&nbsp;   def set(self, key: str, value: Any, target: str = "user") -> None:

&nbsp;       """Set configuration value in specified target"""

&nbsp;       

&nbsp;   def validate(self) -> GlobalConfig:

&nbsp;       """Validate and return typed configuration object"""

&nbsp;       

&nbsp;   def reload(self) -> None:

&nbsp;       """Manually trigger configuration reload"""

&nbsp;       

&nbsp;   def export(self, format: str = "yaml") -> str:

&nbsp;       """Export effective configuration"""

```



\### 6.3 Schema Management API



```bash

\# Create new schema from template

globule schema create <name> --template <type>



\# Validate schema file

globule schema validate <file>



\# List available schemas

globule schema list



\# Edit schema

globule schema edit <name>

```



\## 7. Performance Specifications



\### 7.1 Caching Strategy



\- In-memory cache using Python dictionaries

\- Cache invalidation on file change detection

\- Lazy loading for context configurations



\### 7.2 Performance Targets



| Operation | Target Latency | Notes |

|-----------|---------------|-------|

| Config key access (cached) | <1μs | Direct memory access |

| Config key access (miss) | <50ms | File parse + validation |

| Initial load | <200ms | All configs + validation |

| Hot reload detection | 500-700ms | Watchdog file system events |

| CLI get command | <100ms | Including process startup |

| CLI set command | <150ms | Including file write + validation |



\### 7.3 Schema Compilation (Future)



Post-MVP optimization using Pydantic's `create\_model()` to dynamically generate models from user schemas.



\## 8. Error Messages and Logging



\### 8.1 User-Facing Error Messages



```yaml

\# Validation error

Error: Invalid configuration value

&nbsp; File: ~/.config/globule/config.yaml

&nbsp; Line: 15

&nbsp; Field: synthesis.ai\_suggestions\_aggression

&nbsp; Value: "aggressive"

&nbsp; Valid options: passive, moderate, proactive



\# File not found

Error: Configuration file not found

&nbsp; Expected: ~/.config/globule/contexts/work.yaml

&nbsp; Suggestion: Run 'globule config create-context work'



\# Malformed YAML

Error: Invalid YAML syntax

&nbsp; File: ~/.config/globule/config.yaml

&nbsp; Line: 22

&nbsp; Issue: Inconsistent indentation (expected 2 spaces, found 3)

```



\### 8.2 Logging Format



```python

\# Configuration reload

2025-07-11 10:23:45 INFO \[config] Configuration reloaded from ~/.config/globule/config.yaml

2025-07-11 10:23:45 INFO \[config] Active context: creative\_writing

2025-07-11 10:23:45 DEBUG \[config] Cascade: env(2) > creative\_writing(5) > user(3) > system(10)



\# Validation failure with fallback

2025-07-11 10:24:12 ERROR \[config] Validation failed for user config, using LKG

2025-07-11 10:24:12 ERROR \[config] Details: synthesis.progressive\_discovery\_depth = 10 (max: 5)

```



\## 9. Thread Safety



\### 9.1 Concurrent Access



\- ConfigManager instances are thread-safe for read operations

\- Write operations acquire a threading.Lock

\- Cache updates are atomic



\### 9.2 Implementation



```python

import threading



class ConfigManager:

&nbsp;   def \_\_init\_\_(self):

&nbsp;       self.\_lock = threading.RLock()

&nbsp;       self.\_cache = {}

&nbsp;       

&nbsp;   def get(self, key: str) -> Any:

&nbsp;       # Read-only, no lock needed if cache is populated

&nbsp;       return self.\_get\_from\_cache(key)

&nbsp;       

&nbsp;   def set(self, key: str, value: Any) -> None:

&nbsp;       with self.\_lock:

&nbsp;           # Validate, update file, invalidate cache

&nbsp;           self.\_update\_config(key, value)

```



\## 10. Migration and Compatibility



\### 10.1 Version Management



\- Configuration files include version field

\- System validates version compatibility on load

\- Future versions will include migration tools



\### 10.2 Backward Compatibility



\- New optional fields have defaults

\- Deprecated fields logged but still functional

\- Major version changes require explicit migration



\## 11. Testing Requirements



\### 11.1 Unit Tests



\- Cascade resolution with all precedence combinations

\- YAML parsing with malformed inputs

\- Pydantic validation edge cases

\- Environment variable parsing and type conversion

\- Thread safety under concurrent access



\### 11.2 Integration Tests



\- Hot reload with actual file changes

\- CLI commands with various inputs

\- Schema validation pipeline

\- Error recovery scenarios



\### 11.3 Performance Tests



\- Cache hit rate measurement

\- Load time with various config sizes

\- Memory usage profiling

\- Concurrent access stress testing



\## 12. Security Considerations



\### 12.1 Input Validation



\- All YAML parsing uses safe\_load

\- File paths sanitized to prevent directory traversal

\- Environment variables filtered by prefix



\### 12.2 Access Control



\- System config files require elevated permissions to modify

\- User cannot modify system defaults via CLI

\- Context files isolated to user directory



\## 13. Future Enhancements (Post-MVP)



1\. \*\*Context Inheritance\*\*: One-level parent-child relationships

2\. \*\*Schema Compiler\*\*: Dynamic Pydantic model generation

3\. \*\*Configuration Server\*\*: For distributed deployments

4\. \*\*Encryption\*\*: For sensitive configuration values

5\. \*\*Audit Trail\*\*: Detailed change history with rollback



\## 14. Decision Log



| Decision | Rationale | Date |

|----------|-----------|------|

| YAML over TOML | Better for deeply nested schemas, comment preservation | 2025-07-11 |

| Full restart for hot-reload | Reliability over speed in development | 2025-07-11 |

| No deep context inheritance | Simplicity and predictability for MVP | 2025-07-11 |

| Pydantic for validation | Type safety and IDE support | 2025-07-11 |

| XDG compliance | Standard locations, user expectations | 2025-07-11 |



\## 15. Open Questions for Review



1\. Should we support YAML anchors/aliases for configuration reuse?

2\. Do we need a `globule config diff` command to compare configurations?

3\. Should context activation be explicit or derived from current directory?

4\. Is the 500-700ms hot-reload latency acceptable for development?

5\. Should we implement partial config exports (e.g., only synthesis settings)?



---



\*This LLD is ready for review. Once approved, it will serve as the definitive specification for implementing the Configuration System.\*

