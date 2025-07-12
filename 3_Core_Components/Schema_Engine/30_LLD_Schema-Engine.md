\# Schema Definition Engine - Low Level Design

\*Version: 1.0\*  

\*Date: 2025-07-11\*  

\*Status: Draft for Review\*



\## 1. Introduction



This document provides the detailed low-level design for Globule's Schema Definition Engine, building upon the comprehensive research and architectural analysis conducted. The Schema Engine enables users to define custom workflows and data structures, transforming Globule from a simple note-taking tool into a personalized knowledge processing system.



\### 1.1 Scope



This LLD covers:

\- Schema format specification and transpilation architecture

\- Dynamic model generation and caching strategy

\- Schema storage, organization, and lifecycle management

\- Security architecture for custom validators

\- Integration interfaces with other Globule components

\- User-facing tools and APIs



\### 1.2 Dependencies from HLD



From the High Level Design document:

\- User empowerment through editable, shareable configurations

\- Schema-aware input processing from the beginning

\- Support for complex workflows (e.g., valet\_daily example)

\- Local-first architecture with hot-reload capabilities



\## 2. Core Architecture: The Transpiler Pattern



\### 2.1 Architectural Overview



The Schema Definition Engine implements a \*\*transpiler architecture\*\* that bridges user-friendly YAML definitions with powerful runtime validation:



```

User YAML → JSON Schema (IR) → Pydantic Model → Cached Validator

```



This design separates concerns:

\- \*\*User Experience\*\*: Simple, documented YAML with comments

\- \*\*Portability\*\*: JSON Schema as intermediate representation

\- \*\*Performance\*\*: Compiled Pydantic models for validation

\- \*\*Security\*\*: Sandboxed execution for custom code



\### 2.2 Technology Stack



| Component | Technology | Rationale |

|-----------|-----------|-----------|

| User Format | YAML | Human-readable, supports comments, familiar to users |

| Parser | ruamel.yaml | Preserves comments and formatting |

| Intermediate Rep | JSON Schema | Industry standard, portable |

| Validation Engine | Pydantic | Fast, Pythonic, type-safe |

| Sandboxing | CodeJail + AppArmor | Process-level isolation for security |

| Caching | LRU Memory Cache | Essential for performance |



\## 3. Schema Format Specification



\### 3.1 YAML Schema Structure



```yaml

\# Schema metadata

$id: "https://globule.app/schemas/link\_curation/v1"

name: "LinkCuration"

version: "1.0"  # Reserved for future use

description: "Schema for saving and organizing web links"



\# Detection triggers

triggers:

&nbsp; contains: \["http://", "https://", "www."]

&nbsp; starts\_with: \["bookmark:", "link:"]

&nbsp; 

\# Field definitions  

fields:

&nbsp; url:

&nbsp;   type: string

&nbsp;   required: true

&nbsp;   pattern: "^https?://"

&nbsp;   description: "The URL to save"

&nbsp;   

&nbsp; title:

&nbsp;   type: string

&nbsp;   required: false

&nbsp;   default: null

&nbsp;   description: "Page title (auto-fetched if not provided)"

&nbsp;   

&nbsp; user\_context:

&nbsp;   type: string

&nbsp;   required: false

&nbsp;   prompt: "Why save this link?"

&nbsp;   description: "User's reason for saving"

&nbsp;   

&nbsp; tags:

&nbsp;   type: array

&nbsp;   items: string

&nbsp;   default: \[]

&nbsp;   description: "Categorization tags"



\# Processing actions

actions:

&nbsp; - fetch\_title:

&nbsp;     when: "title is null"

&nbsp;     from: "url"

&nbsp; - extract\_metadata:

&nbsp;     from: "url"

&nbsp;     fields: \["description", "image"]

&nbsp;     

\# Validation rules

rules:

&nbsp; - name: "url\_accessible"

&nbsp;   validator: "validators.web.check\_url\_accessible"

&nbsp;   external\_access: "network"

&nbsp;   

\# Output configuration

output:

&nbsp; format: "markdown"

&nbsp; template: |

&nbsp;   # {title}

&nbsp;   

&nbsp;   {user\_context}

&nbsp;   

&nbsp;   URL: {url}

&nbsp;   Tags: {tags}

&nbsp;   

&nbsp;   ---

&nbsp;   \*Saved on {timestamp}\*

```



\### 3.2 Schema Inheritance



```yaml

\# Base schema (schemas/base/timestamped.yml)

name: "Timestamped"

abstract: true

fields:

&nbsp; created\_at:

&nbsp;   type: datetime

&nbsp;   auto\_now\_add: true

&nbsp; updated\_at:

&nbsp;   type: datetime

&nbsp;   auto\_now: true



\# Derived schema

name: "BlogPost"

extends: "base/timestamped"

fields:

&nbsp; title: {type: string, required: true}

&nbsp; content: {type: string, required: true}

```



\### 3.3 Composition Patterns



```yaml

name: "CompleteNote"

allOf:

&nbsp; - "base/timestamped"

&nbsp; - "base/taggable"

&nbsp; - "base/locatable"

fields:

&nbsp; content: {type: string, required: true}

```



\## 4. Storage Organization



\### 4.1 Directory Structure



```

~/.globule/schemas/

├── built-in/              # Ships with Globule (read-only)

│   ├── free\_text.yml

│   ├── link\_curation.yml

│   ├── task\_entry.yml

│   └── meeting\_notes.yml

├── user/                  # User-defined schemas

│   ├── personal/

│   │   ├── journal\_entry.yml

│   │   └── dream\_log.yml

│   ├── work/

│   │   ├── standup\_notes.yml

│   │   └── bug\_report.yml

│   └── creative/

│       ├── story\_idea.yml

│       └── character\_sheet.yml

└── shared/               # Downloaded from community

&nbsp;   └── awesome-schemas/

&nbsp;       └── recipe\_card.yml

```



\### 4.2 Naming Conventions



\*\*File Names\*\*:

\- Use snake\_case: `meeting\_notes.yml`, not `MeetingNotes.yml`

\- Be descriptive but concise

\- Avoid version numbers in filenames (use internal version field)



\*\*Schema Identifiers\*\*:

\- Derived from path: `user.work.standup\_notes`

\- Dots represent directory separators

\- Built-in schemas have no prefix: `link\_curation`



\*\*Namespace Resolution\*\*:

1\. Check user schemas first (allows overriding built-ins)

2\. Then check shared schemas

3\. Finally check built-in schemas



\### 4.3 Configuration Alignment



The storage strategy aligns with the main note storage configuration:



```yaml

\# In globule config

schemas:

&nbsp; organization: "hierarchical"  # or "flat", "tag-based"

&nbsp; naming\_style: "snake\_case"    # or "kebab-case", "camelCase"

&nbsp; 

\# These settings apply to both schemas and notes

storage:

&nbsp; prefer\_semantic\_paths: true

&nbsp; max\_path\_depth: 3

```



\## 5. Transpilation Pipeline



\### 5.1 Pipeline Stages



```python

class SchemaTranspiler:

&nbsp;   """Converts user YAML to executable Pydantic models"""

&nbsp;   

&nbsp;   def transpile(self, yaml\_path: Path) -> Type\[BaseModel]:

&nbsp;       # Stage 1: Parse YAML

&nbsp;       with open(yaml\_path) as f:

&nbsp;           schema\_dict = ruamel.yaml.safe\_load(f)

&nbsp;           

&nbsp;       # Stage 2: Validate meta-schema

&nbsp;       self.\_validate\_schema\_structure(schema\_dict)

&nbsp;       

&nbsp;       # Stage 3: Generate JSON Schema IR

&nbsp;       json\_schema = self.\_to\_json\_schema(schema\_dict)

&nbsp;       

&nbsp;       # Stage 4: Resolve inheritance/composition

&nbsp;       if 'extends' in schema\_dict:

&nbsp;           base\_model = self.\_load\_base\_model(schema\_dict\['extends'])

&nbsp;           json\_schema = self.\_merge\_schemas(base\_model, json\_schema)

&nbsp;           

&nbsp;       # Stage 5: Generate Pydantic model

&nbsp;       model\_class = self.\_create\_pydantic\_model(

&nbsp;           name=schema\_dict\['name'],

&nbsp;           json\_schema=json\_schema,

&nbsp;           validators=self.\_extract\_validators(schema\_dict)

&nbsp;       )

&nbsp;       

&nbsp;       return model\_class

```



\### 5.2 JSON Schema Generation



```python

def \_to\_json\_schema(self, schema\_dict: dict) -> dict:

&nbsp;   """Convert Globule schema to JSON Schema"""

&nbsp;   json\_schema = {

&nbsp;       "$schema": "https://json-schema.org/draft/2020-12/schema",

&nbsp;       "$id": schema\_dict.get('$id', f"globule:{schema\_dict\['name']}"),

&nbsp;       "type": "object",

&nbsp;       "properties": {},

&nbsp;       "required": \[]

&nbsp;   }

&nbsp;   

&nbsp;   for field\_name, field\_def in schema\_dict.get('fields', {}).items():

&nbsp;       json\_prop = self.\_convert\_field(field\_def)

&nbsp;       json\_schema\['properties']\[field\_name] = json\_prop

&nbsp;       

&nbsp;       if field\_def.get('required', False):

&nbsp;           json\_schema\['required'].append(field\_name)

&nbsp;           

&nbsp;   # Handle conditional rules

&nbsp;   if 'rules' in schema\_dict:

&nbsp;       json\_schema\['allOf'] = self.\_convert\_rules(schema\_dict\['rules'])

&nbsp;       

&nbsp;   return json\_schema

```



\### 5.3 Dynamic Model Creation



```python

def \_create\_pydantic\_model(self, name: str, json\_schema: dict, 

&nbsp;                         validators: dict) -> Type\[BaseModel]:

&nbsp;   """Generate Pydantic model from JSON Schema"""

&nbsp;   

&nbsp;   # Convert JSON Schema types to Python types

&nbsp;   field\_definitions = {}

&nbsp;   for prop, schema in json\_schema\['properties'].items():

&nbsp;       py\_type = self.\_json\_type\_to\_python(schema)

&nbsp;       required = prop in json\_schema.get('required', \[])

&nbsp;       default = ... if required else schema.get('default', None)

&nbsp;       

&nbsp;       field\_definitions\[prop] = (py\_type, default)

&nbsp;   

&nbsp;   # Add custom validators

&nbsp;   namespace = {

&nbsp;       '\_\_validators\_\_': validators,

&nbsp;       '\_\_module\_\_': f'globule.schemas.{name.lower()}'

&nbsp;   }

&nbsp;   

&nbsp;   # Create the model

&nbsp;   model = create\_model(

&nbsp;       name,

&nbsp;       \*\*field\_definitions,

&nbsp;       \_\_base\_\_=BaseModel,

&nbsp;       \_\_module\_\_=namespace\['\_\_module\_\_'],

&nbsp;       \_\_validators\_\_=namespace.get('\_\_validators\_\_', {})

&nbsp;   )

&nbsp;   

&nbsp;   return model

```



\## 6. Caching Architecture



\### 6.1 Cache Design



```python

from functools import lru\_cache

from typing import Dict, Type

import hashlib



class SchemaCache:

&nbsp;   """High-performance cache for compiled schema models"""

&nbsp;   

&nbsp;   def \_\_init\_\_(self, max\_size: int = 128):

&nbsp;       self.\_cache: Dict\[str, CacheEntry] = {}

&nbsp;       self.\_lru = LRUCache(max\_size)

&nbsp;       

&nbsp;   def get\_model(self, schema\_path: Path) -> Type\[BaseModel]:

&nbsp;       """Get compiled model, regenerating if needed"""

&nbsp;       cache\_key = self.\_compute\_cache\_key(schema\_path)

&nbsp;       

&nbsp;       if cache\_key in self.\_cache:

&nbsp;           entry = self.\_cache\[cache\_key]

&nbsp;           if self.\_is\_valid(entry, schema\_path):

&nbsp;               self.\_lru.touch(cache\_key)

&nbsp;               return entry.model

&nbsp;               

&nbsp;       # Cache miss or stale

&nbsp;       model = self.\_compile\_schema(schema\_path)

&nbsp;       self.\_cache\[cache\_key] = CacheEntry(

&nbsp;           model=model,

&nbsp;           mtime=schema\_path.stat().st\_mtime,

&nbsp;           checksum=self.\_file\_checksum(schema\_path)

&nbsp;       )

&nbsp;       self.\_lru.add(cache\_key)

&nbsp;       

&nbsp;       return model

&nbsp;       

&nbsp;   def invalidate(self, schema\_path: Path) -> None:

&nbsp;       """Remove schema from cache"""

&nbsp;       cache\_key = self.\_compute\_cache\_key(schema\_path)

&nbsp;       if cache\_key in self.\_cache:

&nbsp;           del self.\_cache\[cache\_key]

&nbsp;           self.\_lru.remove(cache\_key)

```



\### 6.2 Hot Reload Integration



```python

class SchemaWatcher:

&nbsp;   """Monitors schema directory for changes"""

&nbsp;   

&nbsp;   def \_\_init\_\_(self, schema\_dir: Path, cache: SchemaCache):

&nbsp;       self.schema\_dir = schema\_dir

&nbsp;       self.cache = cache

&nbsp;       self.observer = Observer()

&nbsp;       

&nbsp;   def start(self):

&nbsp;       handler = SchemaFileHandler(self.cache)

&nbsp;       self.observer.schedule(

&nbsp;           handler, 

&nbsp;           str(self.schema\_dir), 

&nbsp;           recursive=True

&nbsp;       )

&nbsp;       self.observer.start()

&nbsp;       

class SchemaFileHandler(FileSystemEventHandler):

&nbsp;   def on\_modified(self, event):

&nbsp;       if event.src\_path.endswith('.yml'):

&nbsp;           logger.info(f"Schema modified: {event.src\_path}")

&nbsp;           self.cache.invalidate(Path(event.src\_path))

```



\## 7. Security Architecture



\### 7.1 Sandboxed Validator Execution



```python

class SecureValidatorExecutor:

&nbsp;   """Executes custom validators in sandboxed environment"""

&nbsp;   

&nbsp;   def \_\_init\_\_(self):

&nbsp;       self.codejail = CodeJail(

&nbsp;           user='sandbox',

&nbsp;           limits={

&nbsp;               'CPU': 1,      # 1 second max

&nbsp;               'MEMORY': 128  # 128MB max

&nbsp;           }

&nbsp;       )

&nbsp;       

&nbsp;   def execute\_validator(self, validator\_path: str, 

&nbsp;                        value: Any, context: dict) -> Any:

&nbsp;       """Run validator in sandbox"""

&nbsp;       

&nbsp;       # Serialize input data

&nbsp;       input\_json = json.dumps({

&nbsp;           'value': value,

&nbsp;           'context': context

&nbsp;       })

&nbsp;       

&nbsp;       # Prepare sandboxed environment

&nbsp;       code = f"""

import json

import sys

from {validator\_path} import validate



input\_data = json.loads(sys.stdin.read())

result = validate(input\_data\['value'], input\_data\['context'])

print(json.dumps({{'success': True, 'result': result}}))

"""

&nbsp;       

&nbsp;       # Execute in sandbox

&nbsp;       result = self.codejail.execute(

&nbsp;           code=code,

&nbsp;           stdin=input\_json,

&nbsp;           profile='validator\_strict'  # AppArmor profile

&nbsp;       )

&nbsp;       

&nbsp;       return self.\_parse\_result(result)

```



\### 7.2 AppArmor Profiles



```

\# /etc/apparmor.d/globule.validator.strict

profile globule\_validator\_strict {

&nbsp; # No network access

&nbsp; deny network,

&nbsp; 

&nbsp; # Read-only access to Python libs

&nbsp; /usr/lib/python3\*/     r,

&nbsp; /usr/lib/python3\*/\*\*   r,

&nbsp; 

&nbsp; # Read access to validator module only

&nbsp; @{HOME}/.globule/validators/\*\* r,

&nbsp; 

&nbsp; # Write access to temp only

&nbsp; /tmp/globule-sandbox-\*/\*\* rw,

&nbsp; 

&nbsp; # No other filesystem access

&nbsp; deny @{HOME}/\*\* rwx,

&nbsp; deny /etc/\*\* rwx,

&nbsp; deny /proc/\*\* rwx,

}

```



\## 8. Integration Interfaces



\### 8.1 Schema Detection API



```python

class SchemaDetector:

&nbsp;   """Detects appropriate schema for input"""

&nbsp;   

&nbsp;   def detect\_schema(self, text: str, hint: Optional\[str] = None) -> SchemaMatch:

&nbsp;       # 1. Explicit hint (highest priority)

&nbsp;       if hint:

&nbsp;           return SchemaMatch(schema\_id=hint, confidence=1.0)

&nbsp;           

&nbsp;       # 2. Pattern-based triggers

&nbsp;       for schema in self.registry.all\_schemas():

&nbsp;           if self.\_check\_triggers(text, schema.triggers):

&nbsp;               return SchemaMatch(

&nbsp;                   schema\_id=schema.id,

&nbsp;                   confidence=0.9,

&nbsp;                   reason="pattern\_match"

&nbsp;               )

&nbsp;               

&nbsp;       # 3. ML classification (future)

&nbsp;       if self.ml\_classifier:

&nbsp;           prediction = self.ml\_classifier.predict(text)

&nbsp;           if prediction.confidence > 0.7:

&nbsp;               return SchemaMatch(

&nbsp;                   schema\_id=prediction.schema\_id,

&nbsp;                   confidence=prediction.confidence,

&nbsp;                   reason="ml\_classification"

&nbsp;               )

&nbsp;               

&nbsp;       # 4. No match - use free\_text

&nbsp;       return SchemaMatch(

&nbsp;           schema\_id="free\_text",

&nbsp;           confidence=0.5,

&nbsp;           reason="default"

&nbsp;       )

```



\### 8.2 Validation API



```python

class SchemaValidator:

&nbsp;   """Main validation interface"""

&nbsp;   

&nbsp;   async def validate(self, 

&nbsp;                     data: dict, 

&nbsp;                     schema\_id: str) -> ValidationResult:

&nbsp;       """Validate data against schema"""

&nbsp;       

&nbsp;       # Get compiled model from cache

&nbsp;       model = self.cache.get\_model(schema\_id)

&nbsp;       

&nbsp;       try:

&nbsp;           # Validate and parse

&nbsp;           instance = model.model\_validate(data)

&nbsp;           

&nbsp;           # Run custom validators if any

&nbsp;           if hasattr(model, '\_\_validators\_\_'):

&nbsp;               for validator in model.\_\_validators\_\_:

&nbsp;                   if validator.external\_access:

&nbsp;                       instance = await self.\_run\_external\_validator(

&nbsp;                           validator, instance

&nbsp;                       )

&nbsp;                   else:

&nbsp;                       instance = validator(instance)

&nbsp;                       

&nbsp;           return ValidationResult(

&nbsp;               success=True,

&nbsp;               data=instance.model\_dump(),

&nbsp;               schema\_id=schema\_id

&nbsp;           )

&nbsp;           

&nbsp;       except ValidationError as e:

&nbsp;           return ValidationResult(

&nbsp;               success=False,

&nbsp;               errors=self.\_format\_errors(e),

&nbsp;               schema\_id=schema\_id

&nbsp;           )

```



\## 9. User-Facing Tools



\### 9.1 CLI Commands



```bash

\# Create new schema interactively

globule schema new

> Schema name: recipe\_card

> Description: Schema for saving recipes

> Add field (name or blank to finish): title

> Field type \[string]: string

> Required? \[y/N]: y

> Add field: ingredients

> Field type \[string]: array

> ...



\# Create from template

globule schema new --template blog\_post my\_blog



\# Validate schema file

globule schema validate ~/.globule/schemas/user/recipe.yml



\# Generate mock data

globule schema mock recipe\_card

> {

>   "title": "Chocolate Chip Cookies",

>   "ingredients": \["flour", "sugar", "eggs"],

>   "prep\_time": 15,

>   "cook\_time": 12

> }



\# Generate documentation

globule schema docs recipe\_card --format markdown > docs/recipe\_schema.md



\# List all schemas

globule schema list

> Built-in schemas:

>   - free\_text

>   - link\_curation

>   - task\_entry

> User schemas:

>   - personal.journal\_entry

>   - work.standup\_notes

>   - creative.story\_idea



\# Test schema with data

globule schema test recipe\_card sample\_recipe.json

```



\### 9.2 Editor Integration



```python

class SchemaLanguageServer:

&nbsp;   """Provides IDE features for schema authoring"""

&nbsp;   

&nbsp;   def get\_json\_schema\_for\_ide(self, schema\_path: Path) -> dict:

&nbsp;       """Generate JSON Schema for editor validation"""

&nbsp;       

&nbsp;       # Meta-schema for Globule schema format

&nbsp;       return {

&nbsp;           "$schema": "https://json-schema.org/draft/2020-12/schema",

&nbsp;           "type": "object",

&nbsp;           "properties": {

&nbsp;               "name": {"type": "string"},

&nbsp;               "version": {"type": "string", "pattern": "^\\\\d+\\\\.\\\\d+$"},

&nbsp;               "extends": {"type": "string"},

&nbsp;               "fields": {

&nbsp;                   "type": "object",

&nbsp;                   "patternProperties": {

&nbsp;                       "^\[a-z\_]+$": {

&nbsp;                           "type": "object",

&nbsp;                           "properties": {

&nbsp;                               "type": {"enum": \["string", "int", "float", "bool", "array", "object"]},

&nbsp;                               "required": {"type": "boolean"},

&nbsp;                               "default": {},

&nbsp;                               "description": {"type": "string"}

&nbsp;                           }

&nbsp;                       }

&nbsp;                   }

&nbsp;               }

&nbsp;           },

&nbsp;           "required": \["name", "fields"]

&nbsp;       }

```



\## 10. Error Handling and User Feedback



\### 10.1 Error Translation



```python

class UserFriendlyErrors:

&nbsp;   """Translates technical errors to helpful messages"""

&nbsp;   

&nbsp;   ERROR\_TEMPLATES = {

&nbsp;       'int\_parsing': "Please enter a whole number for '{field}'",

&nbsp;       'string\_too\_short': "'{field}' must be at least {min\_length} characters",

&nbsp;       'missing': "'{field}' is required",

&nbsp;       'url\_invalid': "Please enter a valid URL starting with http:// or https://"

&nbsp;   }

&nbsp;   

&nbsp;   def format\_validation\_error(self, error: ValidationError, 

&nbsp;                              schema: dict) -> str:

&nbsp;       """Convert Pydantic error to user-friendly message"""

&nbsp;       

&nbsp;       messages = \[]

&nbsp;       for err in error.errors():

&nbsp;           field = '.'.join(str(x) for x in err\['loc'])

&nbsp;           err\_type = err\['type']

&nbsp;           

&nbsp;           # Check for custom message in schema

&nbsp;           custom\_msg = self.\_get\_custom\_message(schema, field, err\_type)

&nbsp;           if custom\_msg:

&nbsp;               messages.append(custom\_msg.format(\*\*err))

&nbsp;           else:

&nbsp;               # Use default template

&nbsp;               template = self.ERROR\_TEMPLATES.get(err\_type, 

&nbsp;                   "Invalid value for '{field}': {msg}")

&nbsp;               messages.append(template.format(field=field, \*\*err))

&nbsp;               

&nbsp;       return '\\n'.join(messages)

```



\## 11. Performance Specifications



\### 11.1 Performance Targets



| Operation | Target Latency | Notes |

|-----------|---------------|-------|

| Schema validation (cached) | <10ms | Direct model access + validation |

| Schema validation (uncached) | <100ms | File I/O + transpilation + validation |

| Schema detection | <5ms | Pattern matching only |

| Hot reload detection | <1s | File system event latency |

| Mock data generation | <50ms | Using pydantic-faker |

| Documentation generation | <200ms | Template rendering |

| Sandbox validator execution | <1.5s | 1s CPU limit + overhead |



\### 11.2 Optimization Strategies



1\. \*\*Aggressive Caching\*\*: All compiled models cached in memory

2\. \*\*Lazy Loading\*\*: Schemas loaded only when needed

3\. \*\*Batch Operations\*\*: Multiple validations can share sandbox startup

4\. \*\*Precompilation\*\*: Option to precompile all schemas on startup



\## 12. Testing Strategy



\### 12.1 Unit Tests



```python

class TestSchemaTranspiler:

&nbsp;   def test\_basic\_schema\_transpilation(self):

&nbsp;       """Test YAML to Pydantic conversion"""

&nbsp;       

&nbsp;   def test\_inheritance(self):

&nbsp;       """Test extends keyword"""

&nbsp;       

&nbsp;   def test\_composition(self):

&nbsp;       """Test allOf combination"""

&nbsp;       

&nbsp;   def test\_circular\_references(self):

&nbsp;       """Test handling of circular deps"""



class TestSecureExecution:

&nbsp;   def test\_sandbox\_isolation(self):

&nbsp;       """Verify no filesystem access"""

&nbsp;       

&nbsp;   def test\_resource\_limits(self):

&nbsp;       """Test CPU and memory limits"""

&nbsp;       

&nbsp;   def test\_malicious\_validator(self):

&nbsp;       """Test various attack vectors"""

```



\### 12.2 Integration Tests



\- End-to-end validation flow

\- Hot reload with concurrent validation

\- Schema detection accuracy

\- Error message quality



\### 12.3 User Experience Tests



\- Tutorial completion rate

\- Time to create first schema

\- Error message comprehension

\- Mock data realism



\## 13. Migration and Compatibility



\### 13.1 Schema Evolution Rules



For MVP:

\- Adding optional fields: Always safe

\- Adding required fields: Breaking change

\- Removing fields: Breaking change

\- Changing types: Breaking change



Future (with versioning):

\- Schemas can declare compatibility rules

\- Migration scripts for data transformation

\- Deprecation warnings



\### 13.2 Backward Compatibility



\- Built-in schemas are immutable

\- User schemas can override built-ins

\- Old schema formats auto-migrated on load



\## 14. Security Considerations



\### 14.1 Threat Model



1\. \*\*Malicious Validators\*\*: Arbitrary code execution

&nbsp;  - Mitigation: CodeJail + AppArmor

2\. \*\*YAML Bombs\*\*: Resource exhaustion via recursive references

&nbsp;  - Mitigation: Resource limits, depth limits

3\. \*\*Schema Injection\*\*: Malicious schema overwrites built-in

&nbsp;  - Mitigation: Permission checks, separate directories

4\. \*\*Information Disclosure\*\*: Validators accessing sensitive data

&nbsp;  - Mitigation: Strict sandboxing, no filesystem access



\### 14.2 Security Checklist



\- \[ ] All YAML parsing uses safe\_load

\- \[ ] Custom validators run in sandbox

\- \[ ] Resource limits enforced

\- \[ ] No network access by default

\- \[ ] Schema directory permissions checked

\- \[ ] Input size limits enforced



\## 15. Future Enhancements



1\. \*\*ML-based Schema Detection\*\*: Train on user's actual data

2\. \*\*Schema Marketplace\*\*: Share schemas with community

3\. \*\*Visual Schema Builder\*\*: GUI for non-technical users

4\. \*\*Schema Versioning\*\*: Semantic versioning with migrations

5\. \*\*External Integrations\*\*: Import schemas from JSON Schema, OpenAPI

6\. \*\*Performance Profiling\*\*: Per-schema performance metrics



\## 16. Decision Log



| Decision | Rationale | Date |

|----------|-----------|------|

| YAML for authoring | User-friendly, supports comments | 2025-07-11 |

| Pydantic as engine | Performance, type-safety, ecosystem | 2025-07-11 |

| CodeJail for sandboxing | Process-level isolation most secure | 2025-07-11 |

| LRU cache mandatory | 10x performance improvement | 2025-07-11 |

| JSON Schema as IR | Portability, tooling support | 2025-07-11 |



\## 17. Resolved Design Decisions



Based on analysis, the following decisions have been made:



1\. \*\*YAML Merge Keys\*\*: YES - Support `<<` for enhanced reusability and DRY schemas

2\. \*\*Schema Templates\*\*: YES - Include 3-5 advanced templates (OAuth, API response, etc.) 

3\. \*\*Deterministic Mock Data\*\*: YES - Use seeding by default with `--random` flag option

4\. \*\*Built-in Schemas\*\*: 5 schemas (free\_text, link\_curation, task\_entry, meeting\_notes, journal\_entry)

5\. \*\*Inline Python\*\*: NO - Use only referenced validators for security and simplicity



---



\*This LLD provides the complete specification for implementing the Schema Definition Engine. It builds on the exceptional research conducted and provides concrete implementation guidance.\*

