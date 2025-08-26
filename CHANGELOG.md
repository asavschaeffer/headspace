# Changelog

All notable changes to the Globule project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [Phase 3] - 2025-08-18 - Enhanced Configuration System

### Added

#### Configuration System
- **Three-Tier Configuration Cascade**: System defaults → config files → environment variables → explicit overrides
- **Pydantic-based Type Validation**: Type-safe configuration models with automatic validation
- **Cross-Platform Configuration Paths**: 
  - Windows: `%PROGRAMDATA%\Globule\config.yaml`, `%APPDATA%\Globule\config.yaml`
  - Linux/macOS: `/etc/globule/config.yaml`, `~/.config/globule/config.yaml`
- **Environment Variable Support**: `GLOBULE_EMBEDDING__PROVIDER`, `GLOBULE_STORAGE__BACKEND`, etc.
- **YAML Configuration Files**: User and system-level configuration with deep merge support
- **Configuration-Driven Factory Pattern**: Automatic provider selection based on configuration

#### Configuration Models
- `EmbeddingConfig`: Provider, model, and endpoint configuration with validation
- `StorageConfig`: Backend and path configuration with type checking
- `GlobuleConfig`: Root configuration model combining all sections
- `PydanticConfigManager`: Central configuration manager with pydantic-settings integration

#### Factory Integration
- `EmbeddingAdapterFactory`: Creates embedding adapters based on configuration
- `StorageManagerFactory`: Creates storage managers based on configuration  
- `ParserProviderFactory`: Creates parser providers based on configuration
- `OrchestratorFactory`: Creates fully configured orchestrator instances
- `create_default_orchestrator()`: Convenience function for default setup

#### Documentation and Examples
- Comprehensive configuration guide (`docs/configuration.md`)
- Quick reference card (`docs/configuration-quick-reference.md`)
- Environment-specific examples (development, staging, production)
- Docker and Kubernetes deployment examples
- Working Python usage examples with all configuration patterns

#### Backward Compatibility
- Legacy `GlobuleConfig` class maintained with Phase 3 system under the hood
- All existing configuration interfaces preserved
- `get_config()` function continues working unchanged
- No breaking changes to existing APIs

### Enhanced

#### Testing
- **High Test Coverage**: 89% coverage across all configuration modules
- **Focused Test Suite**: Consolidated 79 tests into 33 focused tests (80% reduction in maintenance burden)
- **Cross-Platform Testing**: Verified on Windows, Linux, macOS
- **Integration Testing**: End-to-end configuration cascade verification
- **Golden Snapshot Testing**: Configuration stability verification

#### Developer Experience
- Type-safe configuration with IDE autocompletion
- Clear error messages for configuration validation failures
- Sensible defaults for rapid development setup
- Easy customization for production deployments

### Technical Details

#### Implementation
- Built on `pydantic-settings` for robust configuration management
- Custom `MultiYamlSettingsSource` for YAML file cascade loading
- `functools.reduce` for efficient dot-notation configuration access
- Atomic git commits following conventional commit format
- Comprehensive error handling with domain-specific exceptions

#### Architecture
- Added configuration layer to four-layer architecture:
  1. Frontend Layer (CLI, TUI, Web)
  2. **Configuration Layer** (PydanticConfigManager) - NEW
  3. Factory Layer (Configuration-driven factories)
  4. Orchestration Layer (GlobuleOrchestrator)
  5. Adapter Layer (Provider abstractions)
  6. Provider Layer (Concrete implementations)

#### Dependencies
- Added `pydantic-settings` for configuration management
- Added `PyYAML` for YAML configuration file support
- Enhanced `pathlib` usage for cross-platform path handling

### Files Added
- `src/globule/config/manager.py` - Core configuration manager
- `src/globule/config/models.py` - Pydantic configuration models
- `src/globule/config/sources.py` - YAML loading and deep merge utilities
- `src/globule/config/paths.py` - Cross-platform path resolution
- `src/globule/config/errors.py` - Configuration error hierarchy
- `src/globule/config/settings.py` - Legacy compatibility layer
- `src/globule/core/factories.py` - Configuration-driven factories
- `examples/config/system-config.yaml` - System configuration example
- `examples/config/user-config.yaml` - User configuration example
- `examples/environments/development.yaml` - Development environment config
- `examples/environments/staging.yaml` - Staging environment config
- `examples/environments/production.yaml` - Production environment config
- `examples/deployments/docker-compose.yml` - Docker deployment example
- `examples/deployments/kubernetes.yaml` - Kubernetes deployment example
- `examples/usage-examples.py` - Working Python examples
- `docs/configuration.md` - Comprehensive configuration guide
- `docs/configuration-quick-reference.md` - Quick reference card
- `docs/adr/ADR-0003.md` - Architecture Decision Record
- `docs/ARCHITECTURE-PHASE-3.md` - Updated architecture documentation
- `tests/unit/test_config_focused.py` - Consolidated focused test suite

### Removed
- Redundant test files with overlapping coverage (test audit cleanup)
- Hardcoded configuration values scattered throughout codebase

---

## [Phase 2] - 2025-08-14 - The Adapter Layer

### Added

#### Adapter Pattern Implementation
- **Abstract Provider Interfaces**: `IParserProvider`, `IEmbeddingProvider`, `IStorageManager`
- **Adapter Implementations**: `OllamaParsingAdapter`, `OllamaEmbeddingAdapter`, `SqliteStorageAdapter`
- **Dependency Injection**: Constructor injection of provider interfaces into `GlobuleOrchestrator`
- **Error Translation**: Clean conversion of provider-specific errors to domain errors
- **Type Normalization**: Consistent data types returned to core engine

#### Provider Agnosticism
- Core engine completely decoupled from specific providers (Ollama, SQLite)
- Zero knowledge of concrete implementations in business logic
- Easy provider swapping through adapter pattern
- Consistent async interfaces across all providers

#### Testing Infrastructure
- Comprehensive adapter unit tests (100% coverage)
- Integration tests with mock providers
- End-to-end headless processing validation
- Interface compliance testing with dummy implementations

### Enhanced
- **Testability**: Core engine can be tested with mock providers
- **Modularity**: Clean separation between core logic and external services
- **Extensibility**: New providers can be added without changing core engine
- **Error Handling**: Structured error boundaries at adapter layer

### Technical Details
- Built on Abstract Base Classes (ABCs) for interface contracts
- Async/await patterns throughout adapter layer
- Error translation with exception chaining for debugging
- Type safety with proper return type normalization

---

## [Phase 1] - 2025-08-12 - Contracts-First Foundation

### Added

#### Contracts-First Architecture
- **Data Contracts**: Immutable Pydantic models (`GlobuleV1`, `ProcessedGlobuleV1`)
- **Service Contracts**: Abstract Base Classes defining core interfaces
- **Headless Core**: Complete decoupling of business logic from UI and external services
- **Unidirectional Dependencies**: `UI` → `Engine` → `Adapters` → `Providers`

#### Core Models
- `GlobuleV1`: Primary data structure for thought capture
- `ProcessedGlobuleV1`: Enhanced globule with embeddings and metadata
- `FileDecision`: Structured file operation decisions
- Versioned models for backward compatibility

#### Interface Definitions
- `IOrchestrationEngine`: Core business logic interface
- `IParserProvider`: Text parsing service interface  
- `IEmbeddingProvider`: Vector embedding service interface
- `IStorageManager`: Data persistence interface

### Enhanced
- **Parallel Development**: Teams can work on different components independently
- **Clear Documentation**: Contracts serve as enforceable component interaction docs
- **Maintainability**: Enforced separation of concerns through interface boundaries

### Technical Details
- Pydantic v2 for data validation and serialization
- Abstract Base Classes for service contracts
- Immutable data structures throughout the system
- Type hints and validation for all interfaces

---

## Project Initialization - 2025-08-12

### Added
- Initial project structure and configuration
- Basic CLI interface foundation
- Core domain models and concepts
- Development environment setup
- Testing framework configuration