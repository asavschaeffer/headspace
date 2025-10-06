#!/usr/bin/env python3
"""
Model Monitor - Comprehensive monitoring and logging for all AI models
Provides real-time status checking, health monitoring, and detailed logging
"""

import json
import logging
import time
import os
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from pathlib import Path
import sys

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    handlers=[
        logging.FileHandler('model_status.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class ModelStatus(Enum):
    """Status codes for model health"""
    HEALTHY = "✅ HEALTHY"
    DEGRADED = "⚠️ DEGRADED"
    FAILED = "❌ FAILED"
    INITIALIZING = "🔄 INITIALIZING"
    NOT_CONFIGURED = "⚪ NOT CONFIGURED"
    UNKNOWN = "❓ UNKNOWN"

class ModelType(Enum):
    """Types of models being monitored"""
    EMBEDDING = "Embedding Model"
    LLM = "Language Model"
    CHUNKER = "Chunker Model"
    TAGGER = "Tagger Model"

class ModelMonitor:
    """
    Comprehensive model monitoring and status tracking
    Provides detailed logging and health checking for all AI models
    """

    def __init__(self, log_level: str = "DEBUG"):
        """Initialize the model monitor with specified log level"""
        self.logger = logging.getLogger("ModelMonitor")
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.DEBUG))

        self.model_status: Dict[str, Dict[str, Any]] = {}
        self.performance_metrics: Dict[str, List[float]] = {}
        self.error_history: Dict[str, List[Dict]] = {}
        self.start_time = datetime.now()

        self.logger.info("="*80)
        self.logger.info("🚀 MODEL MONITOR INITIALIZED")
        self.logger.info(f"Start time: {self.start_time.isoformat()}")
        self.logger.info(f"Log level: {log_level}")
        self.logger.info("="*80)

    def check_ollama_status(self, ollama_url: Optional[str] = None) -> Tuple[ModelStatus, Dict[str, Any]]:
        """
        Comprehensive Ollama status check with detailed diagnostics
        """
        # Use provided URL or default, handle both base URL and full API URL
        if not ollama_url:
            ollama_url = os.environ.get('OLLAMA_URL', 'http://localhost:11434')

        # Extract base URL if it includes /api path
        base_url = ollama_url.split('/api')[0] if '/api' in ollama_url else ollama_url

        self.logger.debug(f"Checking Ollama status at {base_url}...")
        status_details = {
            "service": "Ollama",
            "url": base_url,
            "timestamp": datetime.now().isoformat()
        }

        try:
            # Check if Ollama is running
            response = requests.get(f'{base_url}/api/tags', timeout=5)

            if response.status_code == 200:
                models = response.json().get('models', [])
                status_details['available_models'] = [m.get('name', 'unknown') for m in models]
                status_details['model_count'] = len(models)
                status_details['response_time_ms'] = response.elapsed.total_seconds() * 1000

                if models:
                    self.logger.info(f"✅ Ollama HEALTHY - {len(models)} model(s) available: {', '.join(status_details['available_models'])}")
                    status_details['status'] = ModelStatus.HEALTHY

                    # Check each model's details
                    for model in models:
                        model_name = model.get('name', 'unknown')
                        model_info = {
                            'size': model.get('size', 'unknown'),
                            'modified': model.get('modified_at', 'unknown'),
                            'digest': model.get('digest', 'unknown')[:12] + '...'
                        }
                        self.logger.debug(f"  Model '{model_name}': size={model_info['size']}, modified={model_info['modified']}")
                else:
                    self.logger.warning("⚠️ Ollama DEGRADED - Service running but no models loaded")
                    status_details['status'] = ModelStatus.DEGRADED
                    status_details['warning'] = "No models loaded. Run 'ollama pull <model>' to download models"

                return status_details['status'], status_details

            else:
                self.logger.error(f"❌ Ollama FAILED - Unexpected response code: {response.status_code}")
                status_details['status'] = ModelStatus.FAILED
                status_details['error'] = f"Unexpected response code: {response.status_code}"
                return ModelStatus.FAILED, status_details

        except requests.exceptions.ConnectionError:
            self.logger.error(f"❌ Ollama NOT RUNNING - Connection refused at {base_url}")
            self.logger.error("   Fix: Start Ollama with 'ollama serve' or install from https://ollama.ai")
            if 'host.docker.internal' in base_url:
                self.logger.error("   NOTE: Running in Docker - make sure Ollama is running on your HOST machine")
            status_details['status'] = ModelStatus.FAILED
            status_details['error'] = f"Connection refused to {base_url}"
            status_details['fix'] = "Start Ollama with 'ollama serve' on your host machine"
            return ModelStatus.FAILED, status_details

        except requests.exceptions.Timeout:
            self.logger.error("❌ Ollama TIMEOUT - Service not responding within 5 seconds")
            status_details['status'] = ModelStatus.FAILED
            status_details['error'] = "Request timeout"
            return ModelStatus.FAILED, status_details

        except Exception as e:
            self.logger.error(f"❌ Ollama ERROR - Unexpected error: {str(e)}")
            status_details['status'] = ModelStatus.FAILED
            status_details['error'] = str(e)
            return ModelStatus.FAILED, status_details

    def check_sentence_transformers_status(self, model_name: str = "all-MiniLM-L6-v2") -> Tuple[ModelStatus, Dict[str, Any]]:
        """
        Check sentence-transformers model status with detailed diagnostics
        """
        self.logger.debug(f"Checking sentence-transformers status for model: {model_name}")
        status_details = {
            "service": "sentence-transformers",
            "model_name": model_name,
            "timestamp": datetime.now().isoformat()
        }

        try:
            from sentence_transformers import SentenceTransformer

            # Try to load the model
            start_time = time.time()
            self.logger.info(f"🔄 Loading sentence-transformer model: {model_name}")

            model = SentenceTransformer(model_name)
            load_time = time.time() - start_time

            # Get model details
            embedding_dim = model.get_sentence_embedding_dimension()
            status_details['embedding_dimension'] = embedding_dim
            status_details['load_time_seconds'] = load_time
            status_details['model_path'] = str(Path.home() / '.cache' / 'torch' / 'sentence_transformers' / model_name)

            # Test the model
            test_text = "This is a test sentence."
            start_time = time.time()
            test_embedding = model.encode([test_text])
            inference_time = time.time() - start_time

            status_details['inference_time_ms'] = inference_time * 1000
            status_details['test_embedding_shape'] = test_embedding.shape

            self.logger.info(f"✅ Sentence-Transformers HEALTHY - Model '{model_name}' loaded successfully")
            self.logger.info(f"   Embedding dimension: {embedding_dim}")
            self.logger.info(f"   Load time: {load_time:.2f}s, Inference time: {inference_time*1000:.2f}ms")

            status_details['status'] = ModelStatus.HEALTHY
            return ModelStatus.HEALTHY, status_details

        except ImportError:
            self.logger.error("❌ Sentence-Transformers NOT INSTALLED")
            self.logger.error("   Fix: pip install sentence-transformers")
            status_details['status'] = ModelStatus.NOT_CONFIGURED
            status_details['error'] = "sentence-transformers not installed"
            status_details['fix'] = "pip install sentence-transformers"
            return ModelStatus.NOT_CONFIGURED, status_details

        except Exception as e:
            self.logger.error(f"❌ Sentence-Transformers FAILED - {str(e)}")
            status_details['status'] = ModelStatus.FAILED
            status_details['error'] = str(e)
            return ModelStatus.FAILED, status_details

    def check_gemini_status(self, api_key: Optional[str] = None) -> Tuple[ModelStatus, Dict[str, Any]]:
        """
        Check Google Gemini API status with detailed diagnostics
        """
        self.logger.debug("Checking Gemini API status...")
        status_details = {
            "service": "Gemini API",
            "timestamp": datetime.now().isoformat()
        }

        if not api_key or api_key == "YOUR_GEMINI_API_KEY":
            self.logger.warning("⚠️ Gemini NOT CONFIGURED - API key missing or invalid")
            status_details['status'] = ModelStatus.NOT_CONFIGURED
            status_details['error'] = "API key not configured"
            status_details['fix'] = "Set GEMINI_API_KEY environment variable"
            return ModelStatus.NOT_CONFIGURED, status_details

        try:
            # Test the API with a simple request
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m.get('name', 'unknown').split('/')[-1] for m in models]

                status_details['available_models'] = available_models
                status_details['model_count'] = len(available_models)
                status_details['response_time_ms'] = response.elapsed.total_seconds() * 1000

                self.logger.info(f"✅ Gemini API HEALTHY - {len(available_models)} model(s) available")
                self.logger.debug(f"   Available models: {', '.join(available_models[:5])}...")

                status_details['status'] = ModelStatus.HEALTHY
                return ModelStatus.HEALTHY, status_details

            elif response.status_code == 403:
                self.logger.error("❌ Gemini API KEY INVALID - Authentication failed")
                status_details['status'] = ModelStatus.FAILED
                status_details['error'] = "Invalid API key"
                return ModelStatus.FAILED, status_details

            else:
                self.logger.error(f"❌ Gemini API ERROR - Status code: {response.status_code}")
                status_details['status'] = ModelStatus.FAILED
                status_details['error'] = f"API returned status code: {response.status_code}"
                return ModelStatus.FAILED, status_details

        except Exception as e:
            self.logger.error(f"❌ Gemini API ERROR - {str(e)}")
            status_details['status'] = ModelStatus.FAILED
            status_details['error'] = str(e)
            return ModelStatus.FAILED, status_details

    def register_model(self, model_id: str, model_type: ModelType, provider: str, model_name: str):
        """
        Register a model for monitoring
        """
        self.logger.info(f"📝 Registering model: {model_id} ({model_type.value}) - {provider}/{model_name}")

        self.model_status[model_id] = {
            "id": model_id,
            "type": model_type.value,
            "provider": provider,
            "model_name": model_name,
            "status": ModelStatus.INITIALIZING,
            "registered_at": datetime.now().isoformat(),
            "last_check": None,
            "metrics": {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "total_latency_ms": 0,
                "average_latency_ms": 0
            }
        }

        if model_id not in self.performance_metrics:
            self.performance_metrics[model_id] = []
        if model_id not in self.error_history:
            self.error_history[model_id] = []

    def update_model_status(self, model_id: str, status: ModelStatus, details: Optional[Dict] = None):
        """
        Update the status of a registered model
        """
        if model_id not in self.model_status:
            self.logger.warning(f"Model {model_id} not registered, registering now...")
            self.register_model(model_id, ModelType.UNKNOWN, "unknown", "unknown")

        self.model_status[model_id]["status"] = status
        self.model_status[model_id]["last_check"] = datetime.now().isoformat()

        if details:
            # Convert ModelStatus enums to strings for JSON serialization
            serializable_details = {}
            for key, value in details.items():
                if isinstance(value, ModelStatus):
                    serializable_details[key] = value.value
                else:
                    serializable_details[key] = value
            self.model_status[model_id]["details"] = serializable_details

        self.logger.info(f"📊 Model Status Update: {model_id} -> {status.value}")
        if details:
            self.logger.debug(f"   Details: {json.dumps(serializable_details, indent=2)}")

    def record_model_usage(self, model_id: str, success: bool, latency_ms: float, error: Optional[str] = None):
        """
        Record usage metrics for a model
        """
        if model_id not in self.model_status:
            self.logger.warning(f"Model {model_id} not registered")
            return

        metrics = self.model_status[model_id]["metrics"]
        metrics["requests"] += 1

        if success:
            metrics["successes"] += 1
            self.logger.debug(f"✓ {model_id}: Request successful ({latency_ms:.2f}ms)")
        else:
            metrics["failures"] += 1
            self.logger.warning(f"✗ {model_id}: Request failed ({latency_ms:.2f}ms) - {error}")

            # Record error in history
            self.error_history[model_id].append({
                "timestamp": datetime.now().isoformat(),
                "error": error,
                "latency_ms": latency_ms
            })

            # Keep only last 100 errors
            if len(self.error_history[model_id]) > 100:
                self.error_history[model_id] = self.error_history[model_id][-100:]

        # Update latency metrics
        metrics["total_latency_ms"] += latency_ms
        metrics["average_latency_ms"] = metrics["total_latency_ms"] / metrics["requests"]

        # Record performance metric
        self.performance_metrics[model_id].append(latency_ms)

        # Keep only last 1000 metrics
        if len(self.performance_metrics[model_id]) > 1000:
            self.performance_metrics[model_id] = self.performance_metrics[model_id][-1000:]

    def get_full_status_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive status report for all models
        """
        uptime = (datetime.now() - self.start_time).total_seconds()

        report = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "models": {}
        }

        for model_id, status in self.model_status.items():
            report["models"][model_id] = {
                **status,
                "status": status["status"].value if isinstance(status["status"], ModelStatus) else status["status"]
            }

            # Add error summary if errors exist
            if model_id in self.error_history and self.error_history[model_id]:
                recent_errors = self.error_history[model_id][-5:]
                report["models"][model_id]["recent_errors"] = recent_errors

        return report

    def print_status_dashboard(self):
        """
        Print a formatted status dashboard to the console
        """
        print("\n" + "="*80)
        print("🎯 MODEL STATUS DASHBOARD")
        print("="*80)
        print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️ Uptime: {(datetime.now() - self.start_time).total_seconds():.0f} seconds")
        print("-"*80)

        if not self.model_status:
            print("No models registered yet.")
        else:
            for model_id, status in self.model_status.items():
                metrics = status.get("metrics", {})
                print(f"\n📦 {model_id}")
                print(f"   Type: {status.get('type', 'Unknown')}")
                print(f"   Provider: {status.get('provider', 'Unknown')}")
                print(f"   Model: {status.get('model_name', 'Unknown')}")
                print(f"   Status: {status.get('status').value if isinstance(status.get('status'), ModelStatus) else status.get('status', 'Unknown')}")
                print(f"   Requests: {metrics.get('requests', 0)} (✓ {metrics.get('successes', 0)} / ✗ {metrics.get('failures', 0)})")

                if metrics.get('requests', 0) > 0:
                    success_rate = (metrics.get('successes', 0) / metrics.get('requests', 0)) * 100
                    print(f"   Success Rate: {success_rate:.1f}%")
                    print(f"   Avg Latency: {metrics.get('average_latency_ms', 0):.2f}ms")

        print("\n" + "="*80 + "\n")

    def check_all_services(self, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run a comprehensive check on all configured services
        """
        self.logger.info("="*80)
        self.logger.info("🔍 RUNNING COMPREHENSIVE SERVICE CHECK")
        self.logger.info("="*80)

        results = {}

        # Get Ollama URL from config or environment
        ollama_url = None
        if config:
            ollama_url = config.get('ollama_url', os.environ.get('OLLAMA_URL'))

        # Check Ollama
        self.logger.info("\n1. Checking Ollama Service...")
        ollama_status, ollama_details = self.check_ollama_status(ollama_url)
        results['ollama'] = ollama_details

        # Check Sentence Transformers
        self.logger.info("\n2. Checking Sentence Transformers...")
        st_status, st_details = self.check_sentence_transformers_status()
        results['sentence_transformers'] = st_details

        # Check Gemini if API key provided
        if config and config.get('gemini_api_key'):
            self.logger.info("\n3. Checking Gemini API...")
            gemini_status, gemini_details = self.check_gemini_status(config['gemini_api_key'])
            results['gemini'] = gemini_details
        else:
            self.logger.info("\n3. Skipping Gemini API (no API key configured)")
            results['gemini'] = {"status": ModelStatus.NOT_CONFIGURED, "error": "No API key provided"}

        self.logger.info("\n" + "="*80)
        self.logger.info("✅ SERVICE CHECK COMPLETE")
        self.logger.info("="*80)

        # Summary
        healthy_count = sum(1 for r in results.values() if r.get('status') == ModelStatus.HEALTHY)
        failed_count = sum(1 for r in results.values() if r.get('status') == ModelStatus.FAILED)
        not_configured_count = sum(1 for r in results.values() if r.get('status') == ModelStatus.NOT_CONFIGURED)

        self.logger.info(f"\nSUMMARY:")
        self.logger.info(f"  ✅ Healthy: {healthy_count}")
        self.logger.info(f"  ❌ Failed: {failed_count}")
        self.logger.info(f"  ⚪ Not Configured: {not_configured_count}")

        return results

if __name__ == "__main__":
    # Test the model monitor
    monitor = ModelMonitor(log_level="DEBUG")

    # Run comprehensive check
    results = monitor.check_all_services()

    # Register some models for monitoring
    monitor.register_model("embedder-1", ModelType.EMBEDDING, "sentence-transformers", "all-MiniLM-L6-v2")
    monitor.register_model("llm-1", ModelType.LLM, "ollama", "llama2")

    # Simulate some usage
    monitor.record_model_usage("embedder-1", True, 15.3)
    monitor.record_model_usage("embedder-1", True, 12.7)
    monitor.record_model_usage("llm-1", False, 5000, "Connection timeout")

    # Print dashboard
    monitor.print_status_dashboard()

    # Get full report
    report = monitor.get_full_status_report()
    print("\nFull JSON Report:")
    print(json.dumps(report, indent=2))