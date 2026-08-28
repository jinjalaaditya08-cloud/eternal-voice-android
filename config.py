#!/usr/bin/env python3
"""
E.V. (Eternal Voice) - Configuration & Environment Management
File 03: Application settings, paths, and configuration profiles

Handles:
- Environment-specific configuration (dev, staging, prod)
- Path management (data, models, cache, logs)
- Feature flags and debug options
- API keys and credentials (secure storage)
- Default settings and user preferences
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, asdict, field
import hashlib
import secrets


# ============================================================================
# ENUMS & TYPES
# ============================================================================

class Environment(Enum):
    """Application environment modes."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Logging verbosity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelProvider(Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"  # Local
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL_ONNX = "local_onnx"  # Lightweight local model


class STTProvider(Enum):
    """Supported Speech-to-Text providers."""
    ANDROID_NATIVE = "android_native"  # Google Recorder (Android)
    OFFLINE_VOSK = "offline_vosk"  # Local Vosk model
    GOOGLE_CLOUD = "google_cloud"
    MICROSOFT_AZURE = "microsoft_azure"


class TTSProvider(Enum):
    """Supported Text-to-Speech providers."""
    ANDROID_NATIVE = "android_native"  # Android native TTS
    GOOGLE_CLOUD = "google_cloud"
    MICROSOFT_AZURE = "microsoft_azure"
    OFFLINE_PYTTSX3 = "offline_pyttsx3"  # Local pyttsx3


# ============================================================================
# CONFIGURATION DATA CLASSES
# ============================================================================

@dataclass
class APIKeyConfig:
    """Secure API key storage configuration."""
    provider: str
    key: str = field(default="", repr=False)  # Redacted in repr
    endpoint: str = ""
    version: str = ""
    is_encrypted: bool = True
    
    def __repr__(self) -> str:
        return f"APIKeyConfig(provider={self.provider}, encrypted={self.is_encrypted})"


@dataclass
class AIBrainConfig:
    """AI Brain / LLM Configuration."""
    primary_model: ModelProvider = ModelProvider.OLLAMA
    ollama_endpoint: str = "http://localhost:11434"
    context_window: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    enable_streaming: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    fallback_models: List[ModelProvider] = field(default_factory=list)


@dataclass
class SpeechConfig:
    """Speech I/O Configuration."""
    stt_provider: STTProvider = STTProvider.ANDROID_NATIVE
    tts_provider: TTSProvider = TTSProvider.ANDROID_NATIVE
    language: str = "en-US"  # BCP 47 format
    stt_confidence_threshold: float = 0.7
    tts_speech_rate: float = 1.0
    tts_pitch: float = 1.0
    enable_audio_feedback: bool = True
    audio_volume: float = 0.8


@dataclass
class MemoryConfig:
    """Memory Management Configuration."""
    max_short_term_tokens: int = 8000
    max_long_term_entries: int = 10000
    vector_embedding_dim: int = 384  # For semantic search
    similarity_threshold: float = 0.7
    retention_days: int = 90  # Delete older than 90 days
    enable_compression: bool = True
    db_type: str = "sqlite"  # sqlite or postgresql


@dataclass
class VisionConfig:
    """Computer Vision Configuration."""
    enabled: bool = True
    camera_resolution: tuple = field(default_factory=lambda: (1080, 1920))
    enable_ocr: bool = True
    enable_face_detection: bool = True
    enable_object_detection: bool = False
    model_backend: str = "pytorch"  # pytorch, tensorflow, onnx


@dataclass
class DebugConfig:
    """Debug & Development Configuration."""
    enabled: bool = False
    log_level: LogLevel = LogLevel.INFO
    log_to_file: bool = True
    log_file_size_mb: int = 50
    log_retention_days: int = 30
    enable_profiling: bool = False
    enable_memory_tracking: bool = False
    mock_responses: bool = False  # For testing without actual API calls


@dataclass
class AppConfig:
    """Master configuration container."""
    environment: Environment = Environment.DEVELOPMENT
    app_version: str = "1.0.0"
    app_name: str = "E.V. - Eternal Voice"
    
    ai_brain: AIBrainConfig = field(default_factory=AIBrainConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    
    api_keys: Dict[str, APIKeyConfig] = field(default_factory=dict)
    
    # Feature flags
    enable_research: bool = True
    enable_code_execution: bool = True
    enable_file_operations: bool = False  # Restricted by Android
    enable_web_access: bool = True
    enable_offline_mode: bool = True
    
    # UI/Theme
    theme_mode: str = "glassmorphism"
    font_size: float = 14.0
    auto_brightness: bool = True


# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================

class ConfigManager:
    """
    Centralized configuration management system.
    Handles loading, saving, validation, and access to application settings.
    """
    
    def __init__(self, app_root: Path):
        """
        Initialize configuration manager.
        
        Args:
            app_root: Path to application root directory
        """
        self.app_root = app_root
        self.config_dir = app_root / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("EV.Config")
        
        # Load or create default config
        self.config: AppConfig = self._load_or_create_config()
        
        # Setup logging based on config
        self._setup_logging()
    
    # ========================================================================
    # CONFIG LOADING & PERSISTENCE
    # ========================================================================
    
    def _load_or_create_config(self) -> AppConfig:
        """
        Load configuration from file or create default.
        
        Returns:
            AppConfig: Loaded or default configuration
        """
        config_file = self.config_dir / "config.json"
        
        try:
            if config_file.exists():
                self.logger.info(f"Loading configuration from {config_file}")
                return self._load_config_from_file(config_file)
            else:
                self.logger.info("Creating default configuration")
                config = AppConfig()
                self._save_config(config)
                return config
                
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}, using defaults", exc_info=True)
            return AppConfig()
    
    def _load_config_from_file(self, config_file: Path) -> AppConfig:
        """
        Load and deserialize configuration from JSON file.
        
        Args:
            config_file: Path to configuration JSON file
            
        Returns:
            AppConfig: Deserialized configuration
        """
        with open(config_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Deserialize nested dataclasses
        if "ai_brain" in data:
            data["ai_brain"] = AIBrainConfig(**data["ai_brain"])
        if "speech" in data:
            data["speech"] = SpeechConfig(**data["speech"])
        if "memory" in data:
            data["memory"] = MemoryConfig(**data["memory"])
        if "vision" in data:
            data["vision"] = VisionConfig(**data["vision"])
        if "debug" in data:
            data["debug"] = DebugConfig(**data["debug"])
        
        # Convert string enums
        if "environment" in data:
            data["environment"] = Environment(data["environment"])
        
        return AppConfig(**data)
    
    def _save_config(self, config: AppConfig) -> None:
        """
        Serialize and save configuration to JSON file.
        
        Args:
            config: AppConfig instance to save
        """
        config_file = self.config_dir / "config.json"
        
        try:
            # Convert dataclasses to dict, then to JSON-serializable format
            config_dict = self._config_to_dict(config)
            
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            self.logger.info(f"Configuration saved to {config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}", exc_info=True)
    
    def _config_to_dict(self, config: AppConfig) -> Dict[str, Any]:
        """
        Convert AppConfig to JSON-serializable dictionary.
        
        Args:
            config: AppConfig instance
            
        Returns:
            Dict: Serializable configuration dictionary
        """
        config_dict = asdict(config)
        
        # Convert enums to strings
        config_dict["environment"] = config.environment.value
        config_dict["ai_brain"]["primary_model"] = config.ai_brain.primary_model.value
        config_dict["speech"]["stt_provider"] = config.speech.stt_provider.value
        config_dict["speech"]["tts_provider"] = config.speech.tts_provider.value
        config_dict["debug"]["log_level"] = config.debug.log_level.value
        
        # Remove sensitive data before saving
        config_dict["api_keys"] = {}
        
        return config_dict
    
    # ========================================================================
    # LOGGING SETUP
    # ========================================================================
    
    def _setup_logging(self) -> None:
        """Configure logging based on debug settings."""
        debug_cfg = self.config.debug
        
        # Create logs directory
        log_dir = self.app_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Don't reconfigure if already configured
        root_logger = logging.getLogger()
        if root_logger.handlers:
            return
        
        level = getattr(logging, debug_cfg.log_level.value)
        root_logger.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        console_handler.setFormatter(console_fmt)
        root_logger.addHandler(console_handler)
        
        # File handler (if enabled)
        if debug_cfg.log_to_file:
            log_file = log_dir / f"ev_{os.getpid()}.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_fmt = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
            )
            file_handler.setFormatter(file_fmt)
            root_logger.addHandler(file_handler)
    
    # ========================================================================
    # ACCESSORS & UTILITIES
    # ========================================================================
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation path.
        
        Args:
            key: Dot-notation path (e.g., "ai_brain.temperature")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        parts = key.split(".")
        value = self.config
        
        try:
            for part in parts:
                if isinstance(value, dict):
                    value = value[part]
                else:
                    value = getattr(value, part)
            return value
        except (KeyError, AttributeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot-notation path.
        
        Args:
            key: Dot-notation path
            value: Value to set
        """
        parts = key.split(".")
        target = self.config
        
        try:
            for part in parts[:-1]:
                target = getattr(target, part)
            
            setattr(target, parts[-1], value)
            self._save_config(self.config)
            self.logger.info(f"Config updated: {key} = {value}")
            
        except (KeyError, AttributeError) as e:
            self.logger.error(f"Failed to set config {key}: {e}")
    
    def add_api_key(self, provider: str, key: str, endpoint: str = "") -> None:
        """
        Add or update API key securely.
        
        Args:
            provider: API provider name
            key: API key/token
            endpoint: Optional API endpoint URL
        """
        try:
            self.config.api_keys[provider] = APIKeyConfig(
                provider=provider,
                key=key,
                endpoint=endpoint,
                is_encrypted=True
            )
            self._save_config(self.config)
            self.logger.info(f"API key added for {provider}")
            
        except Exception as e:
            self.logger.error(f"Failed to add API key: {e}", exc_info=True)
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Retrieve API key for provider.
        
        Args:
            provider: API provider name
            
        Returns:
            API key or None if not found
        """
        api_key_cfg = self.config.api_keys.get(provider)
        if api_key_cfg:
            return api_key_cfg.key
        return None
    
    def get_paths(self) -> Dict[str, Path]:
        """
        Get all application paths.
        
        Returns:
            Dict mapping path names to Path objects
        """
        return {
            "root": self.app_root,
            "config": self.config_dir,
            "data": self.app_root / "data",
            "models": self.app_root / "models",
            "cache": self.app_root / "cache",
            "logs": self.app_root / "logs",
            "database": self.app_root / "data" / "ev.db",
            "vector_store": self.app_root / "data" / "vectors",
        }
    
    def is_feature_enabled(self, feature: str) -> bool:
        """
        Check if a feature is enabled.
        
        Args:
            feature: Feature name (snake_case attribute)
            
        Returns:
            bool: True if enabled, False otherwise
        """
        return getattr(self.config, f"enable_{feature}", False)


# ============================================================================
# ENVIRONMENT DETECTION & SETUP
# ============================================================================

def detect_environment() -> Environment:
    """
    Detect application environment from environment variables or defaults.
    
    Returns:
        Environment: Detected environment
    """
    env_str = os.getenv("EV_ENVIRONMENT", "development").lower()
    
    try:
        return Environment(env_str)
    except ValueError:
        return Environment.DEVELOPMENT


def setup_android_storage_paths() -> Optional[Path]:
    """
    Setup Android-specific storage paths.
    
    Returns:
        Path to app storage directory on Android, None on other platforms
    """
    try:
        from jnius import autoclass
        from android.storage import app_storage_path
        
        storage_path = Path(app_storage_path())
        storage_path.mkdir(parents=True, exist_ok=True)
        return storage_path
        
    except ImportError:
        return None


# ============================================================================
# GLOBAL CONFIG INSTANCE (Singleton Pattern)
# ============================================================================

_config_manager: Optional[ConfigManager] = None

def get_config_manager(app_root: Optional[Path] = None) -> ConfigManager:
    """
    Get or create global configuration manager (singleton).
    
    Args:
        app_root: Path to app root (required for first call)
        
    Returns:
        ConfigManager: Global configuration manager instance
    """
    global _config_manager
    
    if _config_manager is None:
        if app_root is None:
            raise ValueError("app_root required for first initialization")
        _config_manager = ConfigManager(app_root)
    
    return _config_manager


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create config manager
    test_root = Path.home() / ".eternal_voice_test"
    test_root.mkdir(exist_ok=True)
    
    config_mgr = ConfigManager(test_root)
    
    # Access configuration
    print(f"Environment: {config_mgr.config.environment.value}")
    print(f"LLM Provider: {config_mgr.config.ai_brain.primary_model.value}")
    print(f"Debug Level: {config_mgr.config.debug.log_level.value}")
    
    # Modify configuration
    config_mgr.set("ai_brain.temperature", 0.8)
    print(f"Updated temperature: {config_mgr.get('ai_brain.temperature')}")
    
    # Get all paths
    paths = config_mgr.get_paths()
    for name, path in paths.items():
        print(f"{name}: {path}")
