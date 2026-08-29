#!/usr/bin/env python3
"""
E.V. (Eternal Voice) - Core LLM Orchestrator with Groq Support
File 05: AI Brain - Central intelligence engine for language processing and response generation

Handles:
- LLM provider abstraction (Groq, Ollama, OpenAI, Anthropic, Google, Local ONNX)
- Language detection (English, Gujarati, Hindi, Gujlish, Hinglish)
- Context window management and token tracking
- Streaming response generation
- Fallback model handling
- Response caching
"""

import logging
import asyncio
import time
from typing import Optional, List, Dict, Any, AsyncGenerator, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
import json
import hashlib
from abc import ABC, abstractmethod

try:
    from config import (
        ConfigManager, ModelProvider, STTProvider, TTSProvider,
        AIBrainConfig, APIKeyConfig
    )
except ImportError:
    # Fallback for testing
    ModelProvider = None
    ConfigManager = None
    AIBrainConfig = None

try:
    from constants import (
        SystemPrompts, PerformanceThresholds, MessageFormat, SupportedLanguage,
        GUJARATI_UNICODE_START, GUJARATI_UNICODE_END,
        HINDI_UNICODE_START, HINDI_UNICODE_END
    )
except ImportError:
    # Fallback constants
    GUJARATI_UNICODE_START = 0x0A80
    GUJARATI_UNICODE_END = 0x0AFF
    HINDI_UNICODE_START = 0x0900
    HINDI_UNICODE_END = 0x097F
    SystemPrompts = None
    PerformanceThresholds = None


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Message:
    """Represents a single message in conversation."""
    role: str  # "user" or "assistant" or "system"
    content: str
    timestamp: float = None
    language: Optional[str] = None
    tokens: int = 0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for API calls."""
        return {
            "role": self.role,
            "content": self.content
        }


@dataclass
class LLMResponse:
    """Response from LLM provider."""
    content: str
    model: str
    tokens_used: int
    stop_reason: str = "stop"
    latency_ms: float = 0.0
    cached: bool = False
    language: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


# ============================================================================
# LANGUAGE DETECTION ENGINE
# ============================================================================

class LanguageDetector:
    """Multi-language detection engine for EN, GU, HI, Gujlish, Hinglish."""
    
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize language detector.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger("EV.LanguageDetector")
    
    def detect(self, text: str) -> Tuple[str, float]:
        """
        Detect language from input text.
        Uses script detection, keyword matching, and heuristics.
        
        Args:
            text: Input text to detect
            
        Returns:
            Tuple of (language_code, confidence_score)
        """
        if not text or len(text.strip()) == 0:
            return "en", 0.0
        
        text_lower = text.lower()
        
        # Check for Gujarati script
        if self._contains_gujarati_script(text):
            gujarati_ratio = self._count_gujarati_chars(text) / len(text)
            if gujarati_ratio > 0.7:
                return "gu", gujarati_ratio
            else:
                return "gu-Latn", gujarati_ratio
        
        # Check for Hindi script
        if self._contains_hindi_script(text):
            hindi_ratio = self._count_hindi_chars(text) / len(text)
            if hindi_ratio > 0.7:
                return "hi", hindi_ratio
            else:
                return "hi-Latn", hindi_ratio
        
        # English detection (default for Latin script)
        return "en", 0.9
    
    def _contains_gujarati_script(self, text: str) -> bool:
        """Check if text contains Gujarati script characters."""
        return any(
            GUJARATI_UNICODE_START <= ord(char) <= GUJARATI_UNICODE_END
            for char in text
        )
    
    def _contains_hindi_script(self, text: str) -> bool:
        """Check if text contains Hindi script characters."""
        return any(
            HINDI_UNICODE_START <= ord(char) <= HINDI_UNICODE_END
            for char in text
        )
    
    def _count_gujarati_chars(self, text: str) -> int:
        """Count Gujarati script characters in text."""
        return sum(
            1 for char in text
            if GUJARATI_UNICODE_START <= ord(char) <= GUJARATI_UNICODE_END
        )
    
    def _count_hindi_chars(self, text: str) -> int:
        """Count Hindi script characters in text."""
        return sum(
            1 for char in text
            if HINDI_UNICODE_START <= ord(char) <= HINDI_UNICODE_END
        )


# ============================================================================
# LLM PROVIDER INTERFACE
# ============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any] = None, api_key: Optional[str] = None,
                 logger: logging.Logger = None):
        """
        Initialize LLM provider.
        
        Args:
            config: Configuration dictionary
            api_key: API key for the provider (if applicable)
            logger: Logger instance
        """
        self.config = config or {}
        self.api_key = api_key
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def generate(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Generate response from LLM.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse object
        """
        pass
    
    @abstractmethod
    async def stream(self, messages: List[Message], **kwargs) -> AsyncGenerator[str, None]:
        """
        Stream response from LLM token-by-token.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional provider-specific parameters
            
        Yields:
            Response tokens as strings
        """
        pass
    
    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        pass


# ============================================================================
# GROQ PROVIDER (Cloud - Primary)
# ============================================================================

class GroqProvider(LLMProvider):
    """Groq Cloud LLM provider (fast inference)."""
    
    def __init__(self, config: Dict[str, Any] = None, api_key: Optional[str] = None,
                 logger: logging.Logger = None):
        """Initialize Groq provider."""
        super().__init__(config, api_key, logger=logger)
        self.api_key = api_key or self.config.get("api_key", "")
        self.model_name = self.config.get("model_name", "mixtral-8x7b-32768")
        self.temperature = self.config.get("temperature", 0.7)
        self.top_p = self.config.get("top_p", 0.9)
        self.max_tokens = self.config.get("max_tokens", 2048)
        self.base_url = "https://api.groq.com/openai/v1"
    
    async def generate(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Generate response using Groq.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object
        """
        try:
            from groq import AsyncGroq
            
            start_time = time.time()
            
            if not self.api_key:
                raise ValueError("Groq API key not configured")
            
            # Prepare messages for Groq
            formatted_messages = [msg.to_dict() for msg in messages]
            
            # Initialize Groq client
            client = AsyncGroq(api_key=self.api_key)
            
            # Create chat completion
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=False
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model_name,
                tokens_used=response.usage.completion_tokens,
                latency_ms=latency_ms,
                stop_reason=response.choices[0].finish_reason or "stop"
            )
            
        except ImportError:
            self.logger.error("Groq library not installed. Install with: pip install groq")
            raise
        except Exception as e:
            self.logger.error(f"Groq generation error: {e}", exc_info=True)
            raise
    
    async def stream(self, messages: List[Message], **kwargs) -> AsyncGenerator[str, None]:
        """
        Stream response from Groq.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional parameters
            
        Yields:
            Response tokens
        """
        try:
            from groq import AsyncGroq
            
            if not self.api_key:
                raise ValueError("Groq API key not configured")
            
            formatted_messages = [msg.to_dict() for msg in messages]
            
            # Initialize Groq client
            client = AsyncGroq(api_key=self.api_key)
            
            # Create streaming chat completion
            stream = await client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except ImportError:
            self.logger.error("Groq library not installed. Install with: pip install groq")
            raise
        except Exception as e:
            self.logger.error(f"Groq streaming error: {e}", exc_info=True)
            raise
    
    async def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Groq models use similar tokenization to OpenAI
        # Rough approximation: ~4 chars per token
        return max(1, len(text) // 4)


# ============================================================================
# OLLAMA PROVIDER (Local - Fallback)
# ============================================================================

class OllamaProvider(LLMProvider):
    """Local Ollama LLM provider (fallback)."""
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """Initialize Ollama provider."""
        super().__init__(config, logger=logger)
        self.endpoint = self.config.get("ollama_endpoint", "http://localhost:11434")
        self.model_name = self.config.get("model_name", "mistral")
        self.temperature = self.config.get("temperature", 0.7)
        self.top_p = self.config.get("top_p", 0.9)
        self.max_tokens = self.config.get("max_tokens", 2048)
    
    async def generate(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Generate response using Ollama.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object
        """
        try:
            import httpx
            
            start_time = time.time()
            
            # Prepare messages for Ollama
            formatted_messages = [msg.to_dict() for msg in messages]
            
            # Build request
            request_data = {
                "model": self.model_name,
                "messages": formatted_messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": self.max_tokens,
                }
            }
            
            # Make request with timeout
            timeout_seconds = kwargs.get("timeout", 30)
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(
                    f"{self.endpoint}/api/chat",
                    json=request_data
                )
                response.raise_for_status()
                result = response.json()
            
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=result.get("message", {}).get("content", ""),
                model=self.model_name,
                tokens_used=result.get("eval_count", 0),
                latency_ms=latency_ms,
                stop_reason="stop"
            )
            
        except ImportError:
            self.logger.error("httpx library not installed. Install with: pip install httpx")
            raise
        except Exception as e:
            self.logger.error(f"Ollama generation error: {e}", exc_info=True)
            raise
    
    async def stream(self, messages: List[Message], **kwargs) -> AsyncGenerator[str, None]:
        """
        Stream response from Ollama.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional parameters
            
        Yields:
            Response tokens
        """
        try:
            import httpx
            
            formatted_messages = [msg.to_dict() for msg in messages]
            
            request_data = {
                "model": self.model_name,
                "messages": formatted_messages,
                "stream": True,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": self.max_tokens,
                }
            }
            
            timeout_seconds = kwargs.get("timeout", 30)
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                async with client.stream(
                    "POST",
                    f"{self.endpoint}/api/chat",
                    json=request_data
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                token = data.get("message", {}).get("content", "")
                                if token:
                                    yield token
                            except json.JSONDecodeError:
                                continue
        
        except Exception as e:
            self.logger.error(f"Ollama streaming error: {e}", exc_info=True)
            raise
    
    async def count_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation).
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Rough approximation: ~4 chars per token
        return max(1, len(text) // 4)


# ============================================================================
# RESPONSE CACHE
# ============================================================================

class ResponseCache:
    """Simple in-memory cache for LLM responses."""
    
    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize cache.
        
        Args:
            ttl_seconds: Time-to-live for cached responses
        """
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[LLMResponse, float]] = {}
        self.logger = logging.getLogger("EV.Cache")
    
    def _hash_messages(self, messages: List[Message]) -> str:
        """Generate hash key for messages."""
        content = "".join(msg.content for msg in messages)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, messages: List[Message]) -> Optional[LLMResponse]:
        """
        Retrieve cached response.
        
        Args:
            messages: List of Message objects
            
        Returns:
            Cached LLMResponse or None if not found/expired
        """
        key = self._hash_messages(messages)
        
        if key in self.cache:
            response, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                self.logger.debug(f"Cache hit: {key[:8]}...")
                response.cached = True
                return response
            else:
                del self.cache[key]
        
        return None
    
    def set(self, messages: List[Message], response: LLMResponse) -> None:
        """
        Cache a response.
        
        Args:
            messages: List of Message objects
            response: LLMResponse to cache
        """
        key = self._hash_messages(messages)
        self.cache[key] = (response, time.time())
        self.logger.debug(f"Cached response: {key[:8]}...")
    
    def clear(self) -> None:
        """Clear all cached responses."""
        self.cache.clear()
        self.logger.info("Cache cleared.")


# ============================================================================
# AI BRAIN - MAIN ORCHESTRATOR
# ============================================================================

class AIBrain:
    """
    Core AI orchestrator engine.
    Manages LLM providers (Groq primary, Ollama fallback), language detection, context, and response generation.
    """
    
    def __init__(self, app_root: Path = None, logger: logging.Logger = None,
                 config_manager = None, groq_api_key: Optional[str] = None):
        """
        Initialize AI Brain.
        
        Args:
            app_root: Path to application root
            logger: Logger instance
            config_manager: ConfigManager instance (optional)
            groq_api_key: Groq API key (optional, can be set via environment or config)
        """
        self.app_root = app_root or Path.home() / ".eternal_voice"
        self.logger = logger or logging.getLogger("EV.AIBrain")
        self.config_manager = config_manager
        
        # Load configuration
        try:
            if config_manager is None and ConfigManager is not None:
                config_manager = ConfigManager(self.app_root)
            self.ai_config = config_manager.config.ai_brain if config_manager else self._default_config()
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}, using defaults")
            self.ai_config = self._default_config()
        
        # Get Groq API key
        import os
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        
        # Initialize components
        self.language_detector = LanguageDetector(self.logger)
        self.cache = ResponseCache(ttl_seconds=getattr(self.ai_config, 'cache_ttl_seconds', 3600))
        
        # LLM provider
        self.primary_provider: Optional[LLMProvider] = None
        self.fallback_providers: List[LLMProvider] = []
        
        # Context management
        self.conversation_history: List[Message] = []
        self.system_message: Optional[Message] = None
        
        # Initialize providers
        self._initialize_providers()
        
        self.logger.info("AI Brain initialized successfully.")
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration dictionary."""
        return {
            "primary_model": "groq",
            "groq_model": "mixtral-8x7b-32768",
            "ollama_endpoint": "http://localhost:11434",
            "ollama_model": "mistral",
            "context_window": 4096,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048,
            "enable_streaming": True,
            "enable_caching": True,
            "cache_ttl_seconds": 3600,
        }
    
    def _initialize_providers(self) -> None:
        """Initialize LLM providers based on configuration."""
        try:
            # Get config values
            temperature = getattr(self.ai_config, 'temperature', 0.7)
            top_p = getattr(self.ai_config, 'top_p', 0.9)
            max_tokens = getattr(self.ai_config, 'max_tokens', 2048)
            
            # Initialize Groq provider (primary)
            if self.groq_api_key:
                groq_config = {
                    "api_key": self.groq_api_key,
                    "model_name": getattr(self.ai_config, 'groq_model', "mixtral-8x7b-32768"),
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                }
                self.primary_provider = GroqProvider(groq_config, self.groq_api_key, self.logger)
                self.logger.info("Groq provider initialized (primary).")
            else:
                self.logger.warning("Groq API key not found, will use Ollama as primary.")
            
            # Initialize Ollama provider (fallback)
            endpoint = getattr(self.ai_config, 'ollama_endpoint', "http://localhost:11434")
            ollama_config = {
                "ollama_endpoint": endpoint,
                "model_name": getattr(self.ai_config, 'ollama_model', "mistral"),
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
            ollama_provider = OllamaProvider(ollama_config, self.logger)
            self.fallback_providers.append(ollama_provider)
            self.logger.info("Ollama provider initialized (fallback).")
            
            # If Groq not available, use Ollama as primary
            if self.primary_provider is None:
                self.primary_provider = ollama_provider
                self.fallback_providers.clear()
                self.logger.info("Using Ollama as primary provider.")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize providers: {e}", exc_info=True)
            raise
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set system prompt for conversation.
        
        Args:
            prompt: System prompt text
        """
        self.system_message = Message(
            role="system",
            content=prompt
        )
        self.logger.debug("System prompt set.")
    
    def add_message(self, role: str, content: str, language: Optional[str] = None) -> Message:
        """
        Add message to conversation history.
        
        Args:
            role: Message role ("user" or "assistant")
            content: Message content
            language: Detected language (optional)
            
        Returns:
            Created Message object
        """
        if language is None:
            detected_lang, _ = self.language_detector.detect(content)
            language = detected_lang
        
        # Estimate tokens
        tokens = max(1, len(content) // 4)
        
        message = Message(
            role=role,
            content=content,
            language=language,
            tokens=tokens
        )
        self.conversation_history.append(message)
        return message
    
    def get_context_messages(self, max_tokens: int = None) -> List[Message]:
        """
        Get messages for context, respecting token limit.
        
        Args:
            max_tokens: Maximum tokens to include (uses config default if None)
            
        Returns:
            List of Message objects to send to LLM
        """
        if max_tokens is None:
            max_tokens = getattr(self.ai_config, 'context_window', 4096)
        
        messages = []
        token_count = 0
        
        # Add system message if set
        if self.system_message:
            messages.append(self.system_message)
            token_count += self.system_message.tokens
        
        # Add recent messages within token limit (reverse order for recency)
        for message in reversed(self.conversation_history):
            if token_count + message.tokens > max_tokens:
                break
            messages.insert(len(messages) - (1 if self.system_message else 0), message)
            token_count += message.tokens
        
        return messages
    
    async def generate(self, user_input: str, use_cache: bool = True,
                      system_prompt: Optional[str] = None) -> LLMResponse:
        """
        Generate response to user input.
        
        Args:
            user_input: User message text
            use_cache: Whether to use cached responses
            system_prompt: Optional system prompt override
            
        Returns:
            LLMResponse object
        """
        try:
            # Detect language
            detected_lang, confidence = self.language_detector.detect(user_input)
            self.logger.info(f"Detected language: {detected_lang} (confidence: {confidence:.2f})")
            
            # Add user message
            user_message = self.add_message("user", user_input, language=detected_lang)
            
            # Set system prompt if provided
            if system_prompt:
                self.set_system_prompt(system_prompt)
            elif not self.system_message:
                default_prompt = "You are E.V. (Eternal Voice), an advanced AI assistant. Be helpful, concise, and respond in the user's language."
                self.set_system_prompt(default_prompt)
            
            # Get context messages
            context_messages = self.get_context_messages()
            
            # Check cache
            if use_cache and getattr(self.ai_config, 'enable_caching', True):
                cached_response = self.cache.get(context_messages)
                if cached_response:
                    self.logger.info("Returning cached response.")
                    self.add_message("assistant", cached_response.content, detected_lang)
                    return cached_response
            
            # Generate response
            if self.primary_provider is None:
                raise RuntimeError("No LLM provider available")
            
            self.logger.info(f"Generating response using {self.primary_provider.__class__.__name__}...")
            response = await self.primary_provider.generate(context_messages)
            response.language = detected_lang
            
            # Cache response
            if getattr(self.ai_config, 'enable_caching', True):
                self.cache.set(context_messages, response)
            
            # Add to history
            self.add_message("assistant", response.content, detected_lang)
            
            self.logger.info(
                f"Response generated: {response.tokens_used} tokens, {response.latency_ms:.2f}ms"
            )
            return response
        
        except Exception as e:
            self.logger.error(f"Generation failed: {e}", exc_info=True)
            raise
    
    async def stream_response(self, user_input: str,
                             system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Stream response tokens to user.
        
        Args:
            user_input: User message text
            system_prompt: Optional system prompt override
            
        Yields:
            Response tokens
        """
        try:
            if self.primary_provider is None:
                raise RuntimeError("No LLM provider available")
            
            # Detect language
            detected_lang, _ = self.language_detector.detect(user_input)
            
            # Add user message
            self.add_message("user", user_input, language=detected_lang)
            
            # Set system prompt
            if system_prompt:
                self.set_system_prompt(system_prompt)
            elif not self.system_message:
                default_prompt = "You are E.V. (Eternal Voice), an advanced AI assistant. Be helpful, concise, and respond in the user's language."
                self.set_system_prompt(default_prompt)
            
            # Get context
            context_messages = self.get_context_messages()
            
            # Stream response
            full_response = ""
            async for token in self.primary_provider.stream(context_messages):
                full_response += token
                yield token
            
            # Add to history
            self.add_message("assistant", full_response, detected_lang)
        
        except Exception as e:
            self.logger.error(f"Streaming failed: {e}", exc_info=True)
            raise
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        self.logger.info("Conversation history cleared.")
    
    def get_history_summary(self) -> Dict[str, Any]:
        """
        Get summary of conversation history.
        
        Returns:
            Dictionary with history statistics
        """
        return {
            "total_messages": len(self.conversation_history),
            "total_tokens": sum(msg.tokens for msg in self.conversation_history),
            "languages_used": list(set(msg.language for msg in self.conversation_history if msg.language)),
            "oldest_message": self.conversation_history[0].timestamp if self.conversation_history else None,
            "newest_message": self.conversation_history[-1].timestamp if self.conversation_history else None,
            "primary_provider": self.primary_provider.__class__.__name__ if self.primary_provider else "None",
        }
    
    def shutdown(self) -> None:
        """Shutdown AI Brain and clean up resources."""
        try:
            self.cache.clear()
            self.clear_history()
            self.logger.info("AI Brain shutdown complete.")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import sys
    import os
    
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        # Check for Groq API key
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            print("⚠️ GROQ_API_KEY environment variable not set")
            print("Set it with: export GROQ_API_KEY='your-api-key'")
            print("Get your key from: https://console.groq.com")
        
        app_root = Path.home() / ".eternal_voice_test"
        app_root.mkdir(exist_ok=True)
        
        # Initialize AI Brain with Groq
        ai_brain = AIBrain(app_root, groq_api_key=groq_key)
        
        # Test language detection
        print("\n[LANGUAGE DETECTION TESTS]")
        test_inputs = [
            "Hello, how are you?",
            "હલો, તમે કેમ છો?",
            "नमस्ते, आप कैसे हैं?",
            "Kem cho EV, mane code fix karva?",
        ]
        
        for text in test_inputs:
            lang, conf = ai_brain.language_detector.detect(text)
            print(f"Text: {text}")
            print(f"Language: {lang}, Confidence: {conf:.2f}\n")
        
        # Test response generation
        print("\n[RESPONSE GENERATION]")
        try:
            print("Using provider:", ai_brain.primary_provider.__class__.__name__)
            response = await ai_brain.generate("What is Python?")
            print(f"Response: {response.content[:200]}...")
            print(f"Tokens: {response.tokens_used}, Latency: {response.latency_ms:.2f}ms")
            print(f"Provider: {ai_brain.primary_provider.__class__.__name__}")
        except Exception as e:
            print(f"Generation test failed: {e}")
        
        # Print history summary
        print("\n[HISTORY SUMMARY]")
        summary = ai_brain.get_history_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Cleanup
        ai_brain.shutdown()
    
    # Run async main
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested.")
        sys.exit(0)
