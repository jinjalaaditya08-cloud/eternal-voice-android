#!/usr/bin/env python3
"""
E.V. (Eternal Voice) - System Constants & UI Design Tokens
File 04: Application-wide constants, color tokens, state enums, and UI specifications

Handles:
- Design tokens (colors, typography, spacing)
- Application state enums
- Status indicators and their visual representations
- System constants and thresholds
- Language & locale constants
"""

from enum import Enum
from typing import Dict, Tuple
from dataclasses import dataclass


# ============================================================================
# COLOR PALETTE - DESIGN TOKENS
# ============================================================================

class ColorPalette:
    """Glassmorphism color palette in RGBA format (0-1 range)."""
    
    # Background & Surfaces
    BACKGROUND = (0.0, 0.0, 0.0, 1.0)  # Pure black #000000
    SURFACE = (0.07, 0.09, 0.12, 0.65)  # #12161F with 65% opacity
    SURFACE_LIGHT = (0.12, 0.14, 0.18, 0.8)  # Lighter variant
    SURFACE_VARIANT = (0.05, 0.06, 0.09, 0.5)  # Subtle variant
    
    # Primary & Secondary Accents
    PRIMARY = (0.0, 0.94, 1.0, 1.0)  # Electric cyan #00F0FF
    PRIMARY_DARK = (0.0, 0.75, 0.8, 1.0)  # Darker cyan variant
    PRIMARY_LIGHT = (0.3, 1.0, 1.0, 1.0)  # Lighter cyan variant
    
    SECONDARY = (0.0, 1.0, 0.4, 1.0)  # Neon green #00FF66
    SECONDARY_DARK = (0.0, 0.8, 0.32, 1.0)  # Darker green
    
    # Text Colors
    TEXT_PRIMARY = (1.0, 1.0, 1.0, 1.0)  # White
    TEXT_SECONDARY = (0.54, 0.6, 0.68, 1.0)  # Light gray #8A99AD
    TEXT_TERTIARY = (0.4, 0.45, 0.52, 1.0)  # Muted gray
    TEXT_DISABLED = (0.3, 0.3, 0.3, 1.0)  # Disabled gray
    
    # Status Colors
    SUCCESS = (0.0, 1.0, 0.4, 1.0)  # Green #00FF66
    WARNING = (1.0, 0.84, 0.0, 1.0)  # Yellow #FFD700
    ERROR = (1.0, 0.3, 0.3, 1.0)  # Red
    INFO = (0.0, 0.94, 1.0, 1.0)  # Cyan
    
    # Borders & Dividers
    BORDER = (0.0, 0.94, 1.0, 0.3)  # Cyan with 30% opacity
    DIVIDER = (0.2, 0.2, 0.2, 0.5)  # Subtle divider
    
    # Additional Utilities
    TRANSPARENT = (0.0, 0.0, 0.0, 0.0)
    OVERLAY_DARK = (0.0, 0.0, 0.0, 0.7)  # Dark overlay
    OVERLAY_LIGHT = (1.0, 1.0, 1.0, 0.1)  # Light overlay


# ============================================================================
# TYPOGRAPHY SPECIFICATIONS
# ============================================================================

class Typography:
    """Font families and size specifications."""
    
    # Font Families
    PRIMARY_FONT = "Roboto"
    MONO_FONT = "RobotoMono"
    FALLBACK_FONT = "DejaVuSans"
    
    # Font Sizes (in sp - scale-independent pixels)
    HEADING_1 = "32sp"
    HEADING_2 = "28sp"
    HEADING_3 = "24sp"
    HEADING_4 = "20sp"
    
    BODY_LARGE = "16sp"
    BODY_MEDIUM = "14sp"
    BODY_SMALL = "12sp"
    
    LABEL_LARGE = "14sp"
    LABEL_MEDIUM = "12sp"
    LABEL_SMALL = "11sp"
    
    CAPTION = "10sp"
    
    # Font Weights (Kivy uses 0.0 to 1.0 scale, but we specify as integers for clarity)
    WEIGHT_LIGHT = 300
    WEIGHT_REGULAR = 400
    WEIGHT_MEDIUM = 500
    WEIGHT_SEMIBOLD = 600
    WEIGHT_BOLD = 700


# ============================================================================
# SPACING & SIZING
# ============================================================================

class Spacing:
    """Consistent spacing values for layout."""
    
    # Density Independent Pixels (dp)
    EXTRA_SMALL = "4dp"
    SMALL = "8dp"
    MEDIUM = "16dp"
    LARGE = "24dp"
    EXTRA_LARGE = "32dp"
    EXTRA_EXTRA_LARGE = "48dp"
    
    # Border Radius
    CORNER_NONE = "0dp"
    CORNER_SMALL = "4dp"
    CORNER_MEDIUM = "8dp"
    CORNER_LARGE = "12dp"
    CORNER_EXTRA_LARGE = "16dp"
    CORNER_FULL = "999dp"  # For circular elements
    
    # Elevation (shadow blur radius)
    ELEVATION_NONE = 0
    ELEVATION_LOW = 2
    ELEVATION_MEDIUM = 4
    ELEVATION_HIGH = 8
    ELEVATION_EXTRA_HIGH = 12


# ============================================================================
# ANIMATION & TRANSITION TIMING
# ============================================================================

class AnimationTiming:
    """Animation duration constants."""
    
    # Duration in seconds
    DURATION_INSTANT = 0.0
    DURATION_FAST = 0.2
    DURATION_NORMAL = 0.4
    DURATION_SLOW = 0.6
    DURATION_EXTRA_SLOW = 1.0
    
    # Easing functions (Kivy easing names)
    EASING_LINEAR = "linear"
    EASING_IN_QUAD = "in_quad"
    EASING_OUT_QUAD = "out_quad"
    EASING_IN_OUT_QUAD = "in_out_quad"
    EASING_IN_CUBIC = "in_cubic"
    EASING_OUT_CUBIC = "out_cubic"


# ============================================================================
# APPLICATION STATE ENUMS
# ============================================================================

class AppState(Enum):
    """Application lifecycle states."""
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class ScreenState(Enum):
    """Screen navigation states."""
    CHAT = "chat"
    SETTINGS = "settings"
    DEBUG = "debug"
    SPLASH = "splash"


class VoiceState(Enum):
    """Voice I/O pipeline states."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


class AIState(Enum):
    """AI processing states."""
    IDLE = "idle"
    THINKING = "thinking"
    GENERATING = "generating"
    EXECUTING = "executing"
    RESEARCHING = "researching"
    ERROR = "error"


class CodeExecutionState(Enum):
    """Code generation & execution states."""
    IDLE = "idle"
    UNDERSTANDING = "understanding"
    PLANNING = "planning"
    GENERATING = "generating"
    SYNTAX_CHECK = "syntax_check"
    TESTING = "testing"
    REPAIRING = "repairing"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# STATUS INDICATORS
# ============================================================================

class StatusIndicator(Enum):
    """Application status indicators with visual properties."""
    
    ONLINE = "online"
    OFFLINE = "offline"
    THINKING = "thinking"
    LISTENING = "listening"
    SPEAKING = "speaking"
    RESEARCHING = "researching"
    CODING = "coding"
    TESTING = "testing"
    ERROR = "error"
    LOADING = "loading"


# Status to Color mapping
STATUS_COLORS: Dict[StatusIndicator, Tuple[float, float, float, float]] = {
    StatusIndicator.ONLINE: ColorPalette.SUCCESS,
    StatusIndicator.OFFLINE: (0.5, 0.5, 0.5, 1.0),
    StatusIndicator.THINKING: ColorPalette.PRIMARY,
    StatusIndicator.LISTENING: ColorPalette.PRIMARY,
    StatusIndicator.SPEAKING: ColorPalette.SECONDARY,
    StatusIndicator.RESEARCHING: ColorPalette.WARNING,
    StatusIndicator.CODING: ColorPalette.PRIMARY,
    StatusIndicator.TESTING: ColorPalette.WARNING,
    StatusIndicator.ERROR: ColorPalette.ERROR,
    StatusIndicator.LOADING: ColorPalette.PRIMARY,
}

# Status to Text Label mapping
STATUS_LABELS: Dict[StatusIndicator, str] = {
    StatusIndicator.ONLINE: "Online",
    StatusIndicator.OFFLINE: "Offline",
    StatusIndicator.THINKING: "Thinking",
    StatusIndicator.LISTENING: "Listening",
    StatusIndicator.SPEAKING: "Speaking",
    StatusIndicator.RESEARCHING: "Researching",
    StatusIndicator.CODING: "Coding",
    StatusIndicator.TESTING: "Testing",
    StatusIndicator.ERROR: "Error",
    StatusIndicator.LOADING: "Loading",
}


# ============================================================================
# LANGUAGE & LOCALE CONSTANTS
# ============================================================================

class SupportedLanguage(Enum):
    """Supported languages and scripts."""
    ENGLISH = "en"
    ENGLISH_US = "en-US"
    ENGLISH_UK = "en-GB"
    
    GUJARATI = "gu"
    GUJARATI_INDIA = "gu-IN"
    
    HINDI = "hi"
    HINDI_INDIA = "hi-IN"
    
    GUJLISH = "gu-Latn"  # Gujarati in Latin script
    HINGLISH = "hi-Latn"  # Hindi in Latin script


# Language to Name mapping
LANGUAGE_NAMES: Dict[SupportedLanguage, str] = {
    SupportedLanguage.ENGLISH: "English",
    SupportedLanguage.ENGLISH_US: "English (US)",
    SupportedLanguage.ENGLISH_UK: "English (UK)",
    SupportedLanguage.GUJARATI: "ગુજરાતી",
    SupportedLanguage.GUJARATI_INDIA: "ગુજરાતી (ભારત)",
    SupportedLanguage.HINDI: "हिन्दी",
    SupportedLanguage.HINDI_INDIA: "हिन्दी (भारत)",
    SupportedLanguage.GUJLISH: "Gujlish",
    SupportedLanguage.HINGLISH: "Hinglish",
}

# Script detection patterns
GUJARATI_UNICODE_START = 0x0A80
GUJARATI_UNICODE_END = 0x0AFF

HINDI_UNICODE_START = 0x0900
HINDI_UNICODE_END = 0x097F


# ============================================================================
# ANDROID & HARDWARE CONSTANTS
# ============================================================================

class AndroidAPI:
    """Android API level constants."""
    API_30 = 30  # Android 11
    API_31 = 31  # Android 12
    API_32 = 32  # Android 12L
    API_33 = 33  # Android 13
    API_34 = 34  # Android 14
    
    MIN_API_LEVEL = API_30
    TARGET_API_LEVEL = API_34


class Permissions:
    """Android runtime permissions."""
    # Audio
    RECORD_AUDIO = "android.permission.RECORD_AUDIO"
    
    # Camera
    CAMERA = "android.permission.CAMERA"
    
    # Location
    ACCESS_FINE_LOCATION = "android.permission.ACCESS_FINE_LOCATION"
    ACCESS_COARSE_LOCATION = "android.permission.ACCESS_COARSE_LOCATION"
    
    # Storage
    READ_EXTERNAL_STORAGE = "android.permission.READ_EXTERNAL_STORAGE"
    WRITE_EXTERNAL_STORAGE = "android.permission.WRITE_EXTERNAL_STORAGE"
    
    # Network
    INTERNET = "android.permission.INTERNET"
    ACCESS_NETWORK_STATE = "android.permission.ACCESS_NETWORK_STATE"
    
    # Notifications
    POST_NOTIFICATIONS = "android.permission.POST_NOTIFICATIONS"


# ============================================================================
# MEMORY & PERFORMANCE THRESHOLDS
# ============================================================================

class PerformanceThresholds:
    """Performance-related constants and thresholds."""
    
    # Memory limits (in MB)
    MAX_MEMORY_MB = 512
    CACHE_SIZE_MB = 256
    DATABASE_SIZE_MB = 1024
    
    # Token limits
    MAX_CONTEXT_TOKENS = 4096
    MAX_OUTPUT_TOKENS = 2048
    MAX_MEMORY_TOKENS = 8000
    
    # Timeouts (in seconds)
    API_TIMEOUT = 30
    STT_TIMEOUT = 15
    TTS_TIMEOUT = 10
    VOICE_IDLE_TIMEOUT = 5
    
    # Batch processing
    MAX_BATCH_SIZE = 10
    MAX_CONCURRENT_REQUESTS = 2
    
    # Retry logic
    MAX_RETRIES = 3
    RETRY_BACKOFF_SECONDS = 2


# ============================================================================
# MESSAGE & RESPONSE FORMATTING
# ============================================================================

class MessageFormat:
    """Constants for message formatting and structure."""
    
    # Message roles
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_SYSTEM = "system"
    
    # Message types
    TYPE_TEXT = "text"
    TYPE_IMAGE = "image"
    TYPE_CODE = "code"
    TYPE_COMMAND = "command"
    TYPE_ERROR = "error"
    
    # Max lengths
    MAX_MESSAGE_LENGTH = 10000
    MAX_DISPLAY_MESSAGE_LENGTH = 5000


# ============================================================================
# SYSTEM PROMPT TEMPLATES
# ============================================================================

class SystemPrompts:
    """Pre-defined system prompts for different contexts."""
    
    MAIN_ASSISTANT = """You are E.V. (Eternal Voice), an advanced AI assistant running on Android 11+.
You are helpful, concise, and multilingual (English, Gujarati, Hindi, Gujlish, Hinglish).
Always respond in the same language/script the user used.
Be proactive, thoughtful, and respect user privacy."""
    
    CODE_GENERATOR = """You are an expert code generator. Generate production-grade code that is:
- Fully functional and complete
- Well-documented with docstrings
- Type-hinted (Python typing module)
- Error-handled (try/except/finally blocks)
- Optimized for performance
Always provide complete implementations, never use placeholders."""
    
    RESEARCH_AGENT = """You are a research assistant. When conducting research:
1. Search for multiple credible sources
2. Synthesize information from various sources
3. Cite sources appropriately
4. Highlight consensus and disagreements
5. Provide balanced perspectives"""
    
    VISION_ANALYZER = """You are analyzing images from an Android device camera.
Provide clear, actionable descriptions of what you see.
If the user asks "Aa shu che?" (What is this?), describe the image in detail.
When analyzing documents, extract text accurately using OCR."""


# ============================================================================
# ERROR MESSAGES & STRINGS
# ============================================================================

class ErrorMessages:
    """Standard error message templates."""
    
    PERMISSION_DENIED = "Permission denied. Please enable {permission} in settings."
    API_ERROR = "API Error: {error_code} - {error_message}"
    NETWORK_ERROR = "Network connection failed. Please check your internet."
    TIMEOUT_ERROR = "Operation timed out after {seconds} seconds."
    FILE_NOT_FOUND = "File not found: {file_path}"
    INVALID_INPUT = "Invalid input: {reason}"
    INTERNAL_ERROR = "Internal error occurred. Please try again later."
    MODEL_NOT_LOADED = "AI model not loaded. Please wait for initialization."


# ============================================================================
# SUCCESS MESSAGES & CONFIRMATIONS
# ============================================================================

class SuccessMessages:
    """Standard success message templates."""
    
    PERMISSION_GRANTED = "Permission granted: {permission}"
    FILE_SAVED = "File saved successfully: {file_path}"
    OPERATION_COMPLETE = "Operation completed successfully."
    CODE_EXECUTED = "Code executed successfully."
    MODEL_LOADED = "AI model loaded successfully."


# ============================================================================
# UI COMPONENT DEFAULTS
# ============================================================================

@dataclass
class ComponentDefaults:
    """Default properties for UI components."""
    
    # Button
    BUTTON_MIN_WIDTH = "88dp"
    BUTTON_HEIGHT = "48dp"
    BUTTON_RADIUS = "8dp"
    
    # Card
    CARD_RADIUS = "12dp"
    CARD_ELEVATION = Spacing.ELEVATION_MEDIUM
    CARD_PADDING = Spacing.MEDIUM
    
    # Input Field
    INPUT_HEIGHT = "56dp"
    INPUT_PADDING = Spacing.MEDIUM
    
    # Dialog
    DIALOG_RADIUS = "16dp"
    DIALOG_MIN_WIDTH = "80%"
    DIALOG_MAX_WIDTH = "95%"
    
    # Text Field
    TEXT_FIELD_HEIGHT = "56dp"
    TEXT_FIELD_PADDING = "12dp"


# ============================================================================
# DEBUG & LOGGING CONSTANTS
# ============================================================================

class DebugConstants:
    """Debug and logging related constants."""
    
    # Log levels
    LOG_LEVEL_DEBUG = "DEBUG"
    LOG_LEVEL_INFO = "INFO"
    LOG_LEVEL_WARNING = "WARNING"
    LOG_LEVEL_ERROR = "ERROR"
    LOG_LEVEL_CRITICAL = "CRITICAL"
    
    # Log file settings
    MAX_LOG_FILE_SIZE_MB = 50
    LOG_RETENTION_DAYS = 30
    
    # Module names for logging
    MODULE_APP = "EV.App"
    MODULE_AI = "EV.AI"
    MODULE_VOICE = "EV.Voice"
    MODULE_MEMORY = "EV.Memory"
    MODULE_VISION = "EV.Vision"
    MODULE_CODE = "EV.Code"
    MODULE_RESEARCH = "EV.Research"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_status_color(status: StatusIndicator) -> Tuple[float, float, float, float]:
    """Get RGBA color tuple for a status indicator."""
    return STATUS_COLORS.get(status, ColorPalette.TEXT_SECONDARY)


def get_status_label(status: StatusIndicator) -> str:
    """Get human-readable label for a status indicator."""
    return STATUS_LABELS.get(status, "Unknown")


def is_gujarati_script(text: str) -> bool:
    """Check if text contains Gujarati script."""
    return any(GUJARATI_UNICODE_START <= ord(char) <= GUJARATI_UNICODE_END for char in text)


def is_hindi_script(text: str) -> bool:
    """Check if text contains Hindi script."""
    return any(HINDI_UNICODE_START <= ord(char) <= HINDI_UNICODE_END for char in text)


# ============================================================================
# EXAMPLE USAGE & VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("E.V. CONSTANTS & DESIGN TOKENS")
    print("=" * 60)
    
    # Test color palette
    print("\n[COLOR PALETTE]")
    print(f"Background: {ColorPalette.BACKGROUND}")
    print(f"Primary: {ColorPalette.PRIMARY}")
    print(f"Secondary: {ColorPalette.SECONDARY}")
    print(f"Error: {ColorPalette.ERROR}")
    
    # Test status indicators
    print("\n[STATUS INDICATORS]")
    for status in StatusIndicator:
        color = get_status_color(status)
        label = get_status_label(status)
        print(f"{status.value}: {label} -> {color}")
    
    # Test language detection
    print("\n[LANGUAGE DETECTION]")
    gujarati_text = "હલો"
    hindi_text = "नमस्ते"
    english_text = "Hello"
    
    print(f"'{gujarati_text}' is Gujarati: {is_gujarati_script(gujarati_text)}")
    print(f"'{hindi_text}' is Hindi: {is_hindi_script(hindi_text)}")
    print(f"'{english_text}' is Gujarati: {is_gujarati_script(english_text)}")
    print(f"'{english_text}' is Hindi: {is_hindi_script(english_text)}")
    
    # Test typography
    print("\n[TYPOGRAPHY]")
    print(f"Primary Font: {Typography.PRIMARY_FONT}")
    print(f"Heading 1: {Typography.HEADING_1}")
    print(f"Body Medium: {Typography.BODY_MEDIUM}")
    
    # Test spacing
    print("\n[SPACING]")
    print(f"Small: {Spacing.SMALL}")
    print(f"Medium: {Spacing.MEDIUM}")
    print(f"Large: {Spacing.LARGE}")
    
    print("\n" + "=" * 60)
