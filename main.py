#!/usr/bin/env python3
"""
E.V. (Eternal Voice) - Main Entry Point
Android 11+ AI Assistant Application
File 01: Application Bootstrap, Kivy Loop & Crash Recovery

Handles:
- Process initialization & environment setup
- Crash logging & recovery mechanism
- Kivy event loop bootstrap
- Signal handling & graceful shutdown
"""

import os
import sys
import logging
import traceback
import signal
from typing import Optional, NoReturn
from datetime import datetime
from pathlib import Path
import threading
import atexit

# Platform detection
IS_ANDROID = False
try:
    from jnius import autoclass
    from android.permissions import request_permissions, Permission, check_permission
    IS_ANDROID = True
except ImportError:
    IS_ANDROID = False

# Core imports (will be initialized after config)
os.environ["KIVY_WINDOW"] = "pygame"
os.environ["KIVY_GL_BACKEND"] = "gl"

import kivy
from kivy.app import App
from kivy.core.window import Window
from kivy.logger import Logger as KivyLogger


# ============================================================================
# CRASH RECOVERY & LOGGING SYSTEM
# ============================================================================

class CrashHandler:
    """Unified crash logging and recovery system."""
    
    def __init__(self, app_root: Path):
        self.app_root = app_root
        self.crash_dir = app_root / "crash_logs"
        self.crash_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Configure root logger with file + console handlers."""
        logger = logging.getLogger("EV")
        logger.setLevel(logging.DEBUG)
        
        # Console handler (INFO level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_fmt)
        
        # File handler (DEBUG level)
        log_file = self.crash_dir / f"ev_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_fmt)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def log_crash(self, exc_type, exc_value, exc_traceback) -> None:
        """Log unhandled exception to crash file."""
        crash_file = self.crash_dir / f"crash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        try:
            with open(crash_file, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"E.V. CRASH REPORT - {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Exception Type: {exc_type.__name__}\n")
                f.write(f"Exception Message: {exc_value}\n\n")
                f.write("Traceback:\n")
                f.write("".join(traceback.format_tb(exc_traceback)))
                f.write("\n" + "=" * 80 + "\n")
            
            self.logger.critical(
                f"CRASH LOGGED: {exc_type.__name__}: {exc_value}",
                exc_info=(exc_type, exc_value, exc_traceback)
            )
        except Exception as e:
            self.logger.error(f"Failed to write crash log: {e}")
    
    def safe_cleanup(self) -> None:
        """Perform safe shutdown operations."""
        self.logger.info("E.V. Shutting down gracefully...")
        # Additional cleanup will be added by dependent systems
        self.logger.info("E.V. Shutdown complete.")


# ============================================================================
# ENVIRONMENT INITIALIZATION
# ============================================================================

def initialize_environment() -> tuple[Path, CrashHandler]:
    """
    Initialize application environment paths and crash handler.
    
    Returns:
        tuple: (app_root_path, crash_handler)
    """
    # Determine app root
    if IS_ANDROID:
        from android.storage import app_storage_path
        app_root = Path(app_storage_path())
    else:
        app_root = Path.home() / ".eternal_voice"
    
    app_root.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (app_root / "data").mkdir(exist_ok=True)
    (app_root / "models").mkdir(exist_ok=True)
    (app_root / "cache").mkdir(exist_ok=True)
    (app_root / "logs").mkdir(exist_ok=True)
    
    # Initialize crash handler
    crash_handler = CrashHandler(app_root)
    crash_handler.logger.info(f"E.V. Initialized at {app_root}")
    
    return app_root, crash_handler


# ============================================================================
# ANDROID PERMISSION INITIALIZATION
# ============================================================================

def request_android_permissions(crash_handler: CrashHandler) -> bool:
    """
    Request runtime permissions on Android 11+.
    
    Args:
        crash_handler: CrashHandler instance for logging
        
    Returns:
        bool: True if all permissions granted or not on Android
    """
    if not IS_ANDROID:
        return True
    
    required_permissions = [
        Permission.MICROPHONE,
        Permission.CAMERA,
        Permission.INTERNET,
        Permission.ACCESS_FINE_LOCATION,
        Permission.READ_EXTERNAL_STORAGE,
        Permission.WRITE_EXTERNAL_STORAGE,
    ]
    
    try:
        crash_handler.logger.info("Requesting Android runtime permissions...")
        request_permissions(required_permissions)
        crash_handler.logger.info("Permission request initiated.")
        return True
    except Exception as e:
        crash_handler.logger.warning(f"Permission request error (may be normal): {e}")
        return False


# ============================================================================
# KIVY WINDOW CONFIGURATION
# ============================================================================

def configure_kivy_window() -> None:
    """Configure Kivy window properties and rendering options."""
    if IS_ANDROID:
        # Android display configuration
        Window.size = (1080, 2340)  # 1:19.5 typical Android ratio
        Window.orientation = "portrait"
    else:
        # Desktop preview mode
        Window.size = (540, 1170)  # Scale down for testing
    
    # Rendering configuration
    Window.clearcolor = (0, 0, 0, 1)  # Pure black background
    
    # Disable default touch effects
    Window.borderless = False


# ============================================================================
# APPLICATION LAUNCHER
# ============================================================================

def main() -> NoReturn:
    """
    Main entry point for E.V. application.
    Initializes all systems and launches Kivy event loop.
    """
    # Step 1: Initialize environment
    try:
        app_root, crash_handler = initialize_environment()
    except Exception as e:
        print(f"FATAL: Environment initialization failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    
    # Step 2: Configure exception handler
    sys.excepthook = lambda exc_type, exc_value, exc_tb: crash_handler.log_crash(
        exc_type, exc_value, exc_tb
    )
    
    # Register cleanup
    atexit.register(crash_handler.safe_cleanup)
    
    # Step 3: Signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        crash_handler.logger.info(f"Received signal {signum}, initiating shutdown...")
        crash_handler.safe_cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Step 4: Request Android permissions
    if IS_ANDROID:
        try:
            request_android_permissions(crash_handler)
        except Exception as e:
            crash_handler.logger.error(f"Permission request failed: {e}")
    
    # Step 5: Configure Kivy
    try:
        configure_kivy_window()
        crash_handler.logger.info("Kivy window configured.")
    except Exception as e:
        crash_handler.logger.critical(f"Kivy configuration failed: {e}")
        crash_handler.safe_cleanup()
        sys.exit(1)
    
    # Step 6: Import and launch main app
    try:
        # Delayed import to avoid circular dependencies
        from app import EternalVoiceApp
        
        crash_handler.logger.info("Launching E.V. Application...")
        app = EternalVoiceApp(app_root=app_root, crash_handler=crash_handler)
        app.run()
    
    except Exception as e:
        crash_handler.logger.critical(f"Application launch failed: {e}")
        crash_handler.log_crash(type(e), e, e.__traceback__)
        crash_handler.safe_cleanup()
        sys.exit(1)
    
    # Unreachable but satisfies type hints
    sys.exit(0)


# ============================================================================
# SCRIPT GUARD & EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()
