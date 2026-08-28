#!/usr/bin/env python3
"""
E.V. (Eternal Voice) - Main Application Class
File 02: KivyMD Application Lifecycle & Initialization

Handles:
- KivyMD MDApp configuration and lifecycle
- Screen manager initialization
- Theme and style application
- Memory & resource management
- Background service management
"""

import logging
from typing import Optional
from pathlib import Path
from enum import Enum

from kivy.app import App
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.core.text import LabelBase
from kivy.resources import resource_add_path

from kivymd.app import MDApp
from kivymd.uix.screenmanager import MDScreenManager
from kivymd.theming import ThemableBehavior
from kivymd.color_definitions import colors

# Platform-specific imports
IS_ANDROID = False
try:
    from jnius import autoclass, cast
    from android.runnable import run_on_ui_thread
    IS_ANDROID = True
except ImportError:
    IS_ANDROID = False


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class AppState(Enum):
    """Application lifecycle states."""
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class ThemeMode(Enum):
    """Supported theme modes."""
    DARK = "dark"
    GLASSMORPHISM = "glassmorphism"
    LIGHT = "light"


# ============================================================================
# DESIGN TOKEN SPECIFICATIONS
# ============================================================================

DESIGN_TOKENS = {
    "colors": {
        "background": (0, 0, 0, 1),  # Pure black #000000
        "surface": (0.07, 0.09, 0.12, 0.65),  # #12161F with 65% opacity
        "primary": (0, 0.94, 1, 1),  # Electric cyan #00F0FF
        "secondary": (0, 1, 0.4, 1),  # Neon green #00FF66
        "text_primary": (1, 1, 1, 1),  # White
        "text_secondary": (0.54, 0.6, 0.68, 1),  # #8A99AD
        "border": (0, 0.94, 1, 0.3),  # Cyan with 30% opacity
        "error": (1, 0.3, 0.3, 1),  # Red
        "success": (0, 1, 0.4, 1),  # Green
        "warning": (1, 0.84, 0, 1),  # Yellow
    },
    "typography": {
        "font_family": "Roboto",
        "heading_size": "24sp",
        "body_size": "14sp",
        "caption_size": "12sp",
    },
    "spacing": {
        "xs": "4dp",
        "sm": "8dp",
        "md": "16dp",
        "lg": "24dp",
        "xl": "32dp",
    },
}

STATUS_COLORS = {
    "ONLINE": DESIGN_TOKENS["colors"]["success"],
    "THINKING": DESIGN_TOKENS["colors"]["primary"],
    "LISTENING": DESIGN_TOKENS["colors"]["primary"],
    "SPEAKING": DESIGN_TOKENS["colors"]["secondary"],
    "RESEARCHING": DESIGN_TOKENS["colors"]["warning"],
    "CODING": DESIGN_TOKENS["colors"]["primary"],
    "TESTING": DESIGN_TOKENS["colors"]["warning"],
    "OFFLINE": (0.5, 0.5, 0.5, 1),
}


# ============================================================================
# ETERNAL VOICE APPLICATION CLASS
# ============================================================================

class EternalVoiceApp(MDApp):
    """
    Main KivyMD application class for E.V.
    Manages lifecycle, screens, theme, and resource management.
    """
    
    def __init__(self, app_root: Path, crash_handler, **kwargs):
        """
        Initialize E.V. Application.
        
        Args:
            app_root: Path to application root directory
            crash_handler: CrashHandler instance for logging
        """
        super().__init__(**kwargs)
        
        self.app_root = app_root
        self.crash_handler = crash_handler
        self.logger = logging.getLogger("EV.App")
        
        # Application state
        self.state = AppState.INITIALIZING
        self.theme_mode = ThemeMode.GLASSMORPHISM
        
        # Lazy-loaded systems (will be initialized in on_start)
        self.screen_manager: Optional[MDScreenManager] = None
        self.ai_brain = None
        self.memory_manager = None
        self.voice_manager = None
        self.background_service = None
        
        # Configuration
        self.title = "E.V. - Eternal Voice"
        self.icon = str(self.app_root / "assets" / "icon.png")
        
    # ========================================================================
    # LIFECYCLE HOOKS
    # ========================================================================
    
    def build(self) -> MDScreenManager:
        """
        Build and return the root widget (MDScreenManager).
        Called before on_start().
        
        Returns:
            MDScreenManager: Root widget for the application
        """
        try:
            self.logger.info("Building E.V. Application UI...")
            
            # Create screen manager
            self.screen_manager = MDScreenManager()
            
            # Apply theme
            self._apply_theme()
            
            # Load and register custom fonts
            self._load_fonts()
            
            self.logger.info("Application UI build complete.")
            return self.screen_manager
            
        except Exception as e:
            self.logger.critical(f"Failed to build application: {e}", exc_info=True)
            self.state = AppState.ERROR
            raise
    
    def on_start(self) -> None:
        """
        Initialize application systems after Kivy loop starts.
        This is where we load heavy systems asynchronously.
        """
        try:
            self.logger.info("Starting E.V. Application systems...")
            self.state = AppState.ACTIVE
            
            # Schedule initialization tasks
            Clock.schedule_once(self._initialize_ai_brain, 0.5)
            Clock.schedule_once(self._initialize_memory_manager, 1.0)
            Clock.schedule_once(self._initialize_voice_manager, 1.5)
            Clock.schedule_once(self._load_screens, 2.0)
            Clock.schedule_once(self._start_background_service, 2.5)
            
            self.logger.info("E.V. Application started successfully.")
            
        except Exception as e:
            self.logger.critical(f"Failed to start application: {e}", exc_info=True)
            self.state = AppState.ERROR
    
    def on_stop(self) -> bool:
        """
        Called when application is about to stop.
        Perform cleanup and shutdown operations.
        
        Returns:
            bool: True to allow app to close, False to prevent
        """
        try:
            self.logger.info("E.V. Application stopping...")
            self.state = AppState.STOPPED
            
            # Shutdown systems in reverse order
            if self.background_service:
                self._stop_background_service()
            
            if self.voice_manager:
                self._shutdown_voice_manager()
            
            if self.memory_manager:
                self._shutdown_memory_manager()
            
            if self.ai_brain:
                self._shutdown_ai_brain()
            
            self.logger.info("E.V. Application stopped cleanly.")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)
            return True  # Force close anyway
    
    def on_pause(self) -> bool:
        """
        Called when application is paused (Android back button).
        
        Returns:
            bool: True to allow pause, False to prevent
        """
        try:
            self.logger.info("E.V. Application paused.")
            self.state = AppState.PAUSED
            
            # Save state
            if self.memory_manager:
                self.memory_manager.save_checkpoint()
            
            return True  # Allow pause
            
        except Exception as e:
            self.logger.error(f"Error during pause: {e}", exc_info=True)
            return True
    
    def on_resume(self) -> None:
        """Called when application is resumed from pause."""
        try:
            self.logger.info("E.V. Application resumed.")
            self.state = AppState.ACTIVE
            
            # Restore state
            if self.memory_manager:
                self.memory_manager.restore_checkpoint()
            
        except Exception as e:
            self.logger.error(f"Error during resume: {e}", exc_info=True)
    
    # ========================================================================
    # THEME & STYLING
    # ========================================================================
    
    def _apply_theme(self) -> None:
        """Apply glassmorphism theme to the application."""
        try:
            self.logger.info(f"Applying {self.theme_mode.value} theme...")
            
            # Set MDApp theme properties
            self.theme_cls.theme_style = "Dark"
            self.theme_cls.primary_palette = "Cyan"
            
            # Custom color scheme (override defaults)
            self.theme_cls.primary_hue = "500"
            self.theme_cls.accent_palette = "Green"
            
            # Window background
            Window.clearcolor = DESIGN_TOKENS["colors"]["background"]
            
            self.logger.info("Theme applied successfully.")
            
        except Exception as e:
            self.logger.error(f"Failed to apply theme: {e}", exc_info=True)
    
    def _load_fonts(self) -> None:
        """Load custom fonts for the application."""
        try:
            assets_dir = self.app_root / "assets" / "fonts"
            if not assets_dir.exists():
                self.logger.warning(f"Fonts directory not found: {assets_dir}")
                return
            
            # Register Roboto font
            roboto_path = str(assets_dir / "Roboto-Regular.ttf")
            if Path(roboto_path).exists():
                LabelBase.register(
                    name="Roboto",
                    fn_regular=roboto_path
                )
                self.logger.info("Roboto font loaded.")
            
        except Exception as e:
            self.logger.error(f"Failed to load fonts: {e}", exc_info=True)
    
    # ========================================================================
    # SYSTEM INITIALIZATION
    # ========================================================================
    
    def _initialize_ai_brain(self, dt) -> None:
        """Initialize AI orchestrator system."""
        try:
            self.logger.info("Initializing AI Brain...")
            from ai_brain import AIBrain
            
            self.ai_brain = AIBrain(
                app_root=self.app_root,
                logger=self.logger
            )
            self.logger.info("AI Brain initialized.")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AI Brain: {e}", exc_info=True)
    
    def _initialize_memory_manager(self, dt) -> None:
        """Initialize unified memory manager."""
        try:
            self.logger.info("Initializing Memory Manager...")
            from memory_manager import MemoryManager
            
            self.memory_manager = MemoryManager(
                app_root=self.app_root,
                logger=self.logger
            )
            self.logger.info("Memory Manager initialized.")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Memory Manager: {e}", exc_info=True)
    
    def _initialize_voice_manager(self, dt) -> None:
        """Initialize voice I/O manager."""
        try:
            self.logger.info("Initializing Voice Manager...")
            from voice_manager import VoiceManager
            
            self.voice_manager = VoiceManager(
                app_root=self.app_root,
                logger=self.logger
            )
            self.logger.info("Voice Manager initialized.")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Voice Manager: {e}", exc_info=True)
    
    def _load_screens(self, dt) -> None:
        """Load and register application screens."""
        try:
            self.logger.info("Loading screens...")
            
            from chat_screen import ChatScreen
            from settings_screen import SettingsScreen
            
            # Create and add screens
            chat_screen = ChatScreen(
                app=self,
                name="chat"
            )
            self.screen_manager.add_widget(chat_screen)
            
            settings_screen = SettingsScreen(
                app=self,
                name="settings"
            )
            self.screen_manager.add_widget(settings_screen)
            
            # Set default screen
            self.screen_manager.current = "chat"
            
            self.logger.info("Screens loaded successfully.")
            
        except Exception as e:
            self.logger.error(f"Failed to load screens: {e}", exc_info=True)
    
    def _start_background_service(self, dt) -> None:
        """Start background service (Android foreground service)."""
        try:
            if not IS_ANDROID:
                self.logger.info("Skipping background service (not on Android).")
                return
            
            self.logger.info("Starting background service...")
            from background_service import BackgroundService
            
            self.background_service = BackgroundService(
                app_root=self.app_root,
                logger=self.logger
            )
            self.background_service.start()
            self.logger.info("Background service started.")
            
        except Exception as e:
            self.logger.error(f"Failed to start background service: {e}", exc_info=True)
    
    # ========================================================================
    # SYSTEM SHUTDOWN
    # ========================================================================
    
    def _shutdown_ai_brain(self) -> None:
        """Shutdown AI Brain."""
        try:
            if self.ai_brain:
                self.ai_brain.shutdown()
                self.logger.info("AI Brain shutdown.")
        except Exception as e:
            self.logger.error(f"Error shutting down AI Brain: {e}", exc_info=True)
    
    def _shutdown_memory_manager(self) -> None:
        """Shutdown Memory Manager."""
        try:
            if self.memory_manager:
                self.memory_manager.shutdown()
                self.logger.info("Memory Manager shutdown.")
        except Exception as e:
            self.logger.error(f"Error shutting down Memory Manager: {e}", exc_info=True)
    
    def _shutdown_voice_manager(self) -> None:
        """Shutdown Voice Manager."""
        try:
            if self.voice_manager:
                self.voice_manager.shutdown()
                self.logger.info("Voice Manager shutdown.")
        except Exception as e:
            self.logger.error(f"Error shutting down Voice Manager: {e}", exc_info=True)
    
    def _stop_background_service(self) -> None:
        """Stop background service."""
        try:
            if self.background_service:
                self.background_service.stop()
                self.logger.info("Background service stopped.")
        except Exception as e:
            self.logger.error(f"Error stopping background service: {e}", exc_info=True)
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def set_status(self, status: str, color: tuple = None) -> None:
        """
        Update application status indicator.
        
        Args:
            status: Status string (ONLINE, THINKING, LISTENING, etc.)
            color: RGBA color tuple (optional, uses predefined if not provided)
        """
        try:
            if color is None:
                color = STATUS_COLORS.get(status, DESIGN_TOKENS["colors"]["text_secondary"])
            
            self.logger.debug(f"Status updated: {status}")
            
            # Broadcast to current screen if it has status handler
            if self.screen_manager and self.screen_manager.current_screen:
                current_screen = self.screen_manager.current_screen
                if hasattr(current_screen, 'on_status_update'):
                    current_screen.on_status_update(status, color)
                    
        except Exception as e:
            self.logger.error(f"Failed to set status: {e}", exc_info=True)
    
    def navigate_to_screen(self, screen_name: str) -> None:
        """
        Navigate to a specific screen.
        
        Args:
            screen_name: Name of the screen to navigate to
        """
        try:
            if self.screen_manager:
                self.screen_manager.current = screen_name
                self.logger.debug(f"Navigated to screen: {screen_name}")
        except Exception as e:
            self.logger.error(f"Failed to navigate: {e}", exc_info=True)


# ============================================================================
# STANDALONE EXECUTION (for testing)
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path
    
    app_root = Path.home() / ".eternal_voice_test"
    app_root.mkdir(exist_ok=True)
    
    class DummyCrashHandler:
        def __init__(self):
            self.logger = logging.getLogger("DummyCrashHandler")
    
    app = EternalVoiceApp(app_root=app_root, crash_handler=DummyCrashHandler())
    app.run()
