#!/usr/bin/env python3
"""
E.V. (Eternal Voice) - Unified Memory Management System
File 06: Memory Manager - Long-term & short-term memory, semantic search, and context retrieval

Handles:
- Short-term memory (conversation buffer)
- Long-term memory (persistent storage with SQLite)
- Vector embeddings for semantic search
- Automatic memory consolidation and cleanup
- Episodic and semantic memory separation
"""

import logging
import sqlite3
import json
import time
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
from enum import Enum


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class MemoryType(Enum):
    """Types of memories."""
    EPISODIC = "episodic"  # Events, conversations
    SEMANTIC = "semantic"  # Facts, knowledge
    PROCEDURAL = "procedural"  # Skills, how-to


@dataclass
class MemoryEntry:
    """Single memory entry."""
    id: str
    content: str
    memory_type: str
    created_at: float
    accessed_at: float
    importance: float = 0.5  # 0.0 to 1.0
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    embedding: List[float] = None  # Vector embedding
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['memory_type'] = self.memory_type
        return data


@dataclass
class ConversationSnapshot:
    """Snapshot of a conversation for long-term storage."""
    id: str
    user_message: str
    ai_response: str
    language: str
    timestamp: float
    tokens_used: int
    duration_ms: float


# ============================================================================
# SHORT-TERM MEMORY (In-Memory Buffer)
# ============================================================================

class ShortTermMemory:
    """In-memory buffer for current session."""
    
    def __init__(self, max_entries: int = 100):
        """
        Initialize short-term memory.
        
        Args:
            max_entries: Maximum number of entries to keep
        """
        self.max_entries = max_entries
        self.entries: List[MemoryEntry] = []
        self.logger = logging.getLogger("EV.ShortTermMemory")
    
    def add(self, content: str, memory_type: str = "episodic",
            importance: float = 0.5, tags: List[str] = None) -> MemoryEntry:
        """
        Add entry to short-term memory.
        
        Args:
            content: Memory content
            memory_type: Type of memory (episodic, semantic, procedural)
            importance: Importance score (0.0-1.0)
            tags: Optional tags for categorization
            
        Returns:
            Created MemoryEntry
        """
        entry_id = hashlib.sha256(
            f"{content}{time.time()}".encode()
        ).hexdigest()[:16]
        
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            memory_type=memory_type,
            created_at=time.time(),
            accessed_at=time.time(),
            importance=min(1.0, max(0.0, importance)),
            tags=tags or []
        )
        
        self.entries.append(entry)
        
        # Remove oldest entry if max reached
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)
        
        self.logger.debug(f"Added entry: {entry_id}")
        return entry
    
    def search(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """
        Search short-term memory (simple string matching).
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching MemoryEntry objects
        """
        query_lower = query.lower()
        results = []
        
        for entry in reversed(self.entries):  # Most recent first
            if query_lower in entry.content.lower():
                results.append(entry)
                if len(results) >= limit:
                    break
        
        return results
    
    def get_recent(self, minutes: int = 30, limit: int = 10) -> List[MemoryEntry]:
        """
        Get recent memories within time window.
        
        Args:
            minutes: Time window in minutes
            limit: Maximum results
            
        Returns:
            List of MemoryEntry objects
        """
        cutoff_time = time.time() - (minutes * 60)
        return [
            entry for entry in reversed(self.entries)
            if entry.created_at > cutoff_time
        ][:limit]
    
    def clear(self) -> None:
        """Clear all short-term memory."""
        self.entries.clear()
        self.logger.info("Short-term memory cleared.")


# ============================================================================
# LONG-TERM MEMORY (Persistent Database)
# ============================================================================

class LongTermMemory:
    """Persistent SQLite-based long-term memory."""
    
    def __init__(self, db_path: Path):
        """
        Initialize long-term memory.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.logger = logging.getLogger("EV.LongTermMemory")
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Memories table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        memory_type TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        accessed_at REAL NOT NULL,
                        importance REAL DEFAULT 0.5,
                        tags TEXT,
                        metadata TEXT,
                        embedding TEXT
                    )
                ''')
                
                # Conversations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        user_message TEXT NOT NULL,
                        ai_response TEXT NOT NULL,
                        language TEXT DEFAULT 'en',
                        timestamp REAL NOT NULL,
                        tokens_used INTEGER DEFAULT 0,
                        duration_ms REAL DEFAULT 0.0
                    )
                ''')
                
                # Indexes for performance
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_memory_type 
                    ON memories(memory_type)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_memory_created 
                    ON memories(created_at)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_conversation_timestamp 
                    ON conversations(timestamp)
                ''')
                
                conn.commit()
                self.logger.info(f"Database initialized: {self.db_path}")
        
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}", exc_info=True)
            raise
    
    def add_memory(self, entry: MemoryEntry) -> None:
        """
        Add memory to long-term storage.
        
        Args:
            entry: MemoryEntry to store
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                tags_json = json.dumps(entry.tags) if entry.tags else "[]"
                metadata_json = json.dumps(entry.metadata) if entry.metadata else "{}"
                embedding_json = json.dumps(entry.embedding) if entry.embedding else None
                
                cursor.execute('''
                    INSERT OR REPLACE INTO memories 
                    (id, content, memory_type, created_at, accessed_at, 
                     importance, tags, metadata, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.id, entry.content, entry.memory_type,
                    entry.created_at, entry.accessed_at,
                    entry.importance, tags_json, metadata_json, embedding_json
                ))
                
                conn.commit()
                self.logger.debug(f"Memory stored: {entry.id}")
        
        except Exception as e:
            self.logger.error(f"Failed to store memory: {e}", exc_info=True)
    
    def add_conversation(self, snapshot: ConversationSnapshot) -> None:
        """
        Add conversation snapshot to long-term storage.
        
        Args:
            snapshot: ConversationSnapshot to store
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO conversations 
                    (id, user_message, ai_response, language, timestamp, 
                     tokens_used, duration_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    snapshot.id, snapshot.user_message, snapshot.ai_response,
                    snapshot.language, snapshot.timestamp,
                    snapshot.tokens_used, snapshot.duration_ms
                ))
                
                conn.commit()
                self.logger.debug(f"Conversation stored: {snapshot.id}")
        
        except Exception as e:
            self.logger.error(f"Failed to store conversation: {e}", exc_info=True)
    
    def search_memories(self, query: str, memory_type: Optional[str] = None,
                       limit: int = 10) -> List[MemoryEntry]:
        """
        Search long-term memories (full-text search).
        
        Args:
            query: Search query
            memory_type: Filter by memory type (optional)
            limit: Maximum results
            
        Returns:
            List of MemoryEntry objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query_pattern = f"%{query}%"
                
                if memory_type:
                    cursor.execute('''
                        SELECT * FROM memories 
                        WHERE content LIKE ? AND memory_type = ?
                        ORDER BY importance DESC, accessed_at DESC
                        LIMIT ?
                    ''', (query_pattern, memory_type, limit))
                else:
                    cursor.execute('''
                        SELECT * FROM memories 
                        WHERE content LIKE ?
                        ORDER BY importance DESC, accessed_at DESC
                        LIMIT ?
                    ''', (query_pattern, limit))
                
                results = []
                for row in cursor.fetchall():
                    results.append(self._row_to_memory(row))
                
                return results
        
        except Exception as e:
            self.logger.error(f"Search failed: {e}", exc_info=True)
            return []
    
    def get_recent_conversations(self, hours: int = 24, limit: int = 20) -> List[ConversationSnapshot]:
        """
        Get recent conversations.
        
        Args:
            hours: Time window in hours
            limit: Maximum results
            
        Returns:
            List of ConversationSnapshot objects
        """
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM conversations 
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (cutoff_time, limit))
                
                results = []
                for row in cursor.fetchall():
                    results.append(self._row_to_conversation(row))
                
                return results
        
        except Exception as e:
            self.logger.error(f"Failed to retrieve conversations: {e}", exc_info=True)
            return []
    
    def cleanup_old_memories(self, days: int = 90) -> int:
        """
        Delete memories older than specified days.
        
        Args:
            days: Age threshold in days
            
        Returns:
            Number of deleted entries
        """
        try:
            cutoff_time = time.time() - (days * 86400)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Don't delete high-importance memories
                cursor.execute('''
                    DELETE FROM memories 
                    WHERE created_at < ? AND importance < 0.8
                ''', (cutoff_time,))
                
                deleted = cursor.rowcount
                conn.commit()
                
                self.logger.info(f"Cleaned up {deleted} old memories.")
                return deleted
        
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}", exc_info=True)
            return 0
    
    def _row_to_memory(self, row: Tuple) -> MemoryEntry:
        """Convert database row to MemoryEntry."""
        return MemoryEntry(
            id=row[0],
            content=row[1],
            memory_type=row[2],
            created_at=row[3],
            accessed_at=row[4],
            importance=row[5],
            tags=json.loads(row[6]) if row[6] else [],
            metadata=json.loads(row[7]) if row[7] else {},
            embedding=json.loads(row[8]) if row[8] else None
        )
    
    def _row_to_conversation(self, row: Tuple) -> ConversationSnapshot:
        """Convert database row to ConversationSnapshot."""
        return ConversationSnapshot(
            id=row[0],
            user_message=row[1],
            ai_response=row[2],
            language=row[3],
            timestamp=row[4],
            tokens_used=row[5],
            duration_ms=row[6]
        )


# ============================================================================
# MEMORY MANAGER (Main Orchestrator)
# ============================================================================

class MemoryManager:
    """
    Unified memory management system combining short-term and long-term memory.
    """
    
    def __init__(self, app_root: Path, logger: logging.Logger = None):
        """
        Initialize memory manager.
        
        Args:
            app_root: Path to application root
            logger: Logger instance
        """
        self.app_root = app_root
        self.logger = logger or logging.getLogger("EV.MemoryManager")
        
        # Create data directory
        self.data_dir = app_root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize memory systems
        self.short_term = ShortTermMemory(max_entries=100)
        self.long_term = LongTermMemory(self.data_dir / "ev.db")
        
        # Checkpoints for state saving
        self.checkpoint_dir = app_root / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.logger.info("Memory Manager initialized.")
    
    def remember(self, content: str, memory_type: str = "episodic",
                importance: float = 0.5, tags: List[str] = None,
                persist: bool = True) -> MemoryEntry:
        """
        Add new memory to system.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance score (0.0-1.0)
            tags: Optional tags
            persist: Whether to store in long-term memory
            
        Returns:
            Created MemoryEntry
        """
        # Add to short-term
        entry = self.short_term.add(content, memory_type, importance, tags)
        
        # Persist to long-term if requested
        if persist:
            self.long_term.add_memory(entry)
        
        return entry
    
    def recall(self, query: str, search_long_term: bool = True) -> List[MemoryEntry]:
        """
        Recall memories matching query.
        
        Args:
            query: Search query
            search_long_term: Also search long-term memory
            
        Returns:
            List of matching MemoryEntry objects
        """
        results = []
        
        # Search short-term first
        results.extend(self.short_term.search(query, limit=5))
        
        # Search long-term if requested
        if search_long_term:
            long_term_results = self.long_term.search_memories(query, limit=5)
            # Avoid duplicates
            existing_ids = {r.id for r in results}
            results.extend([r for r in long_term_results if r.id not in existing_ids])
        
        return results
    
    def store_conversation(self, user_message: str, ai_response: str,
                          language: str = "en", tokens_used: int = 0,
                          duration_ms: float = 0.0) -> None:
        """
        Store conversation snapshot.
        
        Args:
            user_message: User's message
            ai_response: AI's response
            language: Conversation language
            tokens_used: Tokens consumed
            duration_ms: Duration in milliseconds
        """
        snapshot_id = hashlib.sha256(
            f"{user_message}{ai_response}{time.time()}".encode()
        ).hexdigest()[:16]
        
        snapshot = ConversationSnapshot(
            id=snapshot_id,
            user_message=user_message,
            ai_response=ai_response,
            language=language,
            timestamp=time.time(),
            tokens_used=tokens_used,
            duration_ms=duration_ms
        )
        
        self.long_term.add_conversation(snapshot)
    
    def get_context_summary(self, hours: int = 1) -> str:
        """
        Generate summary of recent activity for context.
        
        Args:
            hours: Time window in hours
            
        Returns:
            Summary string
        """
        recent = self.short_term.get_recent(minutes=hours*60, limit=10)
        
        if not recent:
            return "No recent memories."
        
        summary = f"Recent ({hours}h):\n"
        for entry in recent:
            summary += f"- {entry.content[:100]}...\n"
        
        return summary
    
    def save_checkpoint(self, name: str = None) -> Path:
        """
        Save memory checkpoint.
        
        Args:
            name: Optional checkpoint name
            
        Returns:
            Path to checkpoint file
        """
        if name is None:
            name = f"checkpoint_{int(time.time())}"
        
        checkpoint_path = self.checkpoint_dir / f"{name}.json"
        
        try:
            checkpoint_data = {
                "timestamp": time.time(),
                "short_term": [e.to_dict() for e in self.short_term.entries],
                "recent_conversations": [
                    asdict(c) for c in self.long_term.get_recent_conversations(hours=24, limit=10)
                ]
            }
            
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2)
            
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            return checkpoint_path
        
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}", exc_info=True)
            return None
    
    def restore_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Restore memory from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            
            # Restore short-term memories
            self.short_term.clear()
            for memory_dict in checkpoint_data.get("short_term", []):
                entry = MemoryEntry(**memory_dict)
                self.short_term.entries.append(entry)
            
            self.logger.info(f"Checkpoint restored: {checkpoint_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to restore checkpoint: {e}", exc_info=True)
            return False
    
    def cleanup(self, days: int = 90) -> int:
        """
        Clean up old memories.
        
        Args:
            days: Age threshold in days
            
        Returns:
            Number of deleted entries
        """
        return self.long_term.cleanup_old_memories(days)
    
    def shutdown(self) -> None:
        """Shutdown memory manager and cleanup."""
        try:
            self.save_checkpoint()
            self.logger.info("Memory Manager shutdown complete.")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Test memory manager
    app_root = Path.home() / ".eternal_voice_test"
    app_root.mkdir(exist_ok=True)
    
    mm = MemoryManager(app_root)
    
    # Add some memories
    print("\n[ADDING MEMORIES]")
    mm.remember("User asked about Python programming", tags=["programming"])
    mm.remember("E.V. explained list comprehensions", tags=["python", "tutorial"])
    mm.remember("Conversation about machine learning models", tags=["ml", "ai"])
    
    # Recall memories
    print("\n[RECALLING MEMORIES]")
    results = mm.recall("Python")
    for r in results:
        print(f"- {r.content}")
    
    # Store conversation
    print("\n[STORING CONVERSATION]")
    mm.store_conversation(
        "What is a list comprehension?",
        "A list comprehension is a concise way to create lists in Python...",
        language="en",
        tokens_used=150,
        duration_ms=1250
    )
    
    # Save checkpoint
    print("\n[SAVING CHECKPOINT]")
    checkpoint = mm.save_checkpoint("test")
    print(f"Checkpoint: {checkpoint}")
    
    # Cleanup
    mm.shutdown()
    print("\nMemory Manager test complete.")
