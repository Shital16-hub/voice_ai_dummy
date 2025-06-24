# call_transcription_storage.py
"""
Call Transcription Storage System with Local Storage + MongoDB Migration Ready
Captures complete conversation transcripts with caller identification and history
"""
import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import hashlib
import uuid

logger = logging.getLogger(__name__)

@dataclass
class TranscriptionSegment:
    """Individual transcription segment (user or agent speech)"""
    segment_id: str
    session_id: str
    caller_id: str
    speaker: str  # "user" or "agent" 
    text: str
    timestamp: float
    is_final: bool
    confidence: Optional[float] = None
    duration_ms: Optional[int] = None
    
@dataclass
class ConversationItem:
    """Complete conversation turn"""
    item_id: str
    session_id: str
    caller_id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: float
    interrupted: bool = False
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class CallSession:
    """Complete call session record"""
    session_id: str
    caller_id: str
    phone_number: str
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    total_turns: int = 0
    status: str = "active"  # active, completed, failed
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class CallerProfile:
    """Caller profile with conversation history"""
    caller_id: str
    phone_number: str
    first_call_time: float
    last_call_time: float
    total_calls: int = 0
    total_conversation_turns: int = 0
    metadata: Optional[Dict[str, Any]] = None

class CallTranscriptionStorage:
    """
    Local SQLite storage for call transcriptions with MongoDB migration structure
    
    Features:
    - Complete call transcription logging
    - Caller identification and history tracking  
    - Conversation context restoration
    - MongoDB-ready data structure
    - Async operations for LiveKit integration
    """
    
    def __init__(self, db_path: str = "call_transcriptions.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # In-memory cache for active sessions
        self.active_sessions: Dict[str, CallSession] = {}
        self.caller_cache: Dict[str, CallerProfile] = {}
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with MongoDB-compatible structure"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create tables first
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS call_sessions (
                        session_id TEXT PRIMARY KEY,
                        caller_id TEXT NOT NULL,
                        phone_number TEXT NOT NULL,
                        start_time REAL NOT NULL,
                        end_time REAL,
                        duration_seconds REAL,
                        total_turns INTEGER DEFAULT 0,
                        status TEXT DEFAULT 'active',
                        metadata TEXT,
                        created_at REAL DEFAULT (julianday('now'))
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS transcription_segments (
                        segment_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        caller_id TEXT NOT NULL,
                        speaker TEXT NOT NULL,
                        text TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        is_final BOOLEAN NOT NULL,
                        confidence REAL,
                        duration_ms INTEGER,
                        created_at REAL DEFAULT (julianday('now')),
                        FOREIGN KEY (session_id) REFERENCES call_sessions (session_id)
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_items (
                        item_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        caller_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        interrupted BOOLEAN DEFAULT FALSE,
                        metadata TEXT,
                        created_at REAL DEFAULT (julianday('now')),
                        FOREIGN KEY (session_id) REFERENCES call_sessions (session_id)
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS caller_profiles (
                        caller_id TEXT PRIMARY KEY,
                        phone_number TEXT UNIQUE NOT NULL,
                        first_call_time REAL NOT NULL,
                        last_call_time REAL NOT NULL,
                        total_calls INTEGER DEFAULT 0,
                        total_conversation_turns INTEGER DEFAULT 0,
                        metadata TEXT,
                        updated_at REAL DEFAULT (julianday('now'))
                    )
                """)
                
                # Create indexes separately
                conn.execute("CREATE INDEX IF NOT EXISTS idx_call_sessions_caller_id ON call_sessions(caller_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_call_sessions_phone_number ON call_sessions(phone_number)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_call_sessions_start_time ON call_sessions(start_time)")
                
                conn.execute("CREATE INDEX IF NOT EXISTS idx_transcription_segments_session_id ON transcription_segments(session_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_transcription_segments_caller_id ON transcription_segments(caller_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_transcription_segments_timestamp ON transcription_segments(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_transcription_segments_speaker ON transcription_segments(speaker)")
                
                conn.execute("CREATE INDEX IF NOT EXISTS idx_conversation_items_session_id ON conversation_items(session_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_conversation_items_caller_id ON conversation_items(caller_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_conversation_items_timestamp ON conversation_items(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_conversation_items_role ON conversation_items(role)")
                
                conn.execute("CREATE INDEX IF NOT EXISTS idx_caller_profiles_phone_number ON caller_profiles(phone_number)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_caller_profiles_last_call_time ON caller_profiles(last_call_time)")
                
                conn.commit()
                logger.info("✅ Call transcription database initialized")
                
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    def _generate_ids(self) -> tuple[str, str]:
        """Generate session_id and caller_id"""
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        caller_id = f"caller_{hashlib.md5(str(time.time()).encode()).hexdigest()[:10]}"
        return session_id, caller_id
    
    async def start_call_session(
        self, 
        phone_number: str, 
        session_metadata: Optional[Dict[str, Any]] = None
    ) -> tuple[str, str]:
        """
        Start a new call session and return (session_id, caller_id)
        
        Args:
            phone_number: Caller's phone number
            session_metadata: Additional session metadata
            
        Returns:
            Tuple of (session_id, caller_id)
        """
        try:
            # Check if this is a returning caller
            caller_profile = await self.get_caller_by_phone(phone_number)
            
            if caller_profile:
                caller_id = caller_profile.caller_id
                session_id = f"session_{uuid.uuid4().hex[:12]}"
                logger.info(f"📞 Returning caller detected: {phone_number}")
            else:
                session_id, caller_id = self._generate_ids()
                # Create new caller profile
                caller_profile = CallerProfile(
                    caller_id=caller_id,
                    phone_number=phone_number,
                    first_call_time=time.time(),
                    last_call_time=time.time(),
                    total_calls=1,
                    metadata={}
                )
                await self._save_caller_profile(caller_profile)
                logger.info(f"📞 New caller registered: {phone_number}")
            
            # Create call session
            call_session = CallSession(
                session_id=session_id,
                caller_id=caller_id,
                phone_number=phone_number,
                start_time=time.time(),
                metadata=session_metadata or {}
            )
            
            # Save to database and cache
            await self._save_call_session(call_session)
            self.active_sessions[session_id] = call_session
            
            logger.info(f"🎯 Call session started: {session_id} for {phone_number}")
            return session_id, caller_id
            
        except Exception as e:
            logger.error(f"❌ Failed to start call session: {e}")
            raise
    
    async def save_transcription_segment(
        self,
        session_id: str,
        caller_id: str,
        speaker: str,
        text: str,
        is_final: bool,
        confidence: Optional[float] = None,
        duration_ms: Optional[int] = None
    ) -> str:
        """Save transcription segment to database"""
        try:
            segment = TranscriptionSegment(
                segment_id=f"seg_{uuid.uuid4().hex[:12]}",
                session_id=session_id,
                caller_id=caller_id,
                speaker=speaker,
                text=text,
                timestamp=time.time(),
                is_final=is_final,
                confidence=confidence,
                duration_ms=duration_ms
            )
            
            await asyncio.to_thread(self._save_transcription_segment_sync, segment)
            logger.debug(f"💬 Saved transcription: {speaker} - {text[:50]}...")
            return segment.segment_id
            
        except Exception as e:
            logger.error(f"❌ Failed to save transcription segment: {e}")
            return ""
    
    async def save_conversation_item(
        self,
        session_id: str,
        caller_id: str,
        role: str,
        content: str,
        interrupted: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save complete conversation item"""
        try:
            item = ConversationItem(
                item_id=f"item_{uuid.uuid4().hex[:12]}",
                session_id=session_id,
                caller_id=caller_id,
                role=role,
                content=content,
                timestamp=time.time(),
                interrupted=interrupted,
                metadata=metadata
            )
            
            await asyncio.to_thread(self._save_conversation_item_sync, item)
            
            # Update session turn count
            if session_id in self.active_sessions:
                self.active_sessions[session_id].total_turns += 1
            
            logger.info(f"📝 Saved conversation item: {role} - {content[:50]}...")
            return item.item_id
            
        except Exception as e:
            logger.error(f"❌ Failed to save conversation item: {e}")
            return ""
    
    async def end_call_session(self, session_id: str) -> bool:
        """End call session and update statistics"""
        try:
            if session_id not in self.active_sessions:
                logger.warning(f"⚠️ Session not found in active sessions: {session_id}")
                return False
            
            session = self.active_sessions[session_id]
            session.end_time = time.time()
            session.duration_seconds = session.end_time - session.start_time
            session.status = "completed"
            
            # Save final session state
            await self._save_call_session(session)
            
            # Update caller statistics
            await self._update_caller_statistics(session.caller_id, session.total_turns)
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            logger.info(f"✅ Call session ended: {session_id} (Duration: {session.duration_seconds:.1f}s)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to end call session: {e}")
            return False
    
    async def get_caller_conversation_history(
        self, 
        caller_id: str, 
        limit: int = 50,
        days_back: int = 30
    ) -> List[ConversationItem]:
        """Get conversation history for caller identification and context"""
        try:
            cutoff_time = time.time() - (days_back * 24 * 3600)
            
            query = """
                SELECT item_id, session_id, caller_id, role, content, 
                       timestamp, interrupted, metadata
                FROM conversation_items 
                WHERE caller_id = ? AND timestamp > ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            
            result = await asyncio.to_thread(
                self._execute_query, query, (caller_id, cutoff_time, limit)
            )
            
            history = []
            for row in result:
                metadata = json.loads(row[7]) if row[7] else None
                history.append(ConversationItem(
                    item_id=row[0],
                    session_id=row[1], 
                    caller_id=row[2],
                    role=row[3],
                    content=row[4],
                    timestamp=row[5],
                    interrupted=bool(row[6]),
                    metadata=metadata
                ))
            
            logger.info(f"📚 Retrieved {len(history)} conversation items for caller {caller_id}")
            return history
            
        except Exception as e:
            logger.error(f"❌ Failed to get conversation history: {e}")
            return []
    
    async def get_caller_by_phone(self, phone_number: str) -> Optional[CallerProfile]:
        """Get caller profile by phone number"""
        try:
            # Check cache first
            for caller in self.caller_cache.values():
                if caller.phone_number == phone_number:
                    return caller
            
            query = """
                SELECT caller_id, phone_number, first_call_time, last_call_time,
                       total_calls, total_conversation_turns, metadata
                FROM caller_profiles 
                WHERE phone_number = ?
            """
            
            result = await asyncio.to_thread(
                self._execute_query, query, (phone_number,)
            )
            
            if result:
                row = result[0]
                metadata = json.loads(row[6]) if row[6] else None
                caller = CallerProfile(
                    caller_id=row[0],
                    phone_number=row[1],
                    first_call_time=row[2],
                    last_call_time=row[3],
                    total_calls=row[4],
                    total_conversation_turns=row[5],
                    metadata=metadata
                )
                
                # Cache for future use
                self.caller_cache[caller.caller_id] = caller
                return caller
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get caller by phone: {e}")
            return None
    
    async def get_recent_sessions(self, limit: int = 10) -> List[CallSession]:
        """Get recent call sessions for monitoring"""
        try:
            query = """
                SELECT session_id, caller_id, phone_number, start_time, end_time,
                       duration_seconds, total_turns, status, metadata
                FROM call_sessions 
                ORDER BY start_time DESC 
                LIMIT ?
            """
            
            result = await asyncio.to_thread(
                self._execute_query, query, (limit,)
            )
            
            sessions = []
            for row in result:
                metadata = json.loads(row[8]) if row[8] else None
                sessions.append(CallSession(
                    session_id=row[0],
                    caller_id=row[1],
                    phone_number=row[2],
                    start_time=row[3],
                    end_time=row[4],
                    duration_seconds=row[5],
                    total_turns=row[6],
                    status=row[7],
                    metadata=metadata
                ))
            
            return sessions
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent sessions: {e}")
            return []
    
    async def export_for_mongodb(self, session_id: str) -> Dict[str, Any]:
        """
        Export session data in MongoDB-ready format
        
        This method prepares data for easy MongoDB migration
        """
        try:
            # Get session data
            session_query = """
                SELECT session_id, caller_id, phone_number, start_time, end_time,
                       duration_seconds, total_turns, status, metadata
                FROM call_sessions WHERE session_id = ?
            """
            
            session_result = await asyncio.to_thread(
                self._execute_query, session_query, (session_id,)
            )
            
            if not session_result:
                return {}
            
            session_row = session_result[0]
            
            # Get conversation items
            items_query = """
                SELECT item_id, role, content, timestamp, interrupted, metadata
                FROM conversation_items 
                WHERE session_id = ? 
                ORDER BY timestamp
            """
            
            items_result = await asyncio.to_thread(
                self._execute_query, items_query, (session_id,)
            )
            
            # Get transcription segments
            segments_query = """
                SELECT segment_id, speaker, text, timestamp, is_final, confidence, duration_ms
                FROM transcription_segments 
                WHERE session_id = ? 
                ORDER BY timestamp
            """
            
            segments_result = await asyncio.to_thread(
                self._execute_query, segments_query, (session_id,)
            )
            
            # Format for MongoDB
            mongodb_doc = {
                "_id": session_id,
                "session_id": session_row[0],
                "caller_id": session_row[1],
                "phone_number": session_row[2],
                "start_time": datetime.fromtimestamp(session_row[3]),
                "end_time": datetime.fromtimestamp(session_row[4]) if session_row[4] else None,
                "duration_seconds": session_row[5],
                "total_turns": session_row[6],
                "status": session_row[7],
                "metadata": json.loads(session_row[8]) if session_row[8] else {},
                "conversation_items": [
                    {
                        "item_id": row[0],
                        "role": row[1],
                        "content": row[2],
                        "timestamp": datetime.fromtimestamp(row[3]),
                        "interrupted": bool(row[4]),
                        "metadata": json.loads(row[5]) if row[5] else {}
                    }
                    for row in items_result
                ],
                "transcription_segments": [
                    {
                        "segment_id": row[0],
                        "speaker": row[1],
                        "text": row[2],
                        "timestamp": datetime.fromtimestamp(row[3]),
                        "is_final": bool(row[4]),
                        "confidence": row[5],
                        "duration_ms": row[6]
                    }
                    for row in segments_result
                ],
                "created_at": datetime.utcnow(),
                "version": "1.0"
            }
            
            return mongodb_doc
            
        except Exception as e:
            logger.error(f"❌ Failed to export for MongoDB: {e}")
            return {}
    
    # Private helper methods
    def _save_transcription_segment_sync(self, segment: TranscriptionSegment):
        """Synchronous helper for saving transcription segments"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO transcription_segments 
                (segment_id, session_id, caller_id, speaker, text, timestamp, 
                 is_final, confidence, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                segment.segment_id, segment.session_id, segment.caller_id,
                segment.speaker, segment.text, segment.timestamp,
                segment.is_final, segment.confidence, segment.duration_ms
            ))
            conn.commit()
    
    def _save_conversation_item_sync(self, item: ConversationItem):
        """Synchronous helper for saving conversation items"""
        with sqlite3.connect(self.db_path) as conn:
            metadata_json = json.dumps(item.metadata) if item.metadata else None
            conn.execute("""
                INSERT INTO conversation_items 
                (item_id, session_id, caller_id, role, content, timestamp, 
                 interrupted, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.item_id, item.session_id, item.caller_id, item.role,
                item.content, item.timestamp, item.interrupted, metadata_json
            ))
            conn.commit()
    
    async def _save_call_session(self, session: CallSession):
        """Save call session to database"""
        metadata_json = json.dumps(session.metadata) if session.metadata else None
        
        query = """
            INSERT OR REPLACE INTO call_sessions 
            (session_id, caller_id, phone_number, start_time, end_time, 
             duration_seconds, total_turns, status, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        await asyncio.to_thread(
            self._execute_update, query, (
                session.session_id, session.caller_id, session.phone_number,
                session.start_time, session.end_time, session.duration_seconds,
                session.total_turns, session.status, metadata_json
            )
        )
    
    async def _save_caller_profile(self, caller: CallerProfile):
        """Save caller profile to database"""
        metadata_json = json.dumps(caller.metadata) if caller.metadata else None
        
        query = """
            INSERT OR REPLACE INTO caller_profiles 
            (caller_id, phone_number, first_call_time, last_call_time,
             total_calls, total_conversation_turns, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        await asyncio.to_thread(
            self._execute_update, query, (
                caller.caller_id, caller.phone_number, caller.first_call_time,
                caller.last_call_time, caller.total_calls, 
                caller.total_conversation_turns, metadata_json
            )
        )
        
        # Update cache
        self.caller_cache[caller.caller_id] = caller
    
    async def _update_caller_statistics(self, caller_id: str, turns_added: int):
        """Update caller statistics after call"""
        query = """
            UPDATE caller_profiles 
            SET last_call_time = ?, total_calls = total_calls + 1,
                total_conversation_turns = total_conversation_turns + ?
            WHERE caller_id = ?
        """
        
        await asyncio.to_thread(
            self._execute_update, query, (time.time(), turns_added, caller_id)
        )
        
        # Update cache if present
        if caller_id in self.caller_cache:
            self.caller_cache[caller_id].last_call_time = time.time()
            self.caller_cache[caller_id].total_calls += 1
            self.caller_cache[caller_id].total_conversation_turns += turns_added
    
    def _execute_query(self, query: str, params: tuple = ()) -> List[tuple]:
        """Execute SELECT query"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchall()
    
    def _execute_update(self, query: str, params: tuple = ()):
        """Execute INSERT/UPDATE query"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(query, params)
            conn.commit()

# Global storage instance
call_storage = CallTranscriptionStorage()

async def get_call_storage() -> CallTranscriptionStorage:
    """Get the global call storage instance"""
    return call_storage