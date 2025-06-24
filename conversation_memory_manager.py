# conversation_memory_manager.py
"""
Advanced Conversation Memory Manager for LiveKit Agents
Provides context retention, conversation summarization, and long-term memory
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import sqlite3
from pathlib import Path

from livekit.agents import ChatContext, ChatMessage
import openai

from config import config

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Single conversation turn"""
    timestamp: float
    speaker: str  # "user" or "agent"
    content: str
    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    sentiment: Optional[str] = None

@dataclass
class ConversationSummary:
    """Compressed conversation summary"""
    session_id: str
    start_time: float
    end_time: float
    participant_name: Optional[str]
    phone_number: Optional[str] 
    total_turns: int
    main_topics: List[str]
    outcome: str
    key_information: Dict[str, Any]
    satisfaction_score: Optional[float] = None

@dataclass
class ConversationMemory:
    """Complete conversation memory structure"""
    session_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    context_variables: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[ConversationSummary] = None
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

class ConversationMemoryManager:
    """Advanced memory management for voice conversations"""
    
    def __init__(self, db_path: str = "conversation_memory.db"):
        self.db_path = Path(db_path)
        self.active_memories: Dict[str, ConversationMemory] = {}
        self.compression_threshold = 20  # Turns before compression
        self.max_active_sessions = 100
        self.openai_client = openai.AsyncOpenAI(api_key=config.openai_api_key)
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for persistent memory"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_sessions (
                        session_id TEXT PRIMARY KEY,
                        participant_name TEXT,
                        phone_number TEXT,
                        start_time REAL,
                        end_time REAL,
                        total_turns INTEGER,
                        outcome TEXT,
                        summary_json TEXT,
                        created_at REAL
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_turns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        timestamp REAL,
                        speaker TEXT,
                        content TEXT,
                        intent TEXT,
                        entities_json TEXT,
                        FOREIGN KEY (session_id) REFERENCES conversation_sessions (session_id)
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS context_variables (
                        session_id TEXT,
                        variable_name TEXT,
                        variable_value TEXT,
                        timestamp REAL,
                        PRIMARY KEY (session_id, variable_name),
                        FOREIGN KEY (session_id) REFERENCES conversation_sessions (session_id)
                    )
                """)
                
                conn.commit()
                logger.info("✅ Memory database initialized")
                
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")

    def create_session(self, session_id: str) -> ConversationMemory:
        """Create new conversation memory session"""
        memory = ConversationMemory(session_id=session_id)
        self.active_memories[session_id] = memory
        
        # Cleanup old sessions if too many active
        if len(self.active_memories) > self.max_active_sessions:
            self._cleanup_old_sessions()
            
        logger.info(f"📝 Created conversation memory for session: {session_id}")
        return memory

    def get_or_create_session(self, session_id: str) -> ConversationMemory:
        """Get existing session or create new one"""
        if session_id not in self.active_memories:
            # Try to load from database first
            loaded_memory = self._load_session_from_db(session_id)
            if loaded_memory:
                self.active_memories[session_id] = loaded_memory
            else:
                return self.create_session(session_id)
        
        return self.active_memories[session_id]

    async def add_turn(
        self, 
        session_id: str, 
        speaker: str, 
        content: str,
        intent: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add conversation turn with automatic processing"""
        
        memory = self.get_or_create_session(session_id)
        
        # Create turn with automatic analysis
        turn = ConversationTurn(
            timestamp=time.time(),
            speaker=speaker,
            content=content,
            intent=intent or await self._extract_intent(content),
            entities=entities or await self._extract_entities(content),
            sentiment=await self._analyze_sentiment(content)
        )
        
        memory.turns.append(turn)
        memory.last_updated = time.time()
        
        # Auto-compress if conversation gets long
        if len(memory.turns) > self.compression_threshold:
            await self._compress_conversation(memory)
            
        logger.debug(f"📝 Added turn for {session_id}: {speaker} - {content[:50]}...")

    async def update_context(
        self, 
        session_id: str, 
        variables: Dict[str, Any]
    ) -> None:
        """Update context variables for session"""
        
        memory = self.get_or_create_session(session_id)
        memory.context_variables.update(variables)
        memory.last_updated = time.time()
        
        logger.debug(f"🔄 Updated context for {session_id}: {list(variables.keys())}")

    async def get_conversation_context(
        self, 
        session_id: str, 
        turns_limit: int = 10
    ) -> str:
        """Get formatted conversation context for LLM injection"""
        
        memory = self.get_or_create_session(session_id)
        
        if not memory.turns:
            return ""
        
        # Get recent turns
        recent_turns = memory.turns[-turns_limit:] if len(memory.turns) > turns_limit else memory.turns
        
        # Format for context injection
        context_parts = []
        
        # Add conversation summary if available
        if memory.summary:
            context_parts.append(f"CONVERSATION SUMMARY: {memory.summary.main_topics}")
        
        # Add key context variables
        if memory.context_variables:
            key_info = []
            for key, value in memory.context_variables.items():
                if value and key in ["caller_name", "phone_number", "location", "vehicle_info", "service_type"]:
                    key_info.append(f"{key}: {value}")
            if key_info:
                context_parts.append(f"KEY INFO: {' | '.join(key_info)}")
        
        # Add recent conversation flow
        if recent_turns:
            conversation_flow = []
            for turn in recent_turns[-5:]:  # Last 5 turns for immediate context
                conversation_flow.append(f"{turn.speaker}: {turn.content}")
            context_parts.append(f"RECENT CONVERSATION:\n{chr(10).join(conversation_flow)}")
        
        return "\n\n".join(context_parts)

    async def inject_context_to_chat(
        self, 
        session_id: str, 
        chat_context: ChatContext,
        include_history: bool = True
    ) -> None:
        """Inject conversation memory into ChatContext"""
        
        context_string = await self.get_conversation_context(session_id)
        
        if context_string:
            chat_context.add_message(
                role="system",
                content=f"[CONVERSATION MEMORY]:\n{context_string}"
            )
            logger.debug(f"💉 Injected conversation context for {session_id}")

    async def _extract_intent(self, content: str) -> Optional[str]:
        """Extract intent from user message using simple keyword matching"""
        content_lower = content.lower()
        
        # Intent mapping based on roadside assistance domain
        intent_keywords = {
            "request_towing": ["tow", "towing", "pull", "move my car", "broken down"],
            "request_battery": ["battery", "jump", "jumpstart", "dead battery", "won't start"],
            "request_tire": ["tire", "flat", "puncture", "spare"],
            "request_fuel": ["gas", "fuel", "out of gas", "empty"],
            "request_lockout": ["locked out", "keys", "locked"],
            "provide_location": ["address", "street", "highway", "mile", "exit"],
            "provide_vehicle_info": ["honda", "toyota", "ford", "year", "model"],
            "emergency": ["emergency", "urgent", "stranded", "highway", "dangerous"],
            "question_pricing": ["cost", "price", "how much", "fee"],
            "confirm_service": ["yes", "correct", "that's right", "confirm"]
        }
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return intent
        
        return "general_inquiry"

    async def _extract_entities(self, content: str) -> Dict[str, Any]:
        """Extract named entities from content"""
        entities = {}
        
        # Simple entity extraction for roadside assistance
        content_lower = content.lower()
        
        # Phone number extraction
        import re
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phone_match = re.search(phone_pattern, content)
        if phone_match:
            entities["phone_number"] = phone_match.group()
        
        # Vehicle brands
        vehicle_brands = ["honda", "toyota", "ford", "chevy", "bmw", "audi", "mercedes", "nissan", "hyundai", "kia", "jeep", "dodge"]
        for brand in vehicle_brands:
            if brand in content_lower:
                entities["vehicle_brand"] = brand
                break
        
        # Years
        year_pattern = r'\b(19|20)\d{2}\b'
        year_match = re.search(year_pattern, content)
        if year_match:
            entities["vehicle_year"] = year_match.group()
        
        # Location indicators
        location_words = ["street", "road", "avenue", "boulevard", "highway", "exit", "mile"]
        if any(word in content_lower for word in location_words):
            entities["mentions_location"] = True
            
        return entities

    async def _analyze_sentiment(self, content: str) -> str:
        """Simple sentiment analysis"""
        content_lower = content.lower()
        
        # Positive indicators
        positive_words = ["thanks", "great", "perfect", "good", "appreciate", "helpful"]
        if any(word in content_lower for word in positive_words):
            return "positive"
        
        # Negative indicators  
        negative_words = ["frustrated", "angry", "terrible", "awful", "bad", "horrible", "worst"]
        if any(word in content_lower for word in negative_words):
            return "negative"
        
        # Stress/urgency indicators
        stress_words = ["urgent", "emergency", "stranded", "stuck", "help", "please"]
        if any(word in content_lower for word in stress_words):
            return "stressed"
            
        return "neutral"

    async def _compress_conversation(self, memory: ConversationMemory) -> None:
        """Compress long conversations using LLM summarization"""
        
        if len(memory.turns) <= self.compression_threshold:
            return
            
        try:
            # Get conversation to compress (keep recent turns, compress older ones)
            turns_to_compress = memory.turns[:-10]  # Keep last 10 turns
            recent_turns = memory.turns[-10:]
            
            # Create conversation text for summarization
            conversation_text = []
            for turn in turns_to_compress:
                conversation_text.append(f"{turn.speaker}: {turn.content}")
            
            # Summarize using LLM
            summary_prompt = f"""Summarize this roadside assistance conversation. Focus on:
1. Customer information collected (name, phone, location, vehicle)
2. Service requested and problem description
3. Important details and progress made
4. Current status

Conversation:
{chr(10).join(conversation_text)}

Provide a concise summary in 2-3 sentences."""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=150,
                temperature=0.1
            )
            
            summary_text = response.choices[0].message.content
            
            # Replace compressed turns with summary
            summary_turn = ConversationTurn(
                timestamp=time.time(),
                speaker="system",
                content=f"[CONVERSATION SUMMARY]: {summary_text}",
                intent="summary"
            )
            
            memory.turns = [summary_turn] + recent_turns
            logger.info(f"🗜️ Compressed conversation for session: {memory.session_id}")
            
        except Exception as e:
            logger.error(f"❌ Conversation compression failed: {e}")

    def _cleanup_old_sessions(self) -> None:
        """Remove old inactive sessions from memory"""
        current_time = time.time()
        sessions_to_remove = []
        
        for session_id, memory in self.active_memories.items():
            # Remove sessions older than 1 hour
            if current_time - memory.last_updated > 3600:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            # Save to database before removing
            self._save_session_to_db(self.active_memories[session_id])
            del self.active_memories[session_id]
            
        if sessions_to_remove:
            logger.info(f"🧹 Cleaned up {len(sessions_to_remove)} old sessions")

    def _save_session_to_db(self, memory: ConversationMemory) -> None:
        """Save conversation memory to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Save session summary
                if memory.summary:
                    conn.execute("""
                        INSERT OR REPLACE INTO conversation_sessions 
                        (session_id, participant_name, phone_number, start_time, end_time, 
                         total_turns, outcome, summary_json, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        memory.session_id,
                        memory.summary.participant_name,
                        memory.summary.phone_number,
                        memory.summary.start_time,
                        memory.summary.end_time,
                        memory.summary.total_turns,
                        memory.summary.outcome,
                        json.dumps(asdict(memory.summary)),
                        memory.created_at
                    ))
                
                # Save turns
                for turn in memory.turns:
                    conn.execute("""
                        INSERT OR REPLACE INTO conversation_turns 
                        (session_id, timestamp, speaker, content, intent, entities_json)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        memory.session_id,
                        turn.timestamp,
                        turn.speaker,
                        turn.content,
                        turn.intent,
                        json.dumps(turn.entities)
                    ))
                
                # Save context variables
                for key, value in memory.context_variables.items():
                    conn.execute("""
                        INSERT OR REPLACE INTO context_variables 
                        (session_id, variable_name, variable_value, timestamp)
                        VALUES (?, ?, ?, ?)
                    """, (
                        memory.session_id,
                        key,
                        json.dumps(value),
                        time.time()
                    ))
                
                conn.commit()
                logger.debug(f"💾 Saved session {memory.session_id} to database")
                
        except Exception as e:
            logger.error(f"❌ Failed to save session to database: {e}")

    def _load_session_from_db(self, session_id: str) -> Optional[ConversationMemory]:
        """Load conversation memory from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load session data
                session_row = conn.execute("""
                    SELECT * FROM conversation_sessions WHERE session_id = ?
                """, (session_id,)).fetchone()
                
                if not session_row:
                    return None
                
                # Load turns
                turn_rows = conn.execute("""
                    SELECT timestamp, speaker, content, intent, entities_json 
                    FROM conversation_turns 
                    WHERE session_id = ? ORDER BY timestamp
                """, (session_id,)).fetchall()
                
                # Load context variables
                context_rows = conn.execute("""
                    SELECT variable_name, variable_value 
                    FROM context_variables WHERE session_id = ?
                """, (session_id,)).fetchall()
                
                # Reconstruct memory object
                memory = ConversationMemory(session_id=session_id)
                
                # Add turns
                for row in turn_rows:
                    turn = ConversationTurn(
                        timestamp=row[0],
                        speaker=row[1],
                        content=row[2],
                        intent=row[3],
                        entities=json.loads(row[4]) if row[4] else {}
                    )
                    memory.turns.append(turn)
                
                # Add context variables
                for var_name, var_value in context_rows:
                    memory.context_variables[var_name] = json.loads(var_value)
                
                # Add summary if available
                if session_row[7]:  # summary_json
                    summary_data = json.loads(session_row[7])
                    memory.summary = ConversationSummary(**summary_data)
                
                logger.debug(f"📖 Loaded session {session_id} from database")
                return memory
                
        except Exception as e:
            logger.error(f"❌ Failed to load session from database: {e}")
            return None

    async def finalize_session(
        self, 
        session_id: str, 
        outcome: str = "completed"
    ) -> ConversationSummary:
        """Finalize conversation session with summary"""
        
        memory = self.get_or_create_session(session_id)
        
        if not memory.turns:
            logger.warning(f"⚠️ No turns found for session {session_id}")
            return None
        
        try:
            # Extract key information from conversation
            key_info = self._extract_key_information(memory)
            
            # Generate conversation summary
            summary = ConversationSummary(
                session_id=session_id,
                start_time=memory.created_at,
                end_time=time.time(),
                participant_name=memory.context_variables.get("caller_name"),
                phone_number=memory.context_variables.get("phone_number"),
                total_turns=len(memory.turns),
                main_topics=await self._extract_main_topics(memory),
                outcome=outcome,
                key_information=key_info,
                satisfaction_score=await self._estimate_satisfaction(memory)
            )
            
            memory.summary = summary
            
            # Save to database
            self._save_session_to_db(memory)
            
            # Remove from active memory
            if session_id in self.active_memories:
                del self.active_memories[session_id]
            
            logger.info(f"✅ Finalized session {session_id}: {outcome}")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Session finalization failed: {e}")
            return None

    def _extract_key_information(self, memory: ConversationMemory) -> Dict[str, Any]:
        """Extract key information from conversation"""
        key_info = {}
        
        # Get from context variables
        key_info.update(memory.context_variables)
        
        # Extract from conversation turns
        for turn in memory.turns:
            if turn.entities:
                key_info.update(turn.entities)
        
        # Clean and organize
        organized_info = {
            "customer": {
                "name": key_info.get("caller_name"),
                "phone": key_info.get("phone_number")
            },
            "location": key_info.get("location"),
            "vehicle": {
                "year": key_info.get("vehicle_year"),
                "make": key_info.get("vehicle_make") or key_info.get("vehicle_brand"),
                "model": key_info.get("vehicle_model"),
                "color": key_info.get("vehicle_color")
            },
            "service": {
                "type": key_info.get("service_type"),
                "description": key_info.get("issue_description"),
                "urgency": key_info.get("urgency_level", "normal")
            }
        }
        
        return organized_info

    async def _extract_main_topics(self, memory: ConversationMemory) -> List[str]:
        """Extract main topics from conversation"""
        topics = set()
        
        for turn in memory.turns:
            if turn.intent and turn.intent != "general_inquiry":
                # Convert intent to human-readable topic
                topic_map = {
                    "request_towing": "Towing Service",
                    "request_battery": "Battery/Jump Start",
                    "request_tire": "Tire Service",
                    "request_fuel": "Fuel Delivery",
                    "request_lockout": "Lockout Service",
                    "emergency": "Emergency Assistance",
                    "question_pricing": "Pricing Inquiry"
                }
                topic = topic_map.get(turn.intent, turn.intent.replace("_", " ").title())
                topics.add(topic)
        
        return list(topics)

    async def _estimate_satisfaction(self, memory: ConversationMemory) -> Optional[float]:
        """Estimate customer satisfaction from conversation"""
        satisfaction_indicators = {
            "positive": 0.8,
            "neutral": 0.6,
            "negative": 0.2,
            "stressed": 0.4
        }
        
        sentiments = [turn.sentiment for turn in memory.turns if turn.sentiment and turn.speaker == "user"]
        
        if not sentiments:
            return None
        
        # Weight recent sentiments more heavily
        weighted_scores = []
        for i, sentiment in enumerate(sentiments):
            weight = (i + 1) / len(sentiments)  # Recent turns have higher weight
            score = satisfaction_indicators.get(sentiment, 0.5)
            weighted_scores.append(score * weight)
        
        return sum(weighted_scores) / len(weighted_scores) if weighted_scores else None

    async def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get analytics for a conversation session"""
        memory = self.get_or_create_session(session_id)
        
        if not memory.turns:
            return {}
        
        analytics = {
            "session_duration": time.time() - memory.created_at,
            "total_turns": len(memory.turns),
            "user_turns": len([t for t in memory.turns if t.speaker == "user"]),
            "agent_turns": len([t for t in memory.turns if t.speaker == "agent"]),
            "intents_detected": list(set([t.intent for t in memory.turns if t.intent])),
            "sentiment_distribution": self._get_sentiment_distribution(memory),
            "information_completeness": self._assess_information_completeness(memory),
            "conversation_efficiency": self._calculate_efficiency(memory)
        }
        
        return analytics

    def _get_sentiment_distribution(self, memory: ConversationMemory) -> Dict[str, int]:
        """Get distribution of sentiments in conversation"""
        sentiments = [turn.sentiment for turn in memory.turns if turn.sentiment and turn.speaker == "user"]
        distribution = {}
        for sentiment in sentiments:
            distribution[sentiment] = distribution.get(sentiment, 0) + 1
        return distribution

    def _assess_information_completeness(self, memory: ConversationMemory) -> float:
        """Assess how complete the gathered information is"""
        required_fields = ["caller_name", "phone_number", "location", "service_type"]
        gathered_fields = 0
        
        for field in required_fields:
            if memory.context_variables.get(field):
                gathered_fields += 1
        
        return gathered_fields / len(required_fields)

    def _calculate_efficiency(self, memory: ConversationMemory) -> float:
        """Calculate conversation efficiency (information per turn)"""
        if not memory.turns:
            return 0.0
        
        information_turns = len([t for t in memory.turns if t.entities or t.intent in [
            "provide_location", "provide_vehicle_info", "request_towing"
        ]])
        
        return information_turns / len(memory.turns)

# Global memory manager instance
memory_manager = ConversationMemoryManager()

async def get_memory_manager() -> ConversationMemoryManager:
    """Get the global memory manager instance"""
    return memory_manager