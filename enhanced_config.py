# enhanced_config.py
"""
Enhanced Configuration for Multi-Agent LiveKit Voice System
Supports multiple agents, conversation memory, and advanced features
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class EnhancedLiveKitConfig(BaseSettings):
    """Enhanced configuration for multi-agent voice AI system"""
    
    # ✅ REQUIRED: LiveKit Settings
    livekit_url: str = Field(default="", env="LIVEKIT_URL")
    livekit_api_key: str = Field(default="", env="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(default="", env="LIVEKIT_API_SECRET")
    
    # ✅ REQUIRED: AI Service API Keys
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    deepgram_api_key: str = Field(default="", env="DEEPGRAM_API_KEY")
    
    # ✅ ENHANCED TTS OPTIONS
    elevenlabs_api_key: Optional[str] = Field(default=None, env="ELEVENLABS_API_KEY")
    cartesia_api_key: Optional[str] = Field(default=None, env="CARTESIA_API_KEY")
    
    # ✅ GOOGLE CLOUD STT (Optional)
    google_credentials_file: Optional[str] = Field(default=None, env="GOOGLE_APPLICATION_CREDENTIALS")
    google_project_id: Optional[str] = Field(default=None, env="GOOGLE_CLOUD_PROJECT")
    google_location: str = Field(default="us-central1", env="GOOGLE_CLOUD_LOCATION")
    
    # ✅ TELEPHONY INTEGRATION
    twilio_account_sid: Optional[str] = Field(default=None, env="TWILIO_ACCOUNT_SID")
    twilio_phone_number: Optional[str] = Field(default=None, env="TWILIO_PHONE_NUMBER")
    twilio_auth_token: Optional[str] = Field(default=None, env="TWILIO_AUTH_TOKEN")
    transfer_sip_address: str = Field(default="sip:voiceai@sip.linphone.org", env="TRANSFER_SIP_ADDRESS")
    
    # ✅ QDRANT VECTOR DATABASE
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(default="telephony_knowledge", env="QDRANT_COLLECTION")
    
    # 🚀 CONVERSATION & MEMORY SETTINGS
    enable_conversation_memory: bool = Field(default=True, env="ENABLE_CONVERSATION_MEMORY")
    memory_database_path: str = Field(default="conversation_memory.db", env="MEMORY_DATABASE_PATH")
    conversation_compression_threshold: int = Field(default=20, env="CONVERSATION_COMPRESSION_THRESHOLD")
    max_active_sessions: int = Field(default=100, env="MAX_ACTIVE_SESSIONS")
    session_timeout_minutes: int = Field(default=60, env="SESSION_TIMEOUT_MINUTES")
    
    # 🚀 MULTI-AGENT SETTINGS
    enable_multi_agent: bool = Field(default=True, env="ENABLE_MULTI_AGENT")
    agent_routing_threshold: float = Field(default=0.7, env="AGENT_ROUTING_THRESHOLD")
    max_handoff_attempts: int = Field(default=3, env="MAX_HANDOFF_ATTEMPTS")
    specialist_timeout_seconds: int = Field(default=30, env="SPECIALIST_TIMEOUT_SECONDS")
    
    # 🚀 CONVERSATION FLOW SETTINGS
    natural_conversation_flow: bool = Field(default=True, env="NATURAL_CONVERSATION_FLOW")
    auto_context_injection: bool = Field(default=True, env="AUTO_CONTEXT_INJECTION")
    conversation_stage_tracking: bool = Field(default=True, env="CONVERSATION_STAGE_TRACKING")
    dynamic_instruction_updates: bool = Field(default=True, env="DYNAMIC_INSTRUCTION_UPDATES")
    
    # 🚀 PERFORMANCE OPTIMIZATION
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=1536, env="EMBEDDING_DIMENSIONS")
    embedding_batch_size: int = Field(default=10, env="EMBEDDING_BATCH_SIZE")
    enable_embedding_cache: bool = Field(default=True, env="ENABLE_EMBEDDING_CACHE")
    embedding_cache_size: int = Field(default=1000, env="EMBEDDING_CACHE_SIZE")
    
    # 🚀 RAG OPTIMIZATION
    rag_timeout_ms: int = Field(default=800, env="RAG_TIMEOUT_MS")
    search_limit: int = Field(default=2, env="SEARCH_LIMIT")
    similarity_threshold: float = Field(default=0.25, env="SIMILARITY_THRESHOLD")  # Lower for better coverage
    max_response_length: int = Field(default=150, env="MAX_RESPONSE_LENGTH")
    enable_response_streaming: bool = Field(default=True, env="ENABLE_RESPONSE_STREAMING")
    
    # 🚀 VOICE PROCESSING SETTINGS
    voice_model: str = Field(default="eleven_turbo_v2_5", env="VOICE_MODEL")
    voice_id: str = Field(default="21m00Tcm4TlvDq8ikWAM", env="VOICE_ID")  # Professional male voice
    voice_stability: float = Field(default=0.7, env="VOICE_STABILITY")
    voice_similarity_boost: float = Field(default=0.8, env="VOICE_SIMILARITY_BOOST")
    voice_style: float = Field(default=0.1, env="VOICE_STYLE")
    voice_speed: float = Field(default=0.95, env="VOICE_SPEED")
    
    # 🚀 STT OPTIMIZATION
    stt_model: str = Field(default="nova-2-general", env="STT_MODEL")
    stt_language: str = Field(default="en-US", env="STT_LANGUAGE")
    enable_smart_format: bool = Field(default=True, env="ENABLE_SMART_FORMAT")
    enable_profanity_filter: bool = Field(default=False, env="ENABLE_PROFANITY_FILTER")
    enable_numerals: bool = Field(default=True, env="ENABLE_NUMERALS")
    
    # 🚀 LLM SETTINGS
    llm_model: str = Field(default="gpt-4o-mini", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.2, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=100, env="LLM_MAX_TOKENS")
    
    # 🚀 TURN DETECTION & CONVERSATION FLOW
    min_interruption_duration: float = Field(default=0.6, env="MIN_INTERRUPTION_DURATION")
    min_endpointing_delay: float = Field(default=0.8, env="MIN_ENDPOINTING_DELAY")
    max_endpointing_delay: float = Field(default=4.0, env="MAX_ENDPOINTING_DELAY")
    allow_interruptions: bool = Field(default=True, env="ALLOW_INTERRUPTIONS")
    
    # 🚀 AGENT SPECIALIZATION SETTINGS
    towing_specialist_enabled: bool = Field(default=True, env="TOWING_SPECIALIST_ENABLED")
    battery_specialist_enabled: bool = Field(default=True, env="BATTERY_SPECIALIST_ENABLED")
    tire_specialist_enabled: bool = Field(default=True, env="TIRE_SPECIALIST_ENABLED")
    emergency_response_enabled: bool = Field(default=True, env="EMERGENCY_RESPONSE_ENABLED")
    insurance_specialist_enabled: bool = Field(default=True, env="INSURANCE_SPECIALIST_ENABLED")
    
    # 🚀 ANALYTICS & MONITORING
    enable_conversation_analytics: bool = Field(default=True, env="ENABLE_CONVERSATION_ANALYTICS")
    enable_performance_monitoring: bool = Field(default=True, env="ENABLE_PERFORMANCE_MONITORING")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    enable_detailed_logging: bool = Field(default=False, env="ENABLE_DETAILED_LOGGING")
    
    # 🚀 KNOWLEDGE BASE SETTINGS
    knowledge_auto_update: bool = Field(default=True, env="KNOWLEDGE_AUTO_UPDATE")
    knowledge_cache_ttl_minutes: int = Field(default=60, env="KNOWLEDGE_CACHE_TTL_MINUTES")
    enable_multi_query_search: bool = Field(default=True, env="ENABLE_MULTI_QUERY_SEARCH")
    
    # ✅ PATHS AND DIRECTORIES
    @property
    def project_root(self) -> Path:
        return Path(__file__).parent
    
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"
    
    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"
    
    @property
    def memory_dir(self) -> Path:
        return self.project_root / "memory"
    
    @property
    def qdrant_storage_dir(self) -> Path:
        return self.project_root / "qdrant_storage"
    
    def ensure_directories(self):
        """Create necessary directories"""
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.memory_dir.mkdir(exist_ok=True)
        self.qdrant_storage_dir.mkdir(exist_ok=True)
    
    def get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """Get configuration for specific agent type"""
        base_config = {
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "voice_model": self.voice_model,
            "voice_id": self.voice_id,
            "stt_model": self.stt_model,
            "enable_memory": self.enable_conversation_memory,
            "rag_timeout": self.rag_timeout_ms,
            "similarity_threshold": self.similarity_threshold
        }
        
        # Agent-specific configurations
        agent_configs = {
            "dispatcher": {
                "temperature": 0.2,
                "max_tokens": 80,
                "routing_enabled": True
            },
            "towing_specialist": {
                "temperature": 0.3,
                "max_tokens": 120,
                "specialization": "towing",
                "knowledge_prefix": "towing"
            },
            "battery_specialist": {
                "temperature": 0.3,
                "max_tokens": 100,
                "specialization": "battery",
                "knowledge_prefix": "battery jumpstart electrical"
            },
            "tire_specialist": {
                "temperature": 0.3,
                "max_tokens": 100,
                "specialization": "tire",
                "knowledge_prefix": "tire flat spare wheel"
            },
            "emergency_response": {
                "temperature": 0.1,  # More deterministic for emergencies
                "max_tokens": 60,
                "priority": "emergency",
                "response_speed": "fast"
            },
            "insurance_specialist": {
                "temperature": 0.2,
                "max_tokens": 120,
                "specialization": "insurance",
                "knowledge_prefix": "insurance coverage membership policy"
            }
        }
        
        agent_specific = agent_configs.get(agent_type, {})
        return {**base_config, **agent_specific}
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a specific feature is enabled"""
        feature_flags = {
            "multi_agent": self.enable_multi_agent,
            "conversation_memory": self.enable_conversation_memory,
            "natural_conversation": self.natural_conversation_flow,
            "auto_context": self.auto_context_injection,
            "stage_tracking": self.conversation_stage_tracking,
            "dynamic_instructions": self.dynamic_instruction_updates,
            "analytics": self.enable_conversation_analytics,
            "performance_monitoring": self.enable_performance_monitoring,
            "towing_specialist": self.towing_specialist_enabled,
            "battery_specialist": self.battery_specialist_enabled,
            "tire_specialist": self.tire_specialist_enabled,
            "emergency_response": self.emergency_response_enabled,
            "insurance_specialist": self.insurance_specialist_enabled
        }
        
        return feature_flags.get(feature, False)
    
    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory system configuration"""
        return {
            "enabled": self.enable_conversation_memory,
            "database_path": self.memory_database_path,
            "compression_threshold": self.conversation_compression_threshold,
            "max_active_sessions": self.max_active_sessions,
            "session_timeout_minutes": self.session_timeout_minutes
        }
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

# Global configuration instance
enhanced_config = EnhancedLiveKitConfig()
enhanced_config.ensure_directories()

def validate_enhanced_config():
    """Validate enhanced configuration"""
    required_fields = [
        ("LIVEKIT_URL", enhanced_config.livekit_url),
        ("LIVEKIT_API_KEY", enhanced_config.livekit_api_key),
        ("LIVEKIT_API_SECRET", enhanced_config.livekit_api_secret),
        ("OPENAI_API_KEY", enhanced_config.openai_api_key),
        ("DEEPGRAM_API_KEY", enhanced_config.deepgram_api_key),
    ]
    
    missing_fields = [field for field, value in required_fields if not value]
    
    if missing_fields:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_fields)}")
    
    print("✅ Enhanced Configuration Validated")
    print(f"🎯 Multi-Agent System: {'Enabled' if enhanced_config.enable_multi_agent else 'Disabled'}")
    print(f"🧠 Conversation Memory: {'Enabled' if enhanced_config.enable_conversation_memory else 'Disabled'}")
    print(f"📞 Natural Conversation Flow: {'Enabled' if enhanced_config.natural_conversation_flow else 'Disabled'}")
    print(f"🔄 Auto Context Injection: {'Enabled' if enhanced_config.auto_context_injection else 'Disabled'}")
    print(f"📊 Conversation Analytics: {'Enabled' if enhanced_config.enable_conversation_analytics else 'Disabled'}")
    
    # Specialist agents status
    specialists = []
    if enhanced_config.towing_specialist_enabled:
        specialists.append("Towing")
    if enhanced_config.battery_specialist_enabled:
        specialists.append("Battery")
    if enhanced_config.tire_specialist_enabled:
        specialists.append("Tire")
    if enhanced_config.emergency_response_enabled:
        specialists.append("Emergency")
    if enhanced_config.insurance_specialist_enabled:
        specialists.append("Insurance")
    
    print(f"🎭 Specialist Agents: {', '.join(specialists) if specialists else 'None'}")
    
    # Performance settings
    print(f"⚡ RAG Timeout: {enhanced_config.rag_timeout_ms}ms")
    print(f"🔍 Search Limit: {enhanced_config.search_limit}")
    print(f"📈 Similarity Threshold: {enhanced_config.similarity_threshold}")
    print(f"🎙️ Voice Model: {enhanced_config.voice_model}")
    print(f"🗣️ STT Model: {enhanced_config.stt_model}")
    print(f"🤖 LLM Model: {enhanced_config.llm_model}")

if __name__ == "__main__":
    validate_enhanced_config()