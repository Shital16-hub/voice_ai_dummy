# config.py - FINAL FIXED VERSION
"""
FINAL FIXED Configuration - addresses all issues from logs
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class FixedConfig(BaseSettings):
    """FINAL FIXED Configuration based on log analysis"""
    
    # ✅ REQUIRED: LiveKit Settings
    livekit_url: str = Field(default="", env="LIVEKIT_URL")
    livekit_api_key: str = Field(default="", env="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(default="", env="LIVEKIT_API_SECRET")
    
    # ✅ REQUIRED: AI Service API Keys
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    deepgram_api_key: str = Field(default="", env="DEEPGRAM_API_KEY")
    
    # ✅ OPTIONAL: Enhanced TTS
    elevenlabs_api_key: Optional[str] = Field(default=None, env="ELEVENLABS_API_KEY")
    cartesia_api_key: Optional[str] = Field(default=None, env="CARTESIA_API_KEY")
    
    # ✅ TELEPHONY INTEGRATION
    transfer_sip_address: str = Field(
        default="sip:voiceai@sip.linphone.org", 
        env="TRANSFER_SIP_ADDRESS"
    )
    
    # 🚀 FIXED QDRANT SETTINGS
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(default="telephony_knowledge", env="QDRANT_COLLECTION")
    qdrant_timeout: int = Field(default=10, env="QDRANT_TIMEOUT")  # Increased for reliability
    
    # 🚀 FIXED PERFORMANCE SETTINGS
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=1536, env="EMBEDDING_DIMENSIONS") 
    
    # 🚀 FIXED RAG SETTINGS (Based on your test showing excellent results at 0.2-0.8 range)
    rag_timeout_ms: int = Field(default=3000, env="RAG_TIMEOUT_MS")  # Increased significantly
    search_limit: int = Field(default=5, env="SEARCH_LIMIT")  # Increased for better coverage
    similarity_threshold: float = Field(default=0.15, env="SIMILARITY_THRESHOLD")  # Lowered more
    
    # 🚀 MULTI-LEVEL THRESHOLDS (From your test data analysis)
    high_confidence_threshold: float = Field(default=0.5, env="HIGH_CONFIDENCE_THRESHOLD")  
    medium_confidence_threshold: float = Field(default=0.3, env="MEDIUM_CONFIDENCE_THRESHOLD")  
    minimum_usable_threshold: float = Field(default=0.15, env="MINIMUM_USABLE_THRESHOLD")  
    
    # 🚀 ENHANCED SETTINGS
    chunk_size: int = Field(default=400, env="CHUNK_SIZE")  # Larger for better context
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    max_tokens: int = Field(default=80, env="MAX_TOKENS")  # Increased for complete information
    
    # 🚀 CACHE SETTINGS
    enable_embedding_cache: bool = Field(default=True, env="ENABLE_EMBEDDING_CACHE")
    embedding_cache_size: int = Field(default=1000, env="EMBEDDING_CACHE_SIZE")  # Much larger
    
    # 🚀 VOICE SETTINGS
    max_response_length: int = Field(default=200, env="MAX_RESPONSE_LENGTH")  # Increased
    enable_response_streaming: bool = Field(default=True, env="ENABLE_RESPONSE_STREAMING")
    
    # 🚀 TRANSFER SETTINGS
    auto_transfer_disabled: bool = Field(default=True, env="AUTO_TRANSFER_DISABLED")
    require_transfer_confirmation: bool = Field(default=True, env="REQUIRE_TRANSFER_CONFIRMATION")
    
    # ✅ PATHS
    @property
    def project_root(self) -> Path:
        return Path(__file__).parent
    
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"
    
    @property
    def qdrant_storage_dir(self) -> Path:
        return self.project_root / "qdrant_storage"
    
    def ensure_directories(self):
        """Create necessary directories"""
        self.data_dir.mkdir(exist_ok=True)
        self.qdrant_storage_dir.mkdir(exist_ok=True)
    
    def is_docker_healthy(self) -> bool:
        """Check if Qdrant Docker container is healthy"""
        try:
            import requests
            response = requests.get(f"{self.qdrant_url}/", timeout=3)
            return response.status_code == 200
        except:
            return False
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

# Global configuration instance
config = FixedConfig()
config.ensure_directories()

def validate_config():
    """Validate essential configuration"""
    required_fields = [
        ("OPENAI_API_KEY", config.openai_api_key),
        ("DEEPGRAM_API_KEY", config.deepgram_api_key),
    ]
    
    missing_fields = [field for field, value in required_fields if not value]
    
    if missing_fields:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_fields)}")
    
    print("✅ FINAL FIXED Configuration")
    print(f"📞 Transfer destination: {config.transfer_sip_address}")
    print(f"🔍 Qdrant URL: {config.qdrant_url}")
    print(f"⚡ RAG timeout: {config.rag_timeout_ms}ms (FIXED)")
    print(f"🔍 Search limit: {config.search_limit} (INCREASED)")
    print(f"📊 Similarity threshold: {config.similarity_threshold} (LOWERED)")
    print(f"🎯 High confidence: ≥{config.high_confidence_threshold}")
    print(f"🎯 Medium confidence: ≥{config.medium_confidence_threshold}")
    print(f"🎯 Minimum usable: ≥{config.minimum_usable_threshold}")
    print(f"🧠 Max response: {config.max_response_length} chars")
    print(f"🚫 Auto transfer: {'DISABLED' if config.auto_transfer_disabled else 'ENABLED'}")
    
    # Check Docker health
    if config.is_docker_healthy():
        print("✅ Qdrant Docker container is healthy")
    else:
        print("⚠️ Warning: Qdrant Docker container not responding")

if __name__ == "__main__":
    validate_config()