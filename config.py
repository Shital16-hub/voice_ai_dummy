# config.py - OPTIMIZED FOR LOW LATENCY
"""
Ultra-Optimized Configuration for LiveKit RAG Agent with <2s response time
UPDATED: Aggressive performance settings for telephony
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class OptimizedConfig(BaseSettings):
    """Configuration optimized for ultra-low latency telephony"""
    
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
    
    # 🚀 ULTRA-LOW LATENCY QDRANT SETTINGS
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(default="telephony_knowledge", env="QDRANT_COLLECTION")
    qdrant_timeout: int = Field(default=2, env="QDRANT_TIMEOUT")  # Very short
    
    # 🚀 AGGRESSIVE PERFORMANCE SETTINGS
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")  # Fastest
    embedding_dimensions: int = Field(default=1536, env="EMBEDDING_DIMENSIONS") 
    
    # 🚀 ULTRA-FAST RAG SETTINGS
    rag_timeout_ms: int = Field(default=300, env="RAG_TIMEOUT_MS")  # 300ms!
    search_limit: int = Field(default=1, env="SEARCH_LIMIT")  # Single result
    similarity_threshold: float = Field(default=0.2, env="SIMILARITY_THRESHOLD")  # Lower for speed
    
    # 🚀 CHUNK SETTINGS OPTIMIZED FOR TELEPHONY
    chunk_size: int = Field(default=200, env="CHUNK_SIZE")  # Smaller chunks
    chunk_overlap: int = Field(default=20, env="CHUNK_OVERLAP")  # Less overlap
    max_tokens: int = Field(default=30, env="MAX_TOKENS")  # Very short responses
    
    # 🚀 EMBEDDING CACHE FOR MASSIVE SPEED IMPROVEMENT
    enable_embedding_cache: bool = Field(default=True, env="ENABLE_EMBEDDING_CACHE")
    embedding_cache_size: int = Field(default=200, env="EMBEDDING_CACHE_SIZE")  # Smaller cache
    
    # 🚀 QDRANT SEARCH OPTIMIZATION
    qdrant_exact_search: bool = Field(default=False, env="QDRANT_EXACT_SEARCH")  # Approximate for speed
    qdrant_hnsw_ef: int = Field(default=16, env="QDRANT_HNSW_EF")  # Very low for speed
    
    # 🚀 TELEPHONY-SPECIFIC ULTRA-FAST SETTINGS
    max_response_length: int = Field(default=80, env="MAX_RESPONSE_LENGTH")  # Very short
    enable_response_streaming: bool = Field(default=True, env="ENABLE_RESPONSE_STREAMING")
    
    # 🚀 DOCKER OPTIMIZATION
    use_local_docker: bool = Field(default=True, env="USE_LOCAL_DOCKER")
    docker_health_check_retries: int = Field(default=2, env="DOCKER_HEALTH_CHECK_RETRIES")  # Fewer retries
    
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
            response = requests.get(f"{self.qdrant_url}/", timeout=1)  # Very short timeout
            return response.status_code == 200
        except:
            return False
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

# Global configuration instance
config = OptimizedConfig()
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
    
    print("✅ ULTRA-LOW LATENCY Configuration")
    print(f"📞 Transfer destination: {config.transfer_sip_address}")
    print(f"🔍 Qdrant URL: {config.qdrant_url}")
    print(f"⚡ RAG timeout: {config.rag_timeout_ms}ms (ULTRA-FAST!)")
    print(f"🔍 Search limit: {config.search_limit} (SINGLE RESULT)")
    print(f"📊 Similarity threshold: {config.similarity_threshold} (OPTIMIZED)")
    print(f"🧠 Max response tokens: {config.max_tokens} (VERY SHORT)")
    print(f"🚀 Embedding cache: {config.enable_embedding_cache}")
    print(f"🎯 Target: <2 second total response time")
    
    # Check Docker health
    if config.use_local_docker:
        if config.is_docker_healthy():
            print("✅ Qdrant Docker container is healthy")
        else:
            print("⚠️  Warning: Qdrant Docker container not responding")
            print("   Run: docker-compose up -d to start Qdrant")

if __name__ == "__main__":
    validate_config()