# config_check.py - Check and fix configuration issues
"""
Configuration validation and fix script
Run this before starting the agent to ensure all settings are correct
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_and_fix_config():
    """Check and fix configuration issues"""
    print("🔧 CONFIGURATION CHECK AND FIX")
    print("=" * 50)
    
    issues = []
    fixes = []
    
    # Check required environment variables
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for embeddings and LLM",
        "DEEPGRAM_API_KEY": "Deepgram API key for speech-to-text",
        "ELEVENLABS_API_KEY": "ElevenLabs API key for text-to-speech",
        "LIVEKIT_URL": "LiveKit server URL",
        "LIVEKIT_API_KEY": "LiveKit API key",
        "LIVEKIT_API_SECRET": "LiveKit API secret"
    }
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            issues.append(f"❌ Missing {var} ({description})")
        elif var == "OPENAI_API_KEY" and (len(value) < 20 or not value.startswith("sk-")):
            issues.append(f"❌ Invalid {var} format (should start with 'sk-')")
        else:
            print(f"✅ {var}: {'*' * 10 + value[-4:] if len(value) > 10 else 'Set'}")
    
    # Check .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        issues.append("❌ .env file not found")
        fixes.append("Create .env file with required variables")
    
    # Check Qdrant connection
    try:
        import requests
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        response = requests.get(f"{qdrant_url}/", timeout=2)
        if response.status_code == 200:
            print(f"✅ Qdrant connection: {qdrant_url}")
        else:
            issues.append(f"❌ Qdrant not responding at {qdrant_url}")
            fixes.append("Start Qdrant: docker-compose up -d")
    except Exception as e:
        issues.append(f"❌ Qdrant connection failed: {e}")
        fixes.append("Start Qdrant: docker-compose up -d")
    
    # Check data directory
    data_dir = Path("data")
    if not data_dir.exists():
        issues.append("❌ Data directory not found")
        fixes.append("Create 'data' directory and add knowledge base files")
    elif not list(data_dir.glob("*")):
        issues.append("❌ Data directory is empty")
        fixes.append("Add knowledge base files (Excel, PDF, etc.) to 'data' directory")
    else:
        files = list(data_dir.glob("*"))
        print(f"✅ Data directory: {len(files)} files found")
    
    # Check requirements
    try:
        import livekit.agents
        print(f"✅ LiveKit Agents: {livekit.agents.__version__}")
    except ImportError:
        issues.append("❌ LiveKit Agents not installed")
        fixes.append("Install: pip install livekit-agents[openai,silero,deepgram,elevenlabs,turn-detector]")
    
    try:
        import qdrant_client
        print("✅ Qdrant client installed")
    except ImportError:
        issues.append("❌ Qdrant client not installed")
        fixes.append("Install: pip install qdrant-client")
    
    # Print results
    if issues:
        print("\n❌ ISSUES FOUND:")
        for issue in issues:
            print(f"   {issue}")
        
        print("\n🔧 SUGGESTED FIXES:")
        for fix in fixes:
            print(f"   {fix}")
        
        if not os.getenv("OPENAI_API_KEY"):
            print("\n💡 OPENAI_API_KEY FIX:")
            print("   1. Go to https://platform.openai.com/api-keys")
            print("   2. Create a new API key")
            print("   3. Add to .env file: OPENAI_API_KEY=sk-your-key-here")
        
        return False
    else:
        print("\n✅ ALL CONFIGURATION CHECKS PASSED!")
        return True

def create_sample_env():
    """Create a sample .env file"""
    env_content = """# LiveKit Configuration
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# AI Service API Keys
OPENAI_API_KEY=sk-your_openai_api_key_here
DEEPGRAM_API_KEY=your_deepgram_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=telephony_knowledge

# RAG Settings (optional)
RAG_TIMEOUT_MS=800
SEARCH_LIMIT=2
SIMILARITY_THRESHOLD=0.25

# Voice Settings (optional)
VOICE_ID=21m00Tcm4TlvDq8ikWAM
VOICE_MODEL=eleven_turbo_v2_5
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"✅ Created sample .env file at {env_file}")
        print("   Please edit it with your actual API keys")
    else:
        print("   .env file already exists")

if __name__ == "__main__":
    success = check_and_fix_config()
    
    if not success:
        print("\n" + "=" * 50)
        response = input("Create sample .env file? (y/n): ")
        if response.lower() == 'y':
            create_sample_env()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Ready to run: python main.py dev")
    else:
        print("🔧 Fix the issues above, then run: python main.py dev")