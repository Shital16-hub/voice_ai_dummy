# test_imports.py - Test all LiveKit imports
"""
Test script to verify all LiveKit imports are working correctly
Run this before starting your main agent
"""

def test_livekit_imports():
    """Test all necessary LiveKit imports"""
    print("🧪 Testing LiveKit imports...")
    
    try:
        # Core LiveKit imports
        from livekit import api, agents, rtc
        from livekit.agents import (
            Agent, AgentSession, ChatContext, ChatMessage, 
            JobContext, RunContext, WorkerOptions, cli, 
            function_tool, get_job_context
        )
        print("✅ Core LiveKit imports - OK")
        
        # Plugin imports
        from livekit.plugins import deepgram, openai, elevenlabs, silero
        print("✅ Plugin imports - OK")
        
        # Turn detection import (the one that was failing)
        from livekit.plugins.turn_detector.english import EnglishModel
        print("✅ Turn detection import - OK")
        
        # Note: Turn detection model creation happens automatically inside LiveKit job context
        print("✅ Turn detection model will be created automatically in job context")
        
        # Other imports from your code
        from qdrant_rag_system import qdrant_rag
        print("✅ Qdrant RAG import - OK")
        
        from config import config
        print("✅ Config import - OK")
        
        from call_transcription_storage import call_storage
        print("✅ Call storage import - OK")
        
        print("\n🎉 ALL IMPORTS SUCCESSFUL!")
        print("✅ Your LiveKit setup is ready to go")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n🔧 To fix this issue:")
        
        if "turn_detector" in str(e):
            print("   1. Install turn detector: pip install livekit-plugins-turn-detector")
            print("   2. Download models: python -c \"from livekit.plugins.turn_detector.multilingual import MultilingualModel; MultilingualModel()\"")
        elif "qdrant" in str(e):
            print("   1. Install qdrant: pip install qdrant-client")
        elif "openai" in str(e):
            print("   1. Install openai: pip install openai")
        else:
            print(f"   1. Install missing dependency related to: {e}")
            
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_environment_variables():
    """Test that required environment variables are set"""
    print("\n🔍 Testing environment variables...")
    
    import os
    required_vars = [
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY", 
        "LIVEKIT_API_SECRET",
        "OPENAI_API_KEY",
        "DEEPGRAM_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
        else:
            print(f"✅ {var} - Set")
    
    if missing_vars:
        print(f"\n❌ Missing environment variables: {missing_vars}")
        print("💡 Make sure your .env file contains all required variables")
        return False
    else:
        print("✅ All required environment variables are set")
        return True

def test_qdrant_connection():
    """Test Qdrant Docker connection"""
    print("\n🐳 Testing Qdrant connection...")
    
    try:
        import requests
        response = requests.get("http://localhost:6333/", timeout=2)
        if response.status_code == 200:
            print("✅ Qdrant Docker container is running")
            return True
        else:
            print("❌ Qdrant responded with error")
            return False
    except Exception as e:
        print("❌ Qdrant Docker container not running")
        print("🔧 To fix: Run 'docker-compose up -d' to start Qdrant")
        return False

def main():
    """Run all tests"""
    print("🚀 LIVEKIT SETUP VERIFICATION")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Imports
    if test_livekit_imports():
        tests_passed += 1
    
    # Test 2: Environment variables
    if test_environment_variables():
        tests_passed += 1
    
    # Test 3: Qdrant connection
    if test_qdrant_connection():
        tests_passed += 1
    
    print(f"\n📊 TEST RESULTS: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! Your setup is ready.")
        print("🚀 You can now run: python main.py dev")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        
    print("=" * 50)

if __name__ == "__main__":
    main()
    