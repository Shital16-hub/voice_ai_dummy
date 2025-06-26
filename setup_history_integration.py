# setup_history_integration.py
"""
Setup script to prepare your system for conversation history integration
"""
import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def check_dependencies():
    """Check if all required dependencies are available"""
    logger.info("🔍 Checking dependencies...")
    
    missing_deps = []
    
    # Check core dependencies
    try:
        import openai
        logger.info("✅ OpenAI SDK available")
    except ImportError:
        missing_deps.append("openai")
    
    try:
        from livekit.plugins import openai as lk_openai
        logger.info("✅ LiveKit OpenAI plugin available")
    except ImportError:
        missing_deps.append("livekit-plugins-openai")
    
    try:
        from call_transcription_storage import call_storage
        logger.info("✅ Call transcription storage available")
    except ImportError:
        missing_deps.append("call_transcription_storage.py")
    
    try:
        from config import config
        logger.info("✅ Config module available")
        if config.openai_api_key:
            logger.info("✅ OpenAI API key configured")
        else:
            logger.warning("⚠️ OpenAI API key not found in config")
    except ImportError:
        missing_deps.append("config.py")
    
    if missing_deps:
        logger.error(f"❌ Missing dependencies: {missing_deps}")
        return False
    
    return True

async def check_database():
    """Check if the call transcription database is working"""
    logger.info("🗄️ Checking database...")
    
    try:
        from call_transcription_storage import call_storage
        
        # Test database connection
        status = {"status": "testing"}
        
        # Try to get recent sessions
        recent_sessions = await call_storage.get_recent_sessions(1)
        logger.info(f"✅ Database accessible, found {len(recent_sessions)} recent sessions")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database check failed: {e}")
        return False

async def create_test_data():
    """Create some test conversation data"""
    logger.info("📝 Creating test conversation data...")
    
    try:
        from call_transcription_storage import call_storage
        
        # Create a test caller
        test_phone = "+1555123456"
        
        session_id, caller_id = await call_storage.start_call_session(
            phone_number=test_phone,
            session_metadata={"test": "history_integration_setup"}
        )
        
        # Add test conversation
        test_conversation = [
            ("user", "Hi, I need help with my car battery"),
            ("agent", "I can help with that. What's your location?"),
            ("user", "I'm at the mall parking lot on Main Street"),
            ("agent", "Got it. I'll send a technician for a battery jump start. That's $40"),
            ("user", "Perfect, thank you"),
            ("agent", "You're welcome! ETA is about 20 minutes")
        ]
        
        for role, content in test_conversation:
            await call_storage.save_conversation_item(
                session_id=session_id,
                caller_id=caller_id,
                role=role,
                content=content,
                metadata={"test": True}
            )
        
        await call_storage.end_call_session(session_id)
        
        logger.info(f"✅ Created test data for phone: {test_phone}")
        logger.info(f"   Session ID: {session_id}")
        logger.info(f"   Caller ID: {caller_id}")
        
        return test_phone, caller_id
        
    except Exception as e:
        logger.error(f"❌ Failed to create test data: {e}")
        return None, None

async def test_history_retrieval(test_phone, caller_id):
    """Test the history retrieval functionality"""
    logger.info("🔍 Testing history retrieval...")
    
    try:
        from call_transcription_storage import call_storage
        
        # Test caller profile lookup
        caller_profile = await call_storage.get_caller_by_phone(test_phone)
        if caller_profile:
            logger.info(f"✅ Found caller profile:")
            logger.info(f"   Total calls: {caller_profile.total_calls}")
            logger.info(f"   Total turns: {caller_profile.total_conversation_turns}")
        else:
            logger.warning("⚠️ No caller profile found")
            return False
        
        # Test conversation history retrieval
        history = await call_storage.get_caller_conversation_history(caller_id, limit=10)
        if history:
            logger.info(f"✅ Retrieved {len(history)} conversation items")
            logger.info(f"   Latest: {history[0].role}: {history[0].content[:50]}...")
        else:
            logger.warning("⚠️ No conversation history found")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ History retrieval test failed: {e}")
        return False

async def test_openai_integration():
    """Test OpenAI integration for history analysis"""
    logger.info("🤖 Testing OpenAI integration...")
    
    try:
        from config import config
        import openai
        
        if not config.openai_api_key:
            logger.warning("⚠️ No OpenAI API key configured")
            return False
        
        client = openai.AsyncOpenAI(api_key=config.openai_api_key)
        
        # Test simple completion
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'OpenAI integration test successful'"}],
                max_tokens=10
            ),
            timeout=5.0
        )
        
        result = response.choices[0].message.content.strip()
        logger.info(f"✅ OpenAI integration working: {result}")
        
        return True
        
    except asyncio.TimeoutError:
        logger.error("❌ OpenAI API timeout")
        return False
    except Exception as e:
        logger.error(f"❌ OpenAI integration failed: {e}")
        return False

def check_file_structure():
    """Check if required files exist"""
    logger.info("📁 Checking file structure...")
    
    required_files = [
        "config.py",
        "call_transcription_storage.py",
        "simple_rag_v2.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
        else:
            logger.info(f"✅ Found {file}")
    
    if missing_files:
        logger.error(f"❌ Missing required files: {missing_files}")
        return False
    
    return True

async def main():
    """Run setup and tests"""
    logger.info("🚀 CONVERSATION HISTORY INTEGRATION SETUP")
    logger.info("=" * 60)
    
    success = True
    
    # Step 1: Check file structure
    logger.info("\nStep 1: File Structure Check")
    logger.info("-" * 30)
    if not check_file_structure():
        logger.error("❌ File structure check failed")
        success = False
    
    # Step 2: Check dependencies
    logger.info("\nStep 2: Dependency Check")
    logger.info("-" * 30)
    if not await check_dependencies():
        logger.error("❌ Dependency check failed")
        success = False
    
    # Step 3: Check database
    logger.info("\nStep 3: Database Check")
    logger.info("-" * 30)
    if not await check_database():
        logger.error("❌ Database check failed")
        success = False
    
    # Step 4: Test OpenAI integration
    logger.info("\nStep 4: OpenAI Integration Test")
    logger.info("-" * 30)
    if not await test_openai_integration():
        logger.error("❌ OpenAI integration test failed")
        success = False
    
    # Step 5: Create and test data
    logger.info("\nStep 5: Test Data Creation and Retrieval")
    logger.info("-" * 30)
    test_phone, caller_id = await create_test_data()
    if test_phone and caller_id:
        if not await test_history_retrieval(test_phone, caller_id):
            logger.error("❌ History retrieval test failed")
            success = False
    else:
        logger.error("❌ Test data creation failed")
        success = False
    
    # Final results
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("🎉 SETUP COMPLETED SUCCESSFULLY!")
        logger.info("\n✅ Your system is ready for conversation history integration")
        logger.info("\n📋 NEXT STEPS:")
        logger.info("   1. Save the complete main_multiagent_improved_with_history.py file")
        logger.info("   2. Run: python test_history_integration.py")
        logger.info("   3. If tests pass, run: python main_multiagent_improved_with_history.py dev")
        logger.info("   4. Test with a returning caller using the phone number: " + test_phone)
        
        logger.info("\n💭 EXPECTED BEHAVIOR:")
        logger.info("   - New callers: Standard greeting")
        logger.info("   - Returning callers: Personalized greeting based on history")
        logger.info("   - Example: 'Welcome back! I hope that battery service worked out well. How can I help today?'")
        
    else:
        logger.error("❌ SETUP FAILED!")
        logger.error("\n🔧 TROUBLESHOOTING:")
        logger.error("   1. Ensure all required files are present")
        logger.error("   2. Check OpenAI API key in .env file")
        logger.error("   3. Verify database is accessible")
        logger.error("   4. Check internet connection for OpenAI API")
        logger.error("   5. Run: pip install openai")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Setup cancelled by user")
    except Exception as e:
        logger.error(f"\n❌ Setup failed: {e}")
        sys.exit(1)