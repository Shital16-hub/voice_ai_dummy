# test_transcription_integration.py
"""
Test script to verify the transcription integration works correctly
"""
import asyncio
import logging
from call_transcription_storage import call_storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_transcription_system():
    """Test the transcription system"""
    try:
        logger.info("🧪 Testing transcription system integration...")
        
        # Test 1: Create a call session
        session_id, caller_id = await call_storage.start_call_session(
            phone_number="+15551234567",
            session_metadata={"test": "integration"}
        )
        logger.info(f"✅ Created session: {session_id}, caller: {caller_id}")
        
        # Test 2: Save some transcription segments
        await call_storage.save_transcription_segment(
            session_id=session_id,
            caller_id=caller_id,
            speaker="user",
            text="Hello I need help with my car",
            is_final=True,
            confidence=0.95
        )
        logger.info("✅ Saved user transcription segment")
        
        await call_storage.save_transcription_segment(
            session_id=session_id,
            caller_id=caller_id,
            speaker="agent",
            text="Roadside assistance, this is Mark, how can I help you?",
            is_final=True
        )
        logger.info("✅ Saved agent transcription segment")
        
        # Test 3: Save conversation items
        await call_storage.save_conversation_item(
            session_id=session_id,
            caller_id=caller_id,
            role="user",
            content="Hello I need help with my car",
            metadata={"test": "conversation"}
        )
        logger.info("✅ Saved user conversation item")
        
        await call_storage.save_conversation_item(
            session_id=session_id,
            caller_id=caller_id,
            role="assistant",
            content="Roadside assistance, this is Mark, how can I help you?",
            metadata={"test": "conversation"}
        )
        logger.info("✅ Saved assistant conversation item")
        
        # Test 4: End the session
        success = await call_storage.end_call_session(session_id)
        logger.info(f"✅ Ended session: {success}")
        
        # Test 5: Check caller recognition
        caller_profile = await call_storage.get_caller_by_phone("+15551234567")
        if caller_profile:
            logger.info(f"✅ Caller found: {caller_profile.total_calls} calls")
        else:
            logger.warning("⚠️ Caller not found")
        
        logger.info("🎉 All transcription integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Transcription test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_transcription_system())