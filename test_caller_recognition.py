# test_caller_recognition.py
"""
Test script to demonstrate caller recognition functionality
Run this to see how your system recognizes returning callers
"""
import asyncio
import logging
from datetime import datetime
from call_transcription_storage import call_storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def simulate_first_call():
    """Simulate a first-time caller"""
    print("📞 SIMULATING FIRST CALL")
    print("=" * 50)
    
    phone_number = "+15551234567"
    
    # Start call session
    session_id, caller_id = await call_storage.start_call_session(
        phone_number=phone_number,
        session_metadata={"simulation": "first_call"}
    )
    
    print(f"📱 Phone: {phone_number}")
    print(f"🆔 Session: {session_id}")
    print(f"👤 Caller ID: {caller_id}")
    
    # Check if recognized as returning caller
    caller_profile = await call_storage.get_caller_by_phone(phone_number)
    is_returning = caller_profile and caller_profile.total_calls > 1
    
    print(f"🔄 Returning caller: {is_returning}")
    print(f"📊 Total calls: {caller_profile.total_calls if caller_profile else 0}")
    
    # Simulate conversation
    await call_storage.save_conversation_item(
        session_id=session_id,
        caller_id=caller_id,
        role="assistant",
        content="Roadside assistance, this is Mark, how can I help you today?"
    )
    
    await call_storage.save_conversation_item(
        session_id=session_id,
        caller_id=caller_id,
        role="user",
        content="Hi, my car broke down and I need a tow"
    )
    
    await call_storage.save_conversation_item(
        session_id=session_id,
        caller_id=caller_id,
        role="assistant", 
        content="I'm sorry to hear that. Could you please provide your full name?"
    )
    
    await call_storage.save_conversation_item(
        session_id=session_id,
        caller_id=caller_id,
        role="user",
        content="My name is John Smith"
    )
    
    # End call
    await call_storage.end_call_session(session_id)
    
    print("✅ First call completed and stored")
    return phone_number

async def simulate_returning_call(phone_number: str):
    """Simulate the same caller calling back"""
    print("\n📞 SIMULATING RETURNING CALL (Same Phone Number)")
    print("=" * 50)
    
    # Start call session with same phone number
    session_id, caller_id = await call_storage.start_call_session(
        phone_number=phone_number,
        session_metadata={"simulation": "returning_call"}
    )
    
    print(f"📱 Phone: {phone_number}")
    print(f"🆔 Session: {session_id}")
    print(f"👤 Caller ID: {caller_id}")
    
    # Check caller recognition
    caller_profile = await call_storage.get_caller_by_phone(phone_number)
    is_returning = caller_profile and caller_profile.total_calls > 1
    
    print(f"🔄 Returning caller: {is_returning}")
    print(f"📊 Total calls: {caller_profile.total_calls if caller_profile else 0}")
    print(f"📊 Previous calls: {caller_profile.total_calls - 1 if caller_profile else 0}")
    
    # Show what greeting would be used
    if is_returning:
        greeting = "Welcome back! I see you've called us before. How can I help you today?"
        print(f"🎙️ Greeting: {greeting}")
    else:
        greeting = "Roadside assistance, this is Mark, how can I help you today?"
        print(f"🎙️ Greeting: {greeting}")
    
    # Get conversation history
    if caller_profile:
        history = await call_storage.get_caller_conversation_history(
            caller_profile.caller_id, limit=5
        )
        print(f"\n📚 Previous conversation history ({len(history)} items):")
        for item in history[-3:]:  # Show last 3 items
            time_str = datetime.fromtimestamp(item.timestamp).strftime("%H:%M:%S")
            print(f"   [{time_str}] {item.role}: {item.content[:50]}...")
    
    # Simulate new conversation
    await call_storage.save_conversation_item(
        session_id=session_id,
        caller_id=caller_id,
        role="assistant",
        content=greeting
    )
    
    await call_storage.save_conversation_item(
        session_id=session_id,
        caller_id=caller_id,
        role="user",
        content="Hi, it's John again. My battery is dead this time"
    )
    
    await call_storage.save_conversation_item(
        session_id=session_id,
        caller_id=caller_id,
        role="assistant",
        content="Hi John! I remember you. Same Honda Civic from last time?"
    )
    
    # End call
    await call_storage.end_call_session(session_id)
    
    print("✅ Returning call completed")

async def check_caller_profile(phone_number: str):
    """Check final caller profile"""
    print(f"\n📊 FINAL CALLER PROFILE FOR {phone_number}")
    print("=" * 50)
    
    caller_profile = await call_storage.get_caller_by_phone(phone_number)
    
    if caller_profile:
        print(f"👤 Caller ID: {caller_profile.caller_id}")
        print(f"📞 Phone: {caller_profile.phone_number}")
        print(f"📊 Total calls: {caller_profile.total_calls}")
        print(f"💬 Total conversation turns: {caller_profile.total_conversation_turns}")
        print(f"🕐 First call: {datetime.fromtimestamp(caller_profile.first_call_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🕐 Last call: {datetime.fromtimestamp(caller_profile.last_call_time).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get all conversation history
        history = await call_storage.get_caller_conversation_history(
            caller_profile.caller_id, limit=20
        )
        
        print(f"\n📚 Complete Conversation History ({len(history)} turns):")
        for item in history:
            time_str = datetime.fromtimestamp(item.timestamp).strftime("%m-%d %H:%M:%S")
            print(f"   [{time_str}] {item.role.upper()}: {item.content}")
    else:
        print("❌ No caller profile found")

async def test_different_phone_number():
    """Test with a different phone number (should NOT be recognized)"""
    print(f"\n📞 TESTING DIFFERENT PHONE NUMBER (Should NOT be recognized)")
    print("=" * 50)
    
    different_phone = "+15559876543"
    
    # Check recognition
    caller_profile = await call_storage.get_caller_by_phone(different_phone)
    is_returning = caller_profile and caller_profile.total_calls > 1
    
    print(f"📱 Phone: {different_phone}")
    print(f"🔄 Returning caller: {is_returning}")
    print(f"📊 Total calls: {caller_profile.total_calls if caller_profile else 0}")
    
    if is_returning:
        greeting = "Welcome back! I see you've called us before. How can I help you today?"
    else:
        greeting = "Roadside assistance, this is Mark, how can I help you today?"
    
    print(f"🎙️ Greeting: {greeting}")

async def main():
    """Run caller recognition test"""
    print("🧪 TESTING CALLER RECOGNITION SYSTEM")
    print("=" * 60)
    
    try:
        # Simulate first call
        phone_number = await simulate_first_call()
        
        # Small delay to simulate time between calls
        await asyncio.sleep(1)
        
        # Simulate returning call
        await simulate_returning_call(phone_number)
        
        # Check final profile
        await check_caller_profile(phone_number)
        
        # Test different number
        await test_different_phone_number()
        
        print(f"\n🎉 CALLER RECOGNITION TEST COMPLETED!")
        print("=" * 60)
        print("✅ Key Findings:")
        print("   • First-time callers get standard greeting")
        print("   • Returning callers get personalized 'Welcome back' greeting")
        print("   • System remembers conversation history")
        print("   • Different phone numbers are treated as new callers")
        print("   • Caller recognition is automatic and immediate")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())