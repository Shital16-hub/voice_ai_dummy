# test_transcription_system.py
"""
Test the call transcription system with sample data
Demonstrates how the system works with dummy calls
"""
import asyncio
import logging
import json
import time
from pathlib import Path
from call_transcription_storage import call_storage
from mongodb_migration_helper import MongoDBMigrationHelper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_sample_call_data():
    """Create sample call data to demonstrate the system"""
    
    logger.info("🎯 Creating sample call transcription data...")
    
    # Sample call 1: New customer - Towing service
    session_id_1, caller_id_1 = await call_storage.start_call_session(
        phone_number="+15551234567",
        session_metadata={
            "call_type": "inbound",
            "trunk_id": "trunk_001",
            "call_id": "call_001"
        }
    )
    
    logger.info(f"📞 Started sample call 1: {session_id_1}")
    
    # Simulate conversation transcription
    transcription_data_1 = [
        ("user", "Hello I need help my car broke down", True),
        ("agent", "Roadside assistance, this is Mark. How can I help you today?", True),
        ("user", "My car won't start and I'm stranded", True),
        ("agent", "I'm sorry to hear that. Could you please provide your full name?", True),
        ("user", "My name is John Smith", True),
        ("agent", "Thank you John. Could you provide a good phone number where we can reach you?", True),
        ("user", "555-123-4567", True),
        ("agent", "What is the exact location of your vehicle?", True),
        ("user", "I'm at 123 Main Street in downtown", True),
        ("agent", "Could you tell me the year, make, and model of your vehicle?", True),
        ("user", "It's a 2018 Honda Civic", True),
        ("agent", "What type of service do you need today?", True),
        ("user", "I think I need a tow", True),
        ("agent", "I'll connect you with our towing specialist.", True)
    ]
    
    # Save transcription segments
    for speaker, text, is_final in transcription_data_1:
        await call_storage.save_transcription_segment(
            session_id=session_id_1,
            caller_id=caller_id_1,
            speaker=speaker,
            text=text,
            is_final=is_final,
            confidence=0.95 if is_final else 0.7
        )
        await asyncio.sleep(0.1)  # Simulate real-time spacing
    
    # Save conversation items
    conversation_items_1 = [
        ("user", "Hello I need help my car broke down"),
        ("assistant", "Roadside assistance, this is Mark. How can I help you today?"),
        ("user", "My car won't start and I'm stranded"),
        ("assistant", "I'm sorry to hear that. Could you please provide your full name?"),
        ("user", "My name is John Smith"),
        ("assistant", "Thank you John. Could you provide a good phone number where we can reach you?"),
        ("user", "555-123-4567"),
        ("assistant", "What is the exact location of your vehicle?"),
        ("user", "I'm at 123 Main Street in downtown"),
        ("assistant", "Could you tell me the year, make, and model of your vehicle?"),
        ("user", "It's a 2018 Honda Civic"),
        ("assistant", "What type of service do you need today?"),
        ("user", "I think I need a tow"),
        ("assistant", "I'll connect you with our towing specialist.")
    ]
    
    for role, content in conversation_items_1:
        await call_storage.save_conversation_item(
            session_id=session_id_1,
            caller_id=caller_id_1,
            role=role,
            content=content,
            metadata={"urgency": "normal", "service_type": "towing"}
        )
    
    # End the call
    await call_storage.end_call_session(session_id_1)
    
    # Wait a bit to simulate time between calls
    await asyncio.sleep(1)
    
    # Sample call 2: Returning customer - Battery service
    session_id_2, caller_id_2 = await call_storage.start_call_session(
        phone_number="+15551234567",  # Same phone number - returning customer!
        session_metadata={
            "call_type": "inbound",
            "trunk_id": "trunk_001", 
            "call_id": "call_002"
        }
    )
    
    logger.info(f"📞 Started sample call 2 (returning customer): {session_id_2}")
    
    # Simulate returning customer conversation
    transcription_data_2 = [
        ("user", "Hi this is John again", True),
        ("agent", "Welcome back John! I see you've called us before. How can I help you today?", True),
        ("user", "My battery is dead this time", True),
        ("agent", "I understand. Are you at the same location as last time?", True),
        ("user", "No I'm at work now, 456 Oak Avenue", True),
        ("agent", "Got it. Same Honda Civic?", True),
        ("user", "Yes that's right", True),
        ("agent", "I'll connect you with our battery specialist.", True)
    ]
    
    for speaker, text, is_final in transcription_data_2:
        await call_storage.save_transcription_segment(
            session_id=session_id_2,
            caller_id=caller_id_2,
            speaker=speaker,
            text=text,
            is_final=is_final,
            confidence=0.93
        )
    
    conversation_items_2 = [
        ("user", "Hi this is John again"),
        ("assistant", "Welcome back John! I see you've called us before. How can I help you today?"),
        ("user", "My battery is dead this time"),
        ("assistant", "I understand. Are you at the same location as last time?"),
        ("user", "No I'm at work now, 456 Oak Avenue"),
        ("assistant", "Got it. Same Honda Civic?"),
        ("user", "Yes that's right"),
        ("assistant", "I'll connect you with our battery specialist.")
    ]
    
    for role, content in conversation_items_2:
        await call_storage.save_conversation_item(
            session_id=session_id_2,
            caller_id=caller_id_2,
            role=role,
            content=content,
            metadata={"urgency": "normal", "service_type": "battery"}
        )
    
    await call_storage.end_call_session(session_id_2)
    
    logger.info("✅ Sample call data created successfully")
    return [session_id_1, session_id_2], [caller_id_1, caller_id_2]

async def test_caller_recognition():
    """Test caller recognition functionality"""
    
    logger.info("🔍 Testing caller recognition...")
    
    # Test with existing phone number
    caller_profile = await call_storage.get_caller_by_phone("+15551234567")
    
    if caller_profile:
        logger.info(f"✅ Caller found: {caller_profile.caller_id}")
        logger.info(f"   Total calls: {caller_profile.total_calls}")
        logger.info(f"   Total conversation turns: {caller_profile.total_conversation_turns}")
        
        # Get conversation history
        history = await call_storage.get_caller_conversation_history(
            caller_profile.caller_id, limit=10
        )
        
        logger.info(f"📚 Conversation history: {len(history)} items")
        for item in history[-3:]:  # Show last 3 items
            logger.info(f"   {item.role}: {item.content[:50]}...")
    else:
        logger.warning("⚠️ No caller found with that phone number")

async def test_session_export():
    """Test MongoDB export functionality"""
    
    logger.info("📤 Testing MongoDB export...")
    
    # Get recent sessions
    recent_sessions = await call_storage.get_recent_sessions(5)
    
    if recent_sessions:
        # Export first session
        session_id = recent_sessions[0].session_id
        mongodb_doc = await call_storage.export_for_mongodb(session_id)
        
        if mongodb_doc:
            logger.info(f"✅ Exported session {session_id} for MongoDB")
            logger.info(f"   Conversation items: {len(mongodb_doc.get('conversation_items', []))}")
            logger.info(f"   Transcription segments: {len(mongodb_doc.get('transcription_segments', []))}")
            
            # Save sample to file for inspection
            export_path = Path("sample_mongodb_export.json")
            with open(export_path, 'w') as f:
                # Convert datetime objects for JSON serialization
                json_doc = convert_datetime_to_string(mongodb_doc)
                json.dump(json_doc, f, indent=2)
            
            logger.info(f"💾 Sample MongoDB document saved to: {export_path}")
        else:
            logger.warning("⚠️ Export returned empty document")
    else:
        logger.warning("⚠️ No recent sessions found")

def convert_datetime_to_string(obj):
    """Convert datetime objects to ISO strings for JSON"""
    from datetime import datetime
    
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_datetime_to_string(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetime_to_string(item) for item in obj]
    else:
        return obj

async def test_migration_helper():
    """Test the MongoDB migration helper"""
    
    logger.info("🔄 Testing MongoDB migration helper...")
    
    migration_helper = MongoDBMigrationHelper()
    
    # Generate migration report
    report = await migration_helper.generate_migration_report()
    
    if "error" not in report:
        stats = report["migration_report"]["data_statistics"]
        logger.info(f"📊 Migration report generated:")
        logger.info(f"   Total sessions: {stats['total_sessions']}")
        logger.info(f"   Completed sessions: {stats['completed_sessions']}")
        logger.info(f"   Unique callers: {stats['unique_callers']}")
        
        # Test batch migration (to JSON files)
        results = await migration_helper.batch_migrate_recent_sessions(3)
        logger.info(f"📄 Batch migration test: {results.get('migrated', 0)} sessions exported")
    else:
        logger.error(f"❌ Migration report failed: {report['error']}")

async def display_call_statistics():
    """Display comprehensive call statistics"""
    
    logger.info("📊 Call Transcription System Statistics")
    logger.info("=" * 50)
    
    # Get recent sessions
    recent_sessions = await call_storage.get_recent_sessions(20)
    
    if recent_sessions:
        total_sessions = len(recent_sessions)
        completed_sessions = len([s for s in recent_sessions if s.status == "completed"])
        total_duration = sum(s.duration_seconds or 0 for s in recent_sessions)
        avg_duration = total_duration / completed_sessions if completed_sessions > 0 else 0
        
        logger.info(f"📞 Total sessions: {total_sessions}")
        logger.info(f"✅ Completed sessions: {completed_sessions}")
        logger.info(f"⏱️  Average call duration: {avg_duration:.1f} seconds")
        
        # Unique callers
        unique_callers = len(set(s.caller_id for s in recent_sessions))
        logger.info(f"👥 Unique callers: {unique_callers}")
        
        # Recent sessions details
        logger.info(f"\n📋 Recent Sessions:")
        for session in recent_sessions[:5]:
            duration_str = f"{session.duration_seconds:.1f}s" if session.duration_seconds else "ongoing"
            logger.info(f"   {session.session_id[:12]}... | {session.phone_number} | {duration_str} | {session.status}")
    else:
        logger.info("📭 No sessions found")
    
    logger.info("=" * 50)

async def main():
    """Run comprehensive transcription system test"""
    
    logger.info("🚀 Starting Call Transcription System Test")
    logger.info("=" * 60)
    
    try:
        # Create sample data
        session_ids, caller_ids = await create_sample_call_data()
        
        # Test caller recognition
        await test_caller_recognition()
        
        # Test session export
        await test_session_export()
        
        # Test migration helper
        await test_migration_helper()
        
        # Display statistics
        await display_call_statistics()
        
        logger.info("✅ All tests completed successfully!")
        logger.info("\n🎯 Next Steps:")
        logger.info("1. Review the generated sample_mongodb_export.json file")
        logger.info("2. Check mongodb_migration_report.json for migration details")
        logger.info("3. When ready for MongoDB, update mongodb_migration_helper.py")
        logger.info("4. Run your enhanced agent: python enhanced_multi_agent_with_transcription.py")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())